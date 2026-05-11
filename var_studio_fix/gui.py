"""PyQt5 GUI for VAR Replay Studio.

Layout:
  ┌──────────────────────────────────────────────────────────────┐
  │ Toolbar                                                      │
  ├───────────────────────────────┬──────────────────────────────┤
  │                               │  Tools (sliders)             │
  │       Video stage             │  ─ sharpen / blur / B / C    │
  │       (with overlay)          │  ─ zoom / harmonics / cutoff │
  │                               │  ─ Auto-calibrate · offside  │
  │                               ├──────────────────────────────┤
  │                               │  2D FFT  | Magnitude | Phase │
  ├───────────────────────────────┴──────────────────────────────┤
  │ Transport: ▶ ⏸  ◀◀  ▶▶   speed   timeline                    │
  ├──────────────────────────────────────────────────────────────┤
  │ Video Fourier  : Original | Denoised | Clear  (3 separate)   │
  │ Audio Fourier  : Original | Denoised | Clear  (3 separate)   │
  └──────────────────────────────────────────────────────────────┘
"""
from __future__ import annotations

import sys
import time
from typing import Optional

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from .video_engine import VideoEngine
from .audio_engine import AudioEngine
from .processing import apply_pipeline, to_luminance
from .fft_tools import fft2_spectra, analyze_1d
from .auto_calibrate import detect_pitch_corners
from .homography import homography
from .offside import draw_offside, PITCH_DST
from .hawkeye import render_hawkeye
from .store import STORE


pg.setConfigOptions(antialias=True, background="#0c0f12", foreground="#d6e2ee")


# ---------------------------------------------------------------------------
def ndarray_to_qimage(bgr: np.ndarray) -> QtGui.QImage:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = rgb.shape
    return QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()


class VideoStage(QtWidgets.QLabel):
    """Click-to-pick widget for calibration / attacker / defender selection."""

    pickMode = QtCore.pyqtSignal(str, float, float)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(480, 270)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setStyleSheet("background:#000; border:1px solid #1d2530;")
        self._mode: Optional[str] = None
        self._calib_pts: list[tuple[float, float]] = []
        self._img_w = 1
        self._img_h = 1
        self._draw_w = 1
        self._draw_h = 1

    def set_mode(self, mode: Optional[str]) -> None:
        self._mode = mode
        if mode == "calibrate":
            self._calib_pts.clear()
        if mode == "drawline":
            self._calib_pts.clear()
        self.setCursor(QtCore.Qt.CrossCursor if mode else QtCore.Qt.ArrowCursor)

    def update_frame(self, bgr: np.ndarray) -> None:
        self._img_w, self._img_h = bgr.shape[1], bgr.shape[0]
        qimg = ndarray_to_qimage(bgr)
        pm = QtGui.QPixmap.fromImage(qimg).scaled(
            self.size(), QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation)
        self._draw_w, self._draw_h = pm.width(), pm.height()
        self.setPixmap(pm)

    def _to_image_coords(self, ev) -> tuple[float, float]:
        # Pixmap is centered with KeepAspectRatio
        ox = (self.width() - self._draw_w) / 2
        oy = (self.height() - self._draw_h) / 2
        x = (ev.x() - ox) / max(1, self._draw_w) * self._img_w
        y = (ev.y() - oy) / max(1, self._draw_h) * self._img_h
        return float(x), float(y)

    def mousePressEvent(self, ev: QtGui.QMouseEvent) -> None:
        if not self._mode:
            return
        x, y = self._to_image_coords(ev)
        if not (0 <= x <= self._img_w and 0 <= y <= self._img_h):
            return
        self.pickMode.emit(self._mode, x, y)


class TripletGraph(QtWidgets.QWidget):
    """3 separate plots side by side: Original / Denoised / Clear.
    Each plot stacks the time signal on top and its magnitude spectrum
    underneath, so nothing overlaps.
    """

    def __init__(self, title: str):
        super().__init__()
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lbl = QtWidgets.QLabel(f"<b>{title}</b>")
        lbl.setStyleSheet("color:#cfe;")
        lay.addWidget(lbl)

        row = QtWidgets.QHBoxLayout()
        lay.addLayout(row, 1)

        self.plots = {}
        colors = {"Original": "#9aa4b2", "Denoised": "#5ee0c4", "Clear": "#ffd66b"}
        for name in ("Original", "Denoised", "Clear"):
            box = QtWidgets.QVBoxLayout()
            top = pg.PlotWidget(title=f"{name} — signal")
            bot = pg.PlotWidget(title=f"{name} — |FFT|")
            for pw in (top, bot):
                pw.showGrid(x=True, y=True, alpha=0.2)
                pw.setMouseEnabled(False, False)
            curve_t = top.plot(pen=pg.mkPen(colors[name], width=2))
            curve_f = bot.plot(pen=pg.mkPen(colors[name], width=1))
            box.addWidget(top)
            box.addWidget(bot)
            w = QtWidgets.QWidget()
            w.setLayout(box)
            row.addWidget(w)
            self.plots[name] = (curve_t, curve_f, top, bot)

    def update_triplet(self, trip):
        pairs = (("Original", trip.original, trip.mag_original),
                 ("Denoised", trip.denoised, trip.mag_denoised),
                 ("Clear",    trip.clear,    trip.mag_clear))
        for name, sig, mag in pairs:
            ct, cf, *_ = self.plots[name]
            ct.setData(trip.t, sig)
            cf.setData(trip.freqs, mag)


class SpectrumPanel(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        self.mag_lbl = QtWidgets.QLabel("Magnitude")
        self.pha_lbl = QtWidgets.QLabel("Phase")
        for lbl in (self.mag_lbl, self.pha_lbl):
            lbl.setAlignment(QtCore.Qt.AlignCenter)
            lbl.setMinimumSize(120, 120)
            lbl.setStyleSheet("background:#000;border:1px solid #1d2530;")
            lay.addWidget(lbl)

    def update_image(self, gray: np.ndarray):
        mag, pha = fft2_spectra(gray, size=128)
        for img, lbl in ((mag, self.mag_lbl), (pha, self.pha_lbl)):
            qimg = ndarray_to_qimage(img)
            lbl.setPixmap(QtGui.QPixmap.fromImage(qimg).scaled(
                lbl.size(), QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation))


# ---------------------------------------------------------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VAR Replay Studio — Python MVP")
        self.resize(1000, 700)
        self.setStyleSheet("""
            QMainWindow,QWidget{background:#0c0f12;color:#d6e2ee;
                font-family:'Segoe UI','Helvetica',Arial;}
            QPushButton{background:#1a2230;border:1px solid #2a3548;
                padding:6px 12px;border-radius:6px;color:#dfe9f5;}
            QPushButton:hover{background:#243149;}
            QLineEdit{background:#0f1620;border:1px solid #25304a;padding:5px;
                border-radius:4px;}
            QSlider::groove:horizontal{height:4px;background:#1d2738;}
            QSlider::handle:horizontal{background:#7ee0b0;width:14px;
                margin:-6px 0;border-radius:7px;}
            QGroupBox{border:1px solid #1d2738;border-radius:6px;margin-top:8px;
                padding-top:14px;}
            QGroupBox::title{subcontrol-origin:margin;left:10px;color:#7ee0b0;}
        """)

        self.video = VideoEngine()
        self.audio = AudioEngine()
        self.t = 0.0
        self.playing = False
        self.attacker = None
        self.defender = None
        self.H = None
        self.manual_line: Optional[tuple[tuple[float, float], tuple[float, float]]] = None
        self._draw_pts: list[tuple[float, float]] = []
        self.goal_side = "right"
        self._tick_count = 0
        self._trail: list[tuple[float, float]] = []
        self._last_render_size = (0, 0)
        self._wants_audio_restart = False  # set when filter recomputes while playing

        # Central widget
        # Replace the old Central widget section with this:
        self.scroll_area = QtWidgets.QScrollArea()
        self.setCentralWidget(self.scroll_area)
        self.scroll_area.setWidgetResizable(True)

        central = QtWidgets.QWidget()
        self.scroll_area.setWidget(central)
        outer = QtWidgets.QVBoxLayout(central)

        # Toolbar
        outer.addLayout(self._build_toolbar())

        # Stage row (video | tools | spectra)
        stage_row = QtWidgets.QHBoxLayout()
        outer.addLayout(stage_row, 1)

        self.stage = VideoStage()
        self.stage.pickMode.connect(self.on_pick)
        stage_row.addWidget(self.stage, 3)

        right = QtWidgets.QVBoxLayout()
        right.addWidget(self._build_tools(), 0)
        self.hawk_lbl = QtWidgets.QLabel()
        self.hawk_lbl.setMinimumSize(200, 140)
        self.hawk_lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.hawk_lbl.setStyleSheet("background:#000;border:1px solid #1d2530;")
        right.addWidget(self._wrap_group("Hawk-Eye (top-down)", self.hawk_lbl), 1)
        rw = QtWidgets.QWidget(); rw.setLayout(right)
        stage_row.addWidget(rw, 2)

        # Transport
        outer.addLayout(self._build_transport())

        # Fourier panel
        self.audio_triplet = TripletGraph("Audio — Fourier")
        outer.addWidget(self.audio_triplet, 1)

        # Render timer
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(33)

        self._last_tick = time.time()

    # -- builders ------------------------------------------------------------
    def _wrap_group(self, title, widget):
        g = QtWidgets.QGroupBox(title)
        v = QtWidgets.QVBoxLayout(g)
        v.addWidget(widget)
        return g

    def _build_toolbar(self):
        bar = QtWidgets.QHBoxLayout()
        b_open = QtWidgets.QPushButton("Open file…")
        b_open.clicked.connect(self.open_file)
        bar.addWidget(b_open)

        self.url_in = QtWidgets.QLineEdit()
        self.url_in.setPlaceholderText("Paste video URL (HTTP MP4)…")
        bar.addWidget(self.url_in, 1)
        b_url = QtWidgets.QPushButton("Load URL")
        b_url.clicked.connect(self.load_url)
        bar.addWidget(b_url)

        b_book = QtWidgets.QPushButton("Bookmark")
        b_book.clicked.connect(lambda: STORE.add_bookmark(self.t))
        bar.addWidget(b_book)
        return bar

    def _build_tools(self):
        g = QtWidgets.QGroupBox("Tools")
        f = QtWidgets.QFormLayout(g)

        def slider(lo, hi, step, val, cb):
            s = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            s.setRange(int(lo / step), int(hi / step))
            s.setValue(int(val / step))
            s.valueChanged.connect(lambda v: cb(v * step))
            return s

        f.addRow("Sharpen", slider(0, 2, 0.05, 0,
                 lambda v: STORE.update(sharpen=v)))
        f.addRow("Blur (σ)", slider(0, 10, 0.1, 0,
                 lambda v: STORE.update(blur=v)))
        f.addRow("Brightness", slider(-1, 1, 0.05, 0,
                 lambda v: STORE.update(brightness=v)))
        f.addRow("Contrast", slider(0.2, 3.0, 0.05, 1.0,
                 lambda v: STORE.update(contrast=v)))
        f.addRow("Zoom", slider(1, 8, 0.1, 1,
                 lambda v: STORE.update(zoom=v)))
        f.addRow("Pan X", slider(-1, 1, 0.05, 0,
                 lambda v: STORE.update(pan_x=v)))
        f.addRow("Pan Y", slider(-1, 1, 0.05, 0,
                 lambda v: STORE.update(pan_y=v)))
        f.addRow("Video Scale", slider(1.0, 4.0, 0.25, 1.0,
                 lambda v: STORE.update(output_scale=v)))
        f.addRow("Volume", slider(0.0, 1.0, 0.05, 1.0,
                 lambda v: self._set_volume(v)))
        f.addRow("Harmonics K", slider(1, 64, 1, 16,
                 lambda v: STORE.update(harmonics_k=int(v))))
        f.addRow("Denoise cutoff", slider(0.01, 0.5, 0.01, 0.10,
                 lambda v: STORE.update(denoise_cutoff=v)))
        f.addRow("🔊 Denoise Low (Hz)", slider(0.001, 0.1, 0.005, 0.01,
                 lambda v: self._update_denoise(denoise_low_cutoff=v)))
        f.addRow("🔊 Denoise High (Hz)", slider(0.05, 0.5, 0.01, 0.20,
                 lambda v: self._update_denoise(denoise_high_cutoff=v)))
        f.addRow("🔊 Noise Gate", slider(0, 1, 0.05, 0,
                 lambda v: STORE.update(noise_gate=v)))

        row = QtWidgets.QHBoxLayout()
        b_auto = QtWidgets.QPushButton("✨ Auto-calibrate pitch")
        b_auto.clicked.connect(self.auto_calibrate)
        row.addWidget(b_auto)
        b_cal = QtWidgets.QPushButton("Pick 4 corners")
        b_cal.clicked.connect(lambda: self.stage.set_mode("calibrate"))
        row.addWidget(b_cal)
        f.addRow(row)

        row2 = QtWidgets.QHBoxLayout()
        b_a = QtWidgets.QPushButton("Mark attacker")
        b_a.clicked.connect(lambda: self.stage.set_mode("attacker"))
        row2.addWidget(b_a)
        b_d = QtWidgets.QPushButton("Mark defender")
        b_d.clicked.connect(lambda: self.stage.set_mode("defender"))
        row2.addWidget(b_d)
        f.addRow(row2)

        row3 = QtWidgets.QHBoxLayout()
        b_draw = QtWidgets.QPushButton("✏ Draw offside line (2 clicks)")
        b_draw.clicked.connect(self.start_draw_line)
        row3.addWidget(b_draw)
        f.addRow(row3)

        row4 = QtWidgets.QHBoxLayout()
        row4.addWidget(QtWidgets.QLabel("Attacking goal:"))
        self.goal_box = QtWidgets.QComboBox()
        self.goal_box.addItems(["right", "left"])
        self.goal_box.currentTextChanged.connect(
            lambda s: setattr(self, "goal_side", s))
        row4.addWidget(self.goal_box)
        f.addRow(row4)

        b_clr = QtWidgets.QPushButton("Clear offside")
        b_clr.clicked.connect(self.clear_offside)
        f.addRow(b_clr)

        self.verdict_lbl = QtWidgets.QLabel("—")
        self.verdict_lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.verdict_lbl.setStyleSheet("font-size:18px;font-weight:bold;")
        f.addRow("Verdict", self.verdict_lbl)
        return g

    def _build_transport(self):
        bar = QtWidgets.QHBoxLayout()
        self.b_play = QtWidgets.QPushButton("▶")
        self.b_play.clicked.connect(self.toggle_play)
        bar.addWidget(self.b_play)
        b_prev = QtWidgets.QPushButton("◀ frame")
        b_prev.clicked.connect(lambda: self.step_frame(-1))
        bar.addWidget(b_prev)
        b_next = QtWidgets.QPushButton("frame ▶")
        b_next.clicked.connect(lambda: self.step_frame(1))
        bar.addWidget(b_next)

        bar.addWidget(QtWidgets.QLabel("Speed"))
        self.speed_box = QtWidgets.QComboBox()
        for v in (0.1, 0.25, 0.5, 1.0, 1.5, 2.0):
            self.speed_box.addItem(f"{v}x", v)
        self.speed_box.setCurrentIndex(3)
        bar.addWidget(self.speed_box)

        self.scrub = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.scrub.setRange(0, 1000)
        self.scrub.sliderMoved.connect(self.on_scrub)
        bar.addWidget(self.scrub, 1)

        self.time_lbl = QtWidgets.QLabel("0.00 / 0.00 s")
        bar.addWidget(self.time_lbl)
        return bar

    # -- actions -------------------------------------------------------------
    def open_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open video", "", "Video (*.mp4 *.mov *.mkv *.webm *.avi)")
        if path:
            self._load(path)

    def load_url(self):
        url = self.url_in.text().strip()
        if url:
            self._load(url)

    def _load(self, path: str):
        try:
            self.video.open(path)
            self.audio.load(path)
            self.audio.precompute_denoise(STORE.settings.denoise_low_cutoff,
                                          STORE.settings.denoise_high_cutoff)
            self.t = 0.0
            self.playing = False
            self.b_play.setText("▶")
            self.attacker = self.defender = None
            self.H = None
            self.verdict_lbl.setText("—")
            self.audio.stop()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load failed", str(e))

    def _update_denoise(self, **kwargs) -> None:
        STORE.update(**kwargs)
        self.audio.precompute_denoise(STORE.settings.denoise_low_cutoff,
                                      STORE.settings.denoise_high_cutoff)
        if self.playing:
            self._wants_audio_restart = True  # tick() restarts once filter is ready

    def _set_volume(self, vol: float) -> None:
        STORE.update(volume=vol)
        self.audio.set_volume(vol)

    def toggle_play(self):
        self.playing = not self.playing
        self.b_play.setText("⏸" if self.playing else "▶")
        if self.playing:
            self.audio.play(self.t,
                            denoise=True,
                            low_cutoff=STORE.settings.denoise_low_cutoff,
                            high_cutoff=STORE.settings.denoise_high_cutoff)
            self._last_tick = time.time()
        else:
            self.audio.stop()
            self._last_tick = time.time()

    def step_frame(self, delta: int):
        self.playing = False
        self.b_play.setText("▶")
        self.audio.stop()
        if self.video.loaded:
            self.t = max(0.0, self.t + delta / self.video.fps)

    def on_scrub(self, v):
        if self.video.loaded:
            self.t = (v / 1000.0) * self.video.duration
            self.playing = False
            self.b_play.setText("▶")
            self.audio.stop()

    def on_pick(self, mode: str, x: float, y: float):
        if mode == "calibrate":
            self.stage._calib_pts.append((x, y))
            if len(self.stage._calib_pts) == 4:
                self.H = homography(self.stage._calib_pts, PITCH_DST)
                self.stage.set_mode(None)
                QtWidgets.QMessageBox.information(
                    self, "Calibration", "Homography set.")
        elif mode == "drawline":
            self._draw_pts.append((x, y))
            if len(self._draw_pts) == 2:
                self.manual_line = (self._draw_pts[0], self._draw_pts[1])
                self._draw_pts = []
                self.stage.set_mode(None)
        elif mode == "attacker":
            self.attacker = (x, y); self.stage.set_mode(None)
        elif mode == "defender":
            self.defender = (x, y); self.stage.set_mode(None)

    def start_draw_line(self):
        self._draw_pts = []
        self.manual_line = None
        self.stage.set_mode("drawline")

    def auto_calibrate(self):
        frame = self.video.read_at(self.t)
        if frame is None:
            return
        pts = detect_pitch_corners(frame)
        if pts is None:
            QtWidgets.QMessageBox.warning(
                self, "Auto-calibrate",
                "Could not detect 4 strong pitch lines on this frame.")
            return
        self.H = homography(pts, PITCH_DST)
        QtWidgets.QMessageBox.information(
            self, "Auto-calibrate", "Pitch corners detected ✓")

    def clear_offside(self):
        self.attacker = self.defender = None
        self.manual_line = None
        self._draw_pts = []
        self.verdict_lbl.setText("—")
        self.verdict_lbl.setStyleSheet("font-size:18px;font-weight:bold;")

    # -- main loop -----------------------------------------------------------
    def _render_current(self) -> Optional[np.ndarray]:
        if not self.video.loaded:
            return None
        frame = self.video.read_at(self.t)
        if frame is None:
            return None
        out = apply_pipeline(frame, STORE.settings)
        # preview while drawing the manual line
        if self.stage._mode == "drawline":
            # Show field boundary guide
            h, w = out.shape[:2]
            margin = 30
            cv2.rectangle(out, (margin, margin), (w - margin, h - margin),
                         (100, 100, 150), 1, cv2.LINE_AA)
            cv2.putText(out, "Offside line will be clipped to field",
                        (margin + 5, margin - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (100, 200, 200), 1, cv2.LINE_AA)
            
            if len(self._draw_pts) == 1:
                p = self._draw_pts[0]
                cv2.circle(out, (int(p[0]), int(p[1])), 6, (255, 80, 80), -1)
                cv2.putText(out, "click 2nd point",
                            (int(p[0]) + 10, int(p[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 80, 80), 2, cv2.LINE_AA)
        verdict = draw_offside(out, self.H, self.attacker, self.defender,
                               manual_line=self.manual_line,
                               goal_side=self.goal_side)
        if verdict:
            self.verdict_lbl.setText(verdict)
            self.verdict_lbl.setStyleSheet(
                "font-size:18px;font-weight:bold;color:" +
                ("#ff5050" if verdict == "OFFSIDE" else "#5eff9a"))
        elif self.attacker is None and self.defender is None and self.manual_line is None:
            self.verdict_lbl.setText("—")
        # bookmark ticks
        if STORE.bookmarks and self.video.duration > 0:
            for tb in STORE.bookmarks:
                x = int(tb / self.video.duration * out.shape[1])
                cv2.line(out, (x, 0), (x, 8), (0, 220, 255), 2)
        return out

    def tick(self):
        now = time.time()
        dt = now - self._last_tick
        self._last_tick = now

        # Seamlessly apply new denoise filter once background FFT finishes
        if self._wants_audio_restart and self.audio._filter_ready.is_set():
            self._wants_audio_restart = False
            self.audio.play(self.t,
                            denoise=True,
                            low_cutoff=STORE.settings.denoise_low_cutoff,
                            high_cutoff=STORE.settings.denoise_high_cutoff)
            self._last_tick = time.time()

        if self.playing and self.video.loaded:
            self.t += dt * self.speed_box.currentData()
            if self.t >= self.video.duration:
                self.t = 0.0
                self.audio.play(0.0,
                                denoise=True,
                                low_cutoff=STORE.settings.denoise_low_cutoff,
                                high_cutoff=STORE.settings.denoise_high_cutoff)
                self._last_tick = time.time()

        out = self._render_current()
        if out is None:
            return
        self.stage.update_frame(out)
        self._tick_count += 1

        # Update scrubber + time label
        if self.video.duration > 0:
            self.scrub.blockSignals(True)
            self.scrub.setValue(int(self.t / self.video.duration * 1000))
            self.scrub.blockSignals(False)
            self.time_lbl.setText(f"{self.t:5.2f} / {self.video.duration:5.2f} s")

        heavy = (self._tick_count % 3 == 0) or not self.playing

        # Audio triplet (downsampled tick: ~10 Hz while playing)
        if heavy:
            win = self.audio.window(self.t, n=2048)
            if win is not None and win.size:
                trip_a = analyze_1d(win,
                                    denoise_cutoff=STORE.settings.denoise_cutoff,
                                    denoise_low_cutoff=STORE.settings.denoise_low_cutoff,
                                    denoise_high_cutoff=STORE.settings.denoise_high_cutoff,
                                    harmonics_k=STORE.settings.harmonics_k,
                                    noise_gate=STORE.settings.noise_gate)
                self.audio_triplet.update_triplet(trip_a)

        # Hawk-Eye view (every other tick)
        if self._tick_count % 2 == 0:
            lw = self.hawk_lbl.width()
            lh = self.hawk_lbl.height()
            if lw < 8 or lh < 8:
                pass  # widget not laid out yet; skip this tick
            else:
                hw = max(320, lw)
                hh = int(hw * 68 / 105)
                if self.attacker is not None:
                    self._trail.append(self.attacker)
                    if len(self._trail) > 60:
                        self._trail = self._trail[-60:]
                hawk = render_hawkeye(hw, hh, self.H,
                                      self.attacker, self.defender, self._trail)
                qimg = ndarray_to_qimage(hawk)
                self.hawk_lbl.setPixmap(QtGui.QPixmap.fromImage(qimg).scaled(
                    self.hawk_lbl.size(), QtCore.Qt.KeepAspectRatio,
                    QtCore.Qt.SmoothTransformation))


def run():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
