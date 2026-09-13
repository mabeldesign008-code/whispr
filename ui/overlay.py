"""
Floating pill overlay.

Design language follows the best-in-class dictation apps (Wispr Flow et al):
a near-black pill, high corner radius, restrained colour, a dotted rest
state that expands into a live waveform. The UI should ask nothing of the
user and never draw attention to itself when idle.

Rewritten from the old FloatingOverlay, which had three problems:

1.19  It rendered at 15 fps forever, even collapsed to a 2 px sliver:
      a full Image.new at 2x supersample, a LANCZOS resize, a new
      ImageTk.PhotoImage and canvas.delete("all"), fifteen times a second,
      on the Tk main thread. Now: a dirty-flag render loop that does
      nothing while idle and unchanged, and the rest state is a cached
      static image drawn once.

      _draw_success also allocated a font every single frame.

      Spring.update() was called with a hardcoded dt=0.016 in _draw_waves
      but real dt in update_position, so animation speed differed between
      the 15 fps and 30 fps paths.

Visual states: HIDDEN -> IDLE -> RECORDING -> PROCESSING -> SUCCESS/ERROR
"""

from __future__ import annotations

import math
import time
import tkinter as tk
from enum import Enum
from typing import Callable, Optional

from PIL import Image, ImageDraw, ImageFont, ImageTk

from .theme import (
    ACCENT, ACCENT_DIM, BG_PILL, COMMAND, DANGER, FONT_CANDIDATES, MUTED,
    SUCCESS, TEXT, ease_out_back, ease_out_cubic,
)


class PillState(Enum):
    IDLE = "idle"
    RECORDING = "recording"
    PROCESSING = "processing"
    SUCCESS = "success"
    ERROR = "error"


class Spring:
    """Critically-damped-ish spring. Always integrated with real dt."""

    __slots__ = ("value", "target", "velocity", "stiffness", "damping")

    def __init__(self, value: float, stiffness: float = 220.0, damping: float = 24.0):
        self.value = value
        self.target = value
        self.velocity = 0.0
        self.stiffness = stiffness
        self.damping = damping

    def update(self, dt: float) -> float:
        # Clamp dt: a stalled main thread must not launch the spring.
        dt = min(dt, 0.05)
        force = (self.target - self.value) * self.stiffness
        self.velocity += (force - self.damping * self.velocity) * dt
        self.value += self.velocity * dt
        return self.value

    @property
    def at_rest(self) -> bool:
        return abs(self.value - self.target) < 0.15 and abs(self.velocity) < 0.15

    def snap(self, value: float) -> None:
        self.value = self.target = value
        self.velocity = 0.0


class FloatingPill:
    # Geometry (logical px; rendered at 2x then downsampled)
    W_IDLE = 96
    W_ACTIVE = 232
    H_IDLE = 8
    H_ACTIVE = 44
    SS = 2                      # supersample factor

    FPS_ACTIVE = 60             # while animating
    FPS_IDLE = 0                # no redraw at all when settled

    BARS = 13
    BOTTOM_MARGIN = 72
    HOVER_W = 150
    HOVER_H = 90

    def __init__(
        self,
        parent_root: tk.Tk,
        get_level: Callable[[], float],
        on_stop: Optional[Callable[[], None]] = None,
        on_cancel: Optional[Callable[[], None]] = None,
        on_retry: Optional[Callable[[], None]] = None,
    ):
        self._get_level = get_level
        self._on_stop = on_stop or (lambda: None)
        self._on_cancel = on_cancel or (lambda: None)
        self._on_retry = on_retry or (lambda: None)

        self.root = tk.Toplevel(parent_root)
        self.root.overrideredirect(True)
        self.root.configure(bg="black")

        # Each of these is best-effort. -transparentcolor is Windows-only,
        # and a window manager that rejects any one of them must not stop
        # the app from starting -- the pill just looks less polished.
        for attr, value in (("-topmost", True), ("-transparentcolor", "black")):
            try:
                self.root.attributes(attr, value)
            except Exception:
                pass

        self.state = PillState.IDLE
        self._state_since = time.monotonic()
        self._status_text = ""
        self._partial_text = ""
        self._locked = False
        self._command_mode = False

        # Animation state
        self._width = Spring(self.W_IDLE, 260, 26)
        self._height = Spring(self.H_IDLE, 260, 26)
        self._reveal = Spring(0.0, 200, 22)       # 0 = pill only, 1 = contents
        self._bars = [Spring(0.0, 320, 22) for _ in range(self.BARS)]
        self._glow = 0.0

        self._hovered = False
        self._last_frame = time.monotonic()
        self._dirty = True
        self._photo = None
        self._idle_cache: Optional[ImageTk.PhotoImage] = None
        self._idle_size = (0, 0)
        self._after_id: Optional[str] = None

        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self._cx = sw // 2
        self._cy = sh - self.BOTTOM_MARGIN

        self.canvas = tk.Canvas(
            self.root, width=self.W_ACTIVE, height=self.H_ACTIVE,
            bg="black", highlightthickness=0, bd=0,
        )
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<Enter>", self._on_enter)
        self.canvas.bind("<Leave>", self._on_leave)

        self._font = _load_font(10 * self.SS)
        self._font_small = _load_font(9 * self.SS)

        self._apply_geometry()
        self._tick()

    # ── public API ────────────────────────────────────────────────────────

    def set_state(self, state: PillState, status: str = "") -> None:
        if state == self.state and status == self._status_text:
            return
        self.state = state
        self._status_text = status
        self._state_since = time.monotonic()

        active = state is not PillState.IDLE or self._hovered
        self._width.target = self.W_ACTIVE if active else self.W_IDLE
        self._height.target = self.H_ACTIVE if active else self.H_IDLE
        self._reveal.target = 1.0 if active else 0.0

        if state is not PillState.RECORDING:
            self._locked = False
            self._command_mode = False
        self._command_mode = False
        if state is PillState.RECORDING:
            self._partial_text = ""
            for b in self._bars:
                b.snap(0.0)

        self._wake()

    def set_locked(self, locked: bool) -> None:
        """Hands-free mode. Shown as a persistent dot on the stop button so
        the user can tell at a glance that it is still listening."""
        if locked == self._locked:
            return
        self._locked = locked
        self._wake()

    def set_command_mode(self, active: bool) -> None:
        """Command Mode recolours the pill so it is unmistakably not
        dictation -- text is about to be *replaced*, not inserted, and
        that deserves a different signal."""
        if active == self._command_mode:
            return
        self._command_mode = active
        self._wake()

    def set_partial(self, text: str) -> None:
        """Live transcript while the user is still speaking. Shown instead
        of the waveform once there are words to show -- seeing your words
        appear is far better feedback than a bouncing equaliser."""
        text = (text or "").strip()
        if text == self._partial_text:
            return
        self._partial_text = text
        self._wake()

    def flash_success(self, message: str = "") -> None:
        self.set_state(PillState.SUCCESS, message)
        self.root.after(1400, self._back_to_idle)

    def flash_error(self, message: str = "") -> None:
        self.set_state(PillState.ERROR, message)
        self.root.after(3200, self._back_to_idle)

    def _back_to_idle(self) -> None:
        if self.state in (PillState.SUCCESS, PillState.ERROR):
            self.set_state(PillState.IDLE)

    def destroy(self) -> None:
        if self._after_id:
            try:
                self.root.after_cancel(self._after_id)
            except Exception:
                pass
        try:
            self.root.destroy()
        except Exception:
            pass

    # ── event handlers ────────────────────────────────────────────────────

    def _on_enter(self, _evt=None):
        self._hovered = True
        if self.state is PillState.IDLE:
            self._width.target = self.W_ACTIVE
            self._height.target = self.H_ACTIVE
            self._reveal.target = 1.0
        self._wake()

    def _on_leave(self, _evt=None):
        self._hovered = False
        if self.state is PillState.IDLE:
            self._width.target = self.W_IDLE
            self._height.target = self.H_IDLE
            self._reveal.target = 0.0
        self._wake()

    def _on_click(self, event):
        if self._reveal.value < 0.5:
            return
        w = max(self._width.value, 1)
        rel = event.x / w
        if self.state is PillState.RECORDING:
            if rel < 0.22:
                self._on_cancel()
            elif rel > 0.78:
                self._on_stop()
        elif self.state is PillState.ERROR:
            self._on_retry()

    # ── render loop ───────────────────────────────────────────────────────

    def _wake(self) -> None:
        self._dirty = True

    def _needs_frame(self) -> bool:
        """Render only when something is actually moving."""
        if self._dirty:
            return True
        if self.state in (PillState.RECORDING, PillState.PROCESSING):
            return True
        springs = [self._width, self._height, self._reveal]
        return not all(s.at_rest for s in springs)

    def _tick(self) -> None:
        now = time.monotonic()
        dt = now - self._last_frame
        self._last_frame = now

        if self._needs_frame():
            self._width.update(dt)
            self._height.update(dt)
            self._reveal.update(dt)
            self._apply_geometry()
            self._render(dt)
            self._dirty = False
            interval = 1000 // self.FPS_ACTIVE
        else:
            # Fully settled: poll infrequently, draw nothing.
            interval = 120

        self._after_id = self.root.after(interval, self._tick)

    def _apply_geometry(self) -> None:
        w = max(int(self._width.value), 1)
        h = max(int(self._height.value), 1)
        x = self._cx - w // 2
        y = self._cy - h // 2
        try:
            self.root.geometry(f"{w}x{h}+{x}+{y}")
            self.canvas.configure(width=w, height=h)
        except Exception:
            pass

    def _render(self, dt: float) -> None:
        w = max(int(self._width.value), 1)
        h = max(int(self._height.value), 1)

        # Rest state is a single cached bitmap -- it never changes, so
        # there is no reason to rebuild it 15 times a second.
        if self._reveal.value < 0.02 and self._width.at_rest and self._height.at_rest:
            self._draw_cached_idle(w, h)
            return

        S = self.SS
        img = Image.new("RGBA", (w * S, h * S), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)

        reveal = ease_out_cubic(max(0.0, min(1.0, self._reveal.value)))
        radius = min(h * S // 2, 22 * S)

        # Recording glow: a soft outer ring that breathes.
        if self.state is PillState.RECORDING:
            self._glow = (math.sin(time.monotonic() * 3.2) + 1) / 2
            a = int(46 * self._glow * reveal)
            if a > 3:
                d.rounded_rectangle(
                    [1, 1, w * S - 2, h * S - 2],
                    radius=radius, outline=(*self._accent(), a),
                    width=max(2, 2 * S),
                )

        # Pill body.
        d.rounded_rectangle([0, 0, w * S - 1, h * S - 1], radius=radius, fill=(*BG_PILL, 244))
        # Top inner highlight for a subtle sense of depth.
        d.rounded_rectangle(
            [S, S, w * S - S - 1, h * S - S - 1],
            radius=max(radius - S, 1), outline=(255, 255, 255, int(20 * reveal + 8)), width=1,
        )

        if reveal > 0.12:
            if self.state is PillState.RECORDING:
                self._draw_recording(d, w * S, h * S, S, reveal, dt)
            elif self.state is PillState.PROCESSING:
                self._draw_processing(d, w * S, h * S, S, reveal)
            elif self.state is PillState.SUCCESS:
                self._draw_message(d, w * S, h * S, S, reveal, self._status_text or "Inserted", SUCCESS)
            elif self.state is PillState.ERROR:
                self._draw_message(d, w * S, h * S, S, reveal, self._status_text or "Failed \u21ba", DANGER)
            else:
                self._draw_hint(d, w * S, h * S, S, reveal)
        elif self.state is PillState.IDLE:
            self._draw_dots(d, w * S, h * S, S, 1.0 - reveal)

        final = img.resize((w, h), Image.Resampling.LANCZOS)
        self._photo = ImageTk.PhotoImage(final, master=self.root)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo)

    def _draw_cached_idle(self, w: int, h: int) -> None:
        # Cache keyed on size: the collapsed pill is identical every frame,
        # so there is no reason to rebuild it. Rebuilt if the size changes
        # (DPI switch, display change).
        if self._idle_cache is None or self._idle_size != (w, h):
            S = self.SS
            img = Image.new("RGBA", (w * S, h * S), (0, 0, 0, 0))
            d = ImageDraw.Draw(img)
            d.rounded_rectangle([0, 0, w * S - 1, h * S - 1],
                                radius=h * S // 2, fill=(*BG_PILL, 235))
            self._draw_dots(d, w * S, h * S, S, 1.0)
            # master= is required: without it PhotoImage binds to the
            # default Tk root rather than this Toplevel, and the canvas
            # raises 'image "pyimageN" doesn\'t exist'.
            self._idle_cache = ImageTk.PhotoImage(
                img.resize((w, h), Image.Resampling.LANCZOS), master=self.root
            )
            self._idle_size = (w, h)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._idle_cache)

    # ── state renderers ───────────────────────────────────────────────────

    def _draw_dots(self, d, bw, bh, S, alpha):
        """Rest state: a row of small dots, like Wispr's collapsed pill."""
        n, gap = 9, 5 * S
        r = max(1, int(1.1 * S))
        total = (n - 1) * gap
        x0 = bw // 2 - total // 2
        cy = bh // 2
        for i in range(n):
            a = int(150 * alpha)
            d.ellipse([x0 + i * gap - r, cy - r, x0 + i * gap + r, cy + r],
                      fill=(*MUTED, a))

    def _accent(self):
        """Violet while transforming a selection, blue while dictating."""
        return COMMAND if self._command_mode else ACCENT

    def _draw_recording(self, d, bw, bh, S, reveal, dt):
        self._draw_cancel(d, bw, bh, S, reveal)
        if self._partial_text:
            self._draw_partial(d, bw, bh, S, reveal)
        else:
            self._draw_waveform(d, bw, bh, S, reveal, dt)
        self._draw_stop(d, bw, bh, S, reveal)

    def _draw_partial(self, d, bw, bh, S, reveal):
        """Live text, tail-anchored so the newest words stay visible.

        Measured against the space between the cancel and stop buttons and
        trimmed to fit, so a long transcript can never collide with them.
        """
        left = 32 * S           # clear of the cancel button
        right = bw - (36 if self._locked else 32) * S   # clear of the stop button
        avail = right - left
        cy = bh // 2

        def width(s: str) -> float:
            try:
                return d.textlength(s, font=self._font_small)
            except Exception:
                return len(s) * 5.0 * S

        caret_w = 5 * S
        text = _tail(self._partial_text, 60)
        while text and width(text) > avail - caret_w:
            text = "\u2026" + text[2:]      # drop a char, keep the ellipsis

        if not text:
            return

        tw = width(text)
        x = left + (avail - tw - caret_w) / 2
        d.text((x, cy), text, font=self._font_small,
               fill=(*TEXT, int(240 * reveal)), anchor="lm")

        # Blinking caret keeps it feeling live even mid-pause.
        if (time.monotonic() % 1.0) < 0.6:
            cx = x + tw + 2.5 * S
            d.line([cx, cy - 5 * S, cx, cy + 5 * S],
                   fill=(*self._accent(), int(220 * reveal)),
                   width=max(1, int(1.2 * S)))

    def _draw_waveform(self, d, bw, bh, S, reveal, dt):
        level = min(1.0, max(0.0, self._get_level() * 6.0))
        cx, cy = bw // 2, bh // 2
        bar_w = 3 * S
        gap = 6 * S
        max_h = (bh - 18 * S)
        t = time.monotonic()

        # Idle breathing so the pill never looks frozen while listening,
        # even in a silent room. Scales down as real level takes over.
        breathe = (1.0 - level) * 1.6 * S

        for i in range(self.BARS):
            # Centre bars react most; edges trail. Gives the waveform a
            # physical, weighted feel rather than a flat equaliser.
            dist = abs(i - self.BARS // 2) / (self.BARS / 2)
            weight = 1.0 - dist * 0.55
            wobble = 0.72 + 0.28 * math.sin(t * 9 + i * 0.85)
            idle = breathe * (0.6 + 0.4 * math.sin(t * 2.4 + i * 0.5))
            target = 3.0 * S + idle + level * max_h * weight * wobble
            self._bars[i].target = target
            h = self._bars[i].update(dt)

            x = cx + (i - self.BARS // 2) * (bar_w + gap)
            y0, y1 = cy - h / 2, cy + h / 2

            # Hot centre, cooler edges.
            mix = level * (1.0 - dist * 0.5)
            col = _lerp_rgb(TEXT, self._accent(), min(1.0, mix))
            d.rounded_rectangle(
                [x - bar_w // 2, y0, x + bar_w // 2, y1],
                radius=bar_w // 2, fill=(*col, int(255 * reveal)),
            )

    def _draw_cancel(self, d, bw, bh, S, reveal):
        cx, cy = 19 * S, bh // 2
        r = 8 * S
        a = int(255 * reveal)

        if self._command_mode:
            # Sparkle glyph: this will transform text, not insert it.
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(*COMMAND, a))
            for dx, dy, size in ((0, 0, 3.4), (3.2, -3.2, 1.6)):
                x, y = cx + dx * S, cy + dy * S
                k = size * S
                d.polygon([(x, y - k), (x + k * 0.32, y - k * 0.32),
                           (x + k, y), (x + k * 0.32, y + k * 0.32),
                           (x, y + k), (x - k * 0.32, y + k * 0.32),
                           (x - k, y), (x - k * 0.32, y - k * 0.32)],
                          fill=(255, 255, 255, a))
            return

        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(56, 56, 60, a))
        k = 3 * S
        for dx, dy in (((-k, -k), (k, k)), ((k, -k), (-k, k))):
            d.line([cx + dx[0], cy + dx[1], cx + dy[0], cy + dy[1]],
                   fill=(*MUTED, a), width=max(1, int(1.4 * S)))

    def _draw_stop(self, d, bw, bh, S, reveal):
        cx, cy = bw - 19 * S, bh // 2
        r = 8 * S
        a = int(255 * reveal)

        if self._locked:
            # Steady, with a halo: hands-free and staying that way. No
            # pulse, because a pulsing control reads as "about to stop".
            d.ellipse([cx - r - 2.5 * S, cy - r - 2.5 * S,
                       cx + r + 2.5 * S, cy + r + 2.5 * S],
                      outline=(*self._accent(), int(150 * reveal)),
                      width=max(1, int(1.3 * S)))
            rr = r
        else:
            rr = r * (1.0 + 0.06 * math.sin(time.monotonic() * 5))

        d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=(*DANGER, a))
        s = 2.6 * S
        d.rounded_rectangle([cx - s, cy - s, cx + s, cy + s],
                            radius=max(1, int(0.8 * S)), fill=(255, 255, 255, a))

    def _draw_processing(self, d, bw, bh, S, reveal):
        """Three dots cascading -- reads as 'thinking', not 'stuck'."""
        cx, cy = bw // 2, bh // 2
        t = time.monotonic()
        gap = 9 * S
        for i in range(3):
            phase = (t * 2.6 - i * 0.32) % 2.0
            lift = math.sin(min(phase, 1.0) * math.pi)
            r = (2.0 + 1.1 * lift) * S
            a = int((120 + 135 * lift) * reveal)
            x = cx + (i - 1) * gap
            d.ellipse([x - r, cy - r - lift * 2 * S, x + r, cy + r - lift * 2 * S],
                      fill=(*ACCENT_DIM, a))

    def _draw_message(self, d, bw, bh, S, reveal, text, colour):
        """Icon + label, centred as a single group so the pill stays
        visually balanced instead of leaving dead space on the right."""
        age = time.monotonic() - self._state_since
        pop = ease_out_back(min(1.0, age / 0.28))
        cy = bh // 2

        label = _truncate(text, 24)
        icon_r = 7 * S
        gap = 8 * S

        try:
            tw = d.textlength(label, font=self._font_small)
        except Exception:
            tw = len(label) * 5 * S

        group_w = icon_r * 2 + gap + tw
        x = (bw - group_w) / 2 + icon_r

        r = icon_r * pop
        d.ellipse([x - r, cy - r, x + r, cy + r], fill=(*colour, int(255 * reveal)))

        if colour == SUCCESS:
            d.line([x - 3 * S, cy, x - 0.7 * S, cy + 2.6 * S],
                   fill=(16, 16, 18, 255), width=max(1, int(1.6 * S)))
            d.line([x - 0.7 * S, cy + 2.6 * S, x + 3.4 * S, cy - 2.4 * S],
                   fill=(16, 16, 18, 255), width=max(1, int(1.6 * S)))
        else:
            k = 2.4 * S
            d.line([x - k, cy - k, x + k, cy + k],
                   fill=(255, 255, 255, 255), width=max(1, int(1.6 * S)))
            d.line([x + k, cy - k, x - k, cy + k],
                   fill=(255, 255, 255, 255), width=max(1, int(1.6 * S)))

        d.text((x + icon_r + gap, cy), label, font=self._font_small,
               fill=(*TEXT, int(235 * reveal)), anchor="lm")

    def _draw_hint(self, d, bw, bh, S, reveal):
        d.text((bw // 2, bh // 2), "Hold Ctrl + Win to dictate",
               font=self._font_small, fill=(*MUTED, int(220 * reveal)), anchor="mm")


# ── helpers ───────────────────────────────────────────────────────────────


def _load_font(size: int):
    for name in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _lerp_rgb(a, b, t: float):
    t = max(0.0, min(1.0, t))
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _tail(text: str, limit: int) -> str:
    """Keep the end of a growing transcript, not the start."""
    text = (text or "").strip()
    return text if len(text) <= limit else "\u2026" + text[-(limit - 1):]


def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: limit - 1] + "\u2026"
