"""Shared design tokens.

Near-black surfaces, one accent, restrained typography. The overlay should
be invisible until it has something to say.
"""

# ── palette (RGB tuples so PIL can take them directly) ────────────────────

BG_PILL = (14, 14, 16)        # the floating pill body
BG_APP = (18, 18, 20)         # main window
BG_CARD = (26, 26, 29)
BG_INPUT = (34, 34, 38)
BG_HOVER = (44, 44, 49)

TEXT = (245, 245, 247)
MUTED = (138, 138, 148)
FAINT = (92, 92, 102)

ACCENT = (122, 162, 255)      # cool blue, used sparingly
ACCENT_DIM = (92, 122, 200)
COMMAND = (183, 148, 255)     # violet -- Command Mode, distinct from dictation
SUCCESS = (74, 222, 128)
WARNING = (250, 190, 88)
DANGER = (248, 92, 92)

# ── hex equivalents for Tkinter widgets ───────────────────────────────────


def hexc(rgb) -> str:
    return "#%02x%02x%02x" % rgb


HEX_BG_APP = hexc(BG_APP)
HEX_BG_CARD = hexc(BG_CARD)
HEX_BG_INPUT = hexc(BG_INPUT)
HEX_BG_HOVER = hexc(BG_HOVER)
HEX_TEXT = hexc(TEXT)
HEX_MUTED = hexc(MUTED)
HEX_FAINT = hexc(FAINT)
HEX_ACCENT = hexc(ACCENT)
HEX_COMMAND = hexc(COMMAND)
HEX_SUCCESS = hexc(SUCCESS)
HEX_WARNING = hexc(WARNING)
HEX_DANGER = hexc(DANGER)

# ── typography ────────────────────────────────────────────────────────────

FONT_CANDIDATES = (
    "segoeui.ttf",            # Windows
    "SegoeUI.ttf",
    "Inter-Regular.ttf",
    "DejaVuSans.ttf",         # Linux fallback
    "Arial.ttf",
    "arial.ttf",
)

UI_FONT = "Segoe UI"
MONO_FONT = "Cascadia Mono"

# ── easing ────────────────────────────────────────────────────────────────


def ease_out_cubic(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 1 - (1 - t) ** 3


def ease_in_out_cubic(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 4 * t * t * t if t < 0.5 else 1 - ((-2 * t + 2) ** 3) / 2


def ease_out_back(t: float, overshoot: float = 1.70158) -> float:
    """Slight overshoot -- good for confirmations."""
    t = max(0.0, min(1.0, t))
    c3 = overshoot + 1
    return 1 + c3 * ((t - 1) ** 3) + overshoot * ((t - 1) ** 2)
