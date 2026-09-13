"""
Per-application formatting profiles.

Dictating into a terminal should not produce the same text as dictating
into an email. This is the feature that makes Wispr Flow feel like it
understands you: "dictating an email produces formatted email text;
dictating code comments produces properly structured comments".

A profile is matched on the foreground process name and contributes one
extra instruction line to the refinement prompt. It never changes what the
guard checks -- meaning-preserving rules apply identically everywhere.

Users can override or extend the built-ins by editing
%APPDATA%/WhisprFlow/profiles.json.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Profile:
    name: str
    instruction: str
    processes: List[str] = field(default_factory=list)
    # Articulate rewrites more aggressively. Off for code and terminals,
    # where the literal words matter.
    allow_restructure: bool = False

    def matches(self, process: str) -> bool:
        p = (process or "").lower()
        return any(p == m.lower() for m in self.processes)


# ── built-ins ─────────────────────────────────────────────────────────────

CODE = Profile(
    name="Code",
    processes=["code.exe", "cursor.exe", "devenv.exe", "sublime_text.exe",
               "pycharm64.exe", "idea64.exe", "rider64.exe", "notepad++.exe",
               "windsurf.exe", "zed.exe"],
    instruction=(
        "The user is writing code or code comments. Keep identifiers, "
        "symbols and casing exactly as spoken (camelCase, snake_case, dot "
        "notation). Do not add prose or explanations. Do not capitalise "
        "identifiers. Spoken punctuation like 'dot', 'underscore', 'dash' "
        "inside an identifier should become the symbol."
    ),
)

TERMINAL = Profile(
    name="Terminal",
    processes=["windowsterminal.exe", "powershell.exe", "cmd.exe",
               "wt.exe", "conhost.exe", "alacritty.exe", "wezterm-gui.exe"],
    instruction=(
        "The user is typing a shell command. Output the command only, with "
        "no trailing full stop, no capitalisation of the command, and no "
        "explanation. Preserve flags and paths exactly."
    ),
)

CHAT = Profile(
    name="Chat",
    processes=["slack.exe", "discord.exe", "teams.exe", "ms-teams.exe",
               "whatsapp.exe", "telegram.exe", "signal.exe"],
    instruction=(
        "The user is sending a chat message. Keep it conversational and "
        "brief. Do not add a greeting or sign-off that was not spoken. "
        "Contractions are fine."
    ),
)

EMAIL = Profile(
    name="Email",
    processes=["outlook.exe", "thunderbird.exe", "mailspring.exe"],
    instruction=(
        "The user is writing an email. Use complete sentences and standard "
        "punctuation. Keep the speaker's tone -- do not add pleasantries, "
        "greetings or sign-offs they did not say."
    ),
    allow_restructure=True,
)

DOCUMENT = Profile(
    name="Document",
    processes=["winword.exe", "notion.exe", "obsidian.exe", "typora.exe",
               "onenote.exe", "notepad.exe", "wordpad.exe"],
    instruction=(
        "The user is writing prose in a document. Use complete sentences, "
        "correct punctuation and paragraph-appropriate capitalisation."
    ),
    allow_restructure=True,
)

BROWSER = Profile(
    name="Browser",
    processes=["chrome.exe", "msedge.exe", "firefox.exe", "brave.exe",
               "arc.exe", "opera.exe"],
    instruction=(
        "The user is typing into a web page -- it could be a search box, a "
        "form or a long-form editor. Keep the text close to what was said "
        "and use light punctuation."
    ),
)

DEFAULT = Profile(
    name="Default",
    processes=[],
    instruction="",
)

BUILTIN: List[Profile] = [CODE, TERMINAL, CHAT, EMAIL, DOCUMENT, BROWSER]


class ProfileSet:
    """Resolves a process name to a formatting profile."""

    def __init__(self, path: Optional[Path] = None, enabled: bool = True):
        self.enabled = enabled
        self.path = path
        self.profiles: List[Profile] = list(BUILTIN)
        self._overrides: Dict[str, str] = {}
        if path:
            self.load(path)

    # ── persistence ───────────────────────────────────────────────────────

    def load(self, path: Path) -> None:
        """Merge user profiles from JSON. Malformed files are ignored, not
        fatal -- a typo in a config file must not stop dictation."""
        self.path = Path(path)
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Could not read profiles.json: %s", e)
            return

        for entry in data.get("profiles", []):
            try:
                p = Profile(
                    name=entry["name"],
                    instruction=entry.get("instruction", ""),
                    processes=[s.lower() for s in entry.get("processes", [])],
                    allow_restructure=bool(entry.get("allow_restructure", False)),
                )
            except Exception:
                continue
            # A user profile with a built-in's name replaces it.
            self.profiles = [q for q in self.profiles if q.name.lower() != p.name.lower()]
            self.profiles.insert(0, p)

    def write_template(self) -> None:
        if not self.path or self.path.exists():
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({
                "_comment": (
                    "Custom formatting profiles. A profile matching the "
                    "foreground process adds one instruction to the AI "
                    "cleanup step. Names matching a built-in (Code, "
                    "Terminal, Chat, Email, Document, Browser) replace it."
                ),
                "profiles": [{
                    "name": "Example",
                    "processes": ["myapp.exe"],
                    "instruction": "Write in the style my team uses.",
                    "allow_restructure": False,
                }],
            }, indent=2), encoding="utf-8")
        except Exception as e:
            logger.debug("Could not write profiles template: %s", e)

    # ── resolution ────────────────────────────────────────────────────────

    def resolve(self, process: str) -> Profile:
        if not self.enabled or not process:
            return DEFAULT
        for p in self.profiles:
            if p.matches(process):
                return p
        return DEFAULT

    def get_info(self) -> dict:
        return {
            "enabled": self.enabled,
            "count": len(self.profiles),
            "names": [p.name for p in self.profiles],
        }
