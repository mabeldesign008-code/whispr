"""Tests for Command Mode: transformation, validation, selection capture."""

import asyncio
import sys
import types

import pytest

# pynput needs an X display on Linux and pyperclip needs a clipboard
# backend; neither exists on a headless CI runner. Stub only what is
# missing so the pure logic under test still runs everywhere.
for _name in ("sounddevice", "pyperclip"):
    if _name not in sys.modules:
        try:
            __import__(_name)
        except Exception:
            sys.modules[_name] = types.ModuleType(_name)

if "pynput" not in sys.modules:
    try:
        __import__("pynput.keyboard")
    except Exception:
        _pynput = types.ModuleType("pynput")
        _kb = types.ModuleType("pynput.keyboard")

        class _Key:
            ctrl = "ctrl"
            ctrl_l = "ctrl_l"
            cmd = "cmd"
            shift = "shift"
            esc = "esc"

        class _Controller:
            def press(self, k): pass
            def release(self, k): pass
            def type(self, t): pass

        _kb.Key = _Key
        _kb.Controller = _Controller
        _kb.Listener = object
        _kb.GlobalHotKeys = object
        _pynput.keyboard = _kb
        sys.modules["pynput"] = _pynput
        sys.modules["pynput.keyboard"] = _kb

if not hasattr(sys.modules["pyperclip"], "copy"):
    sys.modules["pyperclip"].copy = lambda t: None
    sys.modules["pyperclip"].paste = lambda: ""

from refine.commands import (
    CommandProcessor, CommandResult, clean_output, validate,
)


class TestCleanOutput:
    def test_strips_preamble(self):
        assert clean_output("Here is the transformed text: Hello.") == "Hello."
        assert clean_output("Here's the rewritten version: Hi") == "Hi"

    def test_strips_wrapping_quotes(self):
        assert clean_output('"Just a quote"') == "Just a quote"

    def test_keeps_internal_quotes(self):
        """Quotes inside the text are content, not wrapping."""
        s = 'He said "hi" and she said "bye"'
        assert clean_output(s) == s

    def test_strips_full_code_fence(self):
        assert clean_output("```\ncode here\n```") == "code here"

    def test_leaves_clean_text_alone(self):
        assert clean_output("Normal text.") == "Normal text."

    def test_empty(self):
        assert clean_output("") == ""
        assert clean_output(None) == ""


class TestValidate:
    def test_accepts_normal_transformation(self):
        assert validate("hello world", "Hello, world.", "fix grammar") is None

    def test_rejects_empty(self):
        assert validate("hello", "", "shorten") is not None

    def test_rejects_refusal(self):
        v = validate("hello", "I'm sorry, I cannot do that.", "shorten")
        assert v and "declined" in v

    def test_rejects_runaway_output(self):
        """Guards against the model answering the selection rather than
        transforming it."""
        v = validate("a" * 100, "b" * 900, "make it formal")
        assert v and "longer" in v

    def test_allows_runaway_when_expanding(self):
        """'expand' legitimately produces much more text."""
        assert validate("a" * 100, "b" * 900, "expand this") is None

    def test_rejects_implausibly_short(self):
        v = validate("a much longer piece of text here", "x", "summarise")
        assert v is not None

    def test_allows_translation_changing_every_word(self):
        """The dictation guard would reject this; Command Mode must not."""
        assert validate("we shipped it monday",
                        "nous l'avons livré lundi", "translate to French") is None

    def test_allows_summary_dropping_most_words(self):
        assert validate("A. B. C. D. E. F. G. H.", "A through H.",
                        "summarise") is None


class TestCanonicalInstruction:
    def _c(self, s):
        return CommandProcessor(api_key="k")._canonical(s)

    @pytest.mark.parametrize("spoken", ["make it shorter", "shorten this",
                                        "cut it down", "be more concise"])
    def test_shorten_aliases_converge(self, spoken):
        assert "shorter" in self._c(spoken).lower()

    def test_keeps_the_users_own_words(self):
        """Aliases add guidance; they must not discard extra detail."""
        out = self._c("make it shorter but keep the numbers")
        assert "keep the numbers" in out

    def test_unknown_instruction_passes_through(self):
        assert self._c("make it rhyme") == "make it rhyme"


class TestCommandProcessor:
    def _run(self, coro):
        return asyncio.run(coro)

    def test_no_api_key(self):
        r = self._run(CommandProcessor(api_key="").apply("hello", "shorten"))
        assert not r.ok and "key" in r.error.lower()

    def test_empty_selection(self):
        r = self._run(CommandProcessor(api_key="k").apply("", "shorten"))
        assert not r.ok and "selected" in r.error.lower()

    def test_empty_instruction(self):
        r = self._run(CommandProcessor(api_key="k").apply("hello", ""))
        assert not r.ok and "instruction" in r.error.lower()

    def test_oversize_selection_rejected(self):
        """A stray Ctrl+A must not send a novel to the API."""
        cp = CommandProcessor(api_key="k")
        r = self._run(cp.apply("x" * 20000, "shorten"))
        assert not r.ok and "too large" in r.error.lower()

    def test_selection_at_the_limit_is_allowed(self):
        cp = CommandProcessor(api_key="")
        r = self._run(cp.apply("x" * cp.MAX_CHARS, "shorten"))
        # Fails on the missing key, not on size.
        assert "too large" not in (r.error or "").lower()

    def test_uses_the_heavier_model(self):
        """Transformations are reasoning tasks and run once per invocation,
        unlike per-utterance dictation cleanup."""
        assert "70b" in CommandProcessor().model

    def test_set_api_key_resets_client(self):
        cp = CommandProcessor(api_key="old")
        cp._client = object()
        cp.set_api_key("new")
        assert cp.api_key == "new" and cp._client is None

    def test_stats_shape(self):
        s = CommandProcessor(api_key="k").get_stats()
        assert {"model", "configured", "invocations", "rejections"} <= set(s)


class TestCommandResult:
    def test_empty_detection(self):
        assert CommandResult(text="   ").is_empty
        assert not CommandResult(text="x").is_empty


class TestSelectionManager:
    """Clipboard behaviour, with pyperclip stubbed."""

    def _manager(self, monkeypatch, clipboard_value, copy_worked):
        import selection as sel_mod

        store = {"v": clipboard_value}
        copied = []

        monkeypatch.setattr(sel_mod.pyperclip, "paste", lambda: store["v"])

        def fake_copy(t):
            store["v"] = t
            copied.append(t)

        monkeypatch.setattr(sel_mod.pyperclip, "copy", fake_copy)
        monkeypatch.setattr(sel_mod, "wait_for_modifier_release", lambda *a, **k: True)
        monkeypatch.setattr(sel_mod, "COPY_SETTLE", 0)
        monkeypatch.setattr(sel_mod, "PASTE_SETTLE", 0)

        mgr = sel_mod.SelectionManager()

        class FakeKb:
            def press(self, k):
                # Simulate the target app answering Ctrl+C.
                if copy_worked and k == "c":
                    store["v"] = "the selected text"

            def release(self, k):
                pass

        mgr._kb = FakeKb()
        return mgr, store, copied

    def test_captures_selection(self, monkeypatch):
        mgr, _, _ = self._manager(monkeypatch, "old clipboard", copy_worked=True)
        sel = mgr.capture()
        assert sel.ok and sel.text == "the selected text"

    def test_detects_nothing_selected(self, monkeypatch):
        """Ctrl+C on an empty selection leaves the sentinel in place."""
        mgr, _, _ = self._manager(monkeypatch, "old clipboard", copy_worked=False)
        sel = mgr.capture()
        assert not sel.ok and "no text selected" in sel.error.lower()

    def test_restores_clipboard_after_capture(self, monkeypatch):
        mgr, store, _ = self._manager(monkeypatch, "old clipboard", copy_worked=True)
        mgr.capture()
        assert store["v"] == "old clipboard"

    def test_sentinel_is_unique_per_call(self, monkeypatch):
        """A fixed sentinel would false-negative if the user had copied it."""
        mgr, _, copied = self._manager(monkeypatch, "x", copy_worked=True)
        mgr.capture()
        mgr.capture()
        sentinels = [c for c in copied if "whisprflow_probe" in c]
        assert len(sentinels) == 2 and sentinels[0] != sentinels[1]

    def test_replace_writes_and_restores(self, monkeypatch):
        import selection as sel_mod
        monkeypatch.setattr(sel_mod, "RESTORE_DELAY", 0)
        mgr, store, copied = self._manager(monkeypatch, "original", copy_worked=True)
        assert mgr.replace("new text")
        assert "new text" in copied

    def test_replace_rejects_empty(self, monkeypatch):
        mgr, _, _ = self._manager(monkeypatch, "x", copy_worked=True)
        assert not mgr.replace("")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
