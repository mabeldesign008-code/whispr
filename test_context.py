"""Tests for app context, profiles, auto-learn and streaming."""

import asyncio
import json
import sys
import types

import numpy as np
import pytest

if "sounddevice" not in sys.modules:
    try:
        import sounddevice  # noqa: F401
    except Exception:
        sys.modules["sounddevice"] = types.ModuleType("sounddevice")

from context.app_context import AppContext, AppContextReader, _clip_tail, _prettify
from context.learner import DictionaryLearner, _looks_like_term
from context.profiles import BUILTIN, DEFAULT, Profile, ProfileSet
from stt.dictionary import UserDictionary
from stt.streaming import StreamingSession
from refine.refiner import Refiner


# ── app context ───────────────────────────────────────────────────────────

class TestAppContext:
    def test_empty_context_produces_no_prompt(self):
        assert AppContext().as_prompt() == ""
        assert AppContext().is_empty

    def test_prompt_includes_app_and_control(self):
        c = AppContext(process="code.exe", app_name="Visual Studio Code",
                       window_title="main.py - myproject", control_type="Edit")
        p = c.as_prompt()
        assert "Visual Studio Code" in p and "Edit" in p

    def test_prompt_strips_app_suffix_from_title(self):
        """Titles are usually 'document — App'; the app name is redundant."""
        c = AppContext(process="winword.exe", app_name="Microsoft Word",
                       window_title="Q3 Report \u2014 Microsoft Word")
        assert "Q3 Report" in c.as_prompt()

    def test_prompt_omits_title_equal_to_app(self):
        c = AppContext(process="slack.exe", app_name="Slack", window_title="Slack")
        assert c.as_prompt().count("Slack") == 1

    def test_reader_never_raises_without_win32(self):
        r = AppContextReader(enabled=True)
        assert isinstance(r.capture(), AppContext)

    def test_disabled_reader_returns_empty(self):
        assert AppContextReader(enabled=False).capture().is_empty

    def test_clip_tail_keeps_the_end(self):
        """The caret is at the end, so nearby context is the useful part."""
        out = _clip_tail("a" * 50 + "IMPORTANT", 20)
        assert out.endswith("IMPORTANT") and len(out) <= 21

    def test_prettify_process_name(self):
        assert _prettify("my_app.exe") == "My App"
        assert _prettify("") == ""

    def test_info_reports_dependencies(self):
        info = AppContextReader().get_info()
        assert {"enabled", "win32", "uiautomation", "psutil"} <= set(info)


# ── profiles ──────────────────────────────────────────────────────────────

class TestProfiles:
    def test_resolves_known_editors_to_code(self):
        ps = ProfileSet()
        for proc in ["code.exe", "Cursor.exe", "devenv.exe"]:
            assert ps.resolve(proc).name == "Code"

    def test_resolves_terminal(self):
        assert ProfileSet().resolve("WindowsTerminal.exe").name == "Terminal"

    def test_matching_is_case_insensitive(self):
        assert ProfileSet().resolve("SLACK.EXE").name == "Chat"

    def test_unknown_process_falls_back_to_default(self):
        p = ProfileSet().resolve("nothing_known.exe")
        assert p is DEFAULT and p.instruction == ""

    def test_empty_process_is_default(self):
        assert ProfileSet().resolve("") is DEFAULT

    def test_disabled_always_default(self):
        assert ProfileSet(enabled=False).resolve("code.exe") is DEFAULT

    def test_code_and_terminal_forbid_restructuring(self):
        """Literal words matter in code; paraphrasing them is destructive."""
        ps = ProfileSet()
        assert not ps.resolve("code.exe").allow_restructure
        assert not ps.resolve("cmd.exe").allow_restructure

    def test_prose_apps_allow_restructuring(self):
        ps = ProfileSet()
        assert ps.resolve("outlook.exe").allow_restructure
        assert ps.resolve("winword.exe").allow_restructure

    def test_every_builtin_has_an_instruction(self):
        assert all(p.instruction.strip() and p.processes for p in BUILTIN)

    def test_user_profile_overrides_builtin(self, tmp_path):
        f = tmp_path / "profiles.json"
        f.write_text(json.dumps({"profiles": [{
            "name": "Code", "processes": ["code.exe"],
            "instruction": "custom rule", "allow_restructure": True}]}))
        assert ProfileSet(f).resolve("code.exe").instruction == "custom rule"

    def test_user_profile_adds_new_app(self, tmp_path):
        f = tmp_path / "profiles.json"
        f.write_text(json.dumps({"profiles": [{
            "name": "Mine", "processes": ["myapp.exe"], "instruction": "x"}]}))
        assert ProfileSet(f).resolve("myapp.exe").name == "Mine"

    def test_malformed_json_is_not_fatal(self, tmp_path):
        """A typo in a config file must never stop dictation."""
        f = tmp_path / "profiles.json"
        f.write_text("{ not json")
        assert ProfileSet(f).resolve("code.exe").name == "Code"

    def test_template_written_once(self, tmp_path):
        f = tmp_path / "profiles.json"
        ps = ProfileSet(f)
        ps.write_template()
        assert f.exists()
        before = f.read_text()
        ps.write_template()
        assert f.read_text() == before


# ── auto-learn ────────────────────────────────────────────────────────────

class TestLearner:
    def _learner(self, tmp_path):
        d = UserDictionary(tmp_path / "dict.txt")
        return DictionaryLearner(tmp_path / "learned.json", d), d

    def test_promotes_after_repeated_sightings(self, tmp_path):
        lr, _ = self._learner(tmp_path)
        for _ in range(3):
            lr.observe("deploying to Kubernetes today")
        assert "Kubernetes" in [c.term for c in lr.candidates()]

    def test_does_not_promote_one_off(self, tmp_path):
        lr, _ = self._learner(tmp_path)
        lr.observe("deploying to Kubernetes today")
        assert lr.candidates() == []

    def test_ignores_ordinary_english(self, tmp_path):
        lr, _ = self._learner(tmp_path)
        for _ in range(8):
            lr.observe("the thing that we should really think about going forward")
        assert lr.candidates() == []

    def test_low_confidence_counts_double(self, tmp_path):
        """A word the recogniser keeps struggling with is a strong signal."""
        lr, _ = self._learner(tmp_path)
        for _ in range(2):
            lr.observe("call Zyntharo now", uncertain_words=["Zyntharo"])
        assert "Zyntharo" in [c.term for c in lr.candidates()]

    def test_skips_terms_already_in_dictionary(self, tmp_path):
        lr, d = self._learner(tmp_path)
        d.add("Kubernetes")
        for _ in range(5):
            lr.observe("Kubernetes again")
        assert "Kubernetes" not in [c.term for c in lr.candidates()]

    def test_accept_moves_into_dictionary(self, tmp_path):
        lr, d = self._learner(tmp_path)
        for _ in range(3):
            lr.observe("using WhisprFlow daily")
        assert lr.accept("WhisprFlow")
        assert "WhisprFlow" in d.terms
        assert "WhisprFlow" not in [c.term for c in lr.candidates()]

    def test_dismiss_is_permanent(self, tmp_path):
        lr, _ = self._learner(tmp_path)
        for _ in range(4):
            lr.observe("the Foobarbaz thing")
        lr.dismiss("Foobarbaz")
        for _ in range(6):
            lr.observe("the Foobarbaz thing")
        assert "Foobarbaz" not in [c.term for c in lr.candidates()]

    def test_state_persists(self, tmp_path):
        lr, d = self._learner(tmp_path)
        for _ in range(3):
            lr.observe("ship the Zephyrnaut build")
        reloaded = DictionaryLearner(tmp_path / "learned.json", d)
        assert "Zephyrnaut" in [c.term for c in reloaded.candidates()]

    def test_disabled_learns_nothing(self, tmp_path):
        d = UserDictionary(tmp_path / "dict.txt")
        lr = DictionaryLearner(tmp_path / "learned.json", d, enabled=False)
        for _ in range(9):
            lr.observe("Kubernetes Kubernetes")
        assert lr.candidates() == []

    def test_ranked_by_score(self, tmp_path):
        lr, _ = self._learner(tmp_path)
        for _ in range(3):
            lr.observe("Alphaton")
        for _ in range(8):
            lr.observe("Betaton")
        assert lr.candidates()[0].term == "Betaton"

    @pytest.mark.parametrize("token,expected", [
        ("WhisprFlow", True), ("GPT4", True), ("on-prem", True),
        ("Kubernetes", True), ("Accra", True),
        ("running", False), ("something", False), ("about", False),
    ])
    def test_term_heuristic(self, token, expected):
        assert _looks_like_term(token) is expected

    def test_empty_text_safe(self, tmp_path):
        lr, _ = self._learner(tmp_path)
        lr.observe("")
        lr.observe(None)
        assert lr.candidates() == []


# ── snippets ──────────────────────────────────────────────────────────────

class TestSnippets:
    def _set(self, tmp_path):
        from context.snippets import SnippetSet
        s = SnippetSet(tmp_path / "s.json")
        s.add("my email address", "ama@example.com")
        s.add("standard signoff", "Best regards,\nAma")
        return s

    def test_exact_utterance_replaced_entirely(self, tmp_path):
        out, used = self._set(tmp_path).expand("my email address")
        assert out == "ama@example.com" and used == ["my email address"]

    def test_inline_substitution(self, tmp_path):
        out, _ = self._set(tmp_path).expand("Send it to my email address, please.")
        assert out == "Send it to ama@example.com, please."

    def test_ignores_case_and_punctuation(self, tmp_path):
        """Transcripts arrive punctuated and capitalised inconsistently."""
        out, _ = self._set(tmp_path).expand("My email address.")
        assert out == "ama@example.com"

    def test_untouched_when_no_trigger(self, tmp_path):
        text = "nothing to expand here"
        out, used = self._set(tmp_path).expand(text)
        assert out == text and used == []

    def test_multiple_triggers(self, tmp_path):
        out, used = self._set(tmp_path).expand(
            "use my email address and standard signoff")
        assert "ama@example.com" in out and "Best regards" in out
        assert len(used) == 2

    def test_longest_trigger_wins(self, tmp_path):
        from context.snippets import SnippetSet
        s = SnippetSet(tmp_path / "s.json")
        s.add("my email", "short@x.com")
        s.add("my work email address", "work@x.com")
        out, _ = s.expand("my work email address")
        assert out == "work@x.com"

    def test_persists(self, tmp_path):
        from context.snippets import SnippetSet
        self._set(tmp_path)
        assert SnippetSet(tmp_path / "s.json").get("my email address") == "ama@example.com"

    def test_remove(self, tmp_path):
        s = self._set(tmp_path)
        assert s.remove("my email address")
        assert s.expand("my email address")[0] == "my email address"

    def test_rejects_overlong_trigger(self, tmp_path):
        from context.snippets import SnippetSet
        s = SnippetSet(tmp_path / "s.json")
        assert not s.add("one two three four five six seven eight nine", "x")

    def test_rejects_empty(self, tmp_path):
        from context.snippets import SnippetSet
        s = SnippetSet(tmp_path / "s.json")
        assert not s.add("", "x") and not s.add("trigger", "")

    def test_disabled_does_nothing(self, tmp_path):
        from context.snippets import SnippetSet
        s = SnippetSet(tmp_path / "s.json", enabled=False)
        s.enabled = True
        s.add("trigger", "expansion")
        s.enabled = False
        assert s.expand("trigger")[0] == "trigger"

    def test_malformed_file_is_not_fatal(self, tmp_path):
        from context.snippets import SnippetSet
        f = tmp_path / "s.json"
        f.write_text("{ not json")
        assert len(SnippetSet(f)) == 0


# ── refiner integration ───────────────────────────────────────────────────

class TestRefinerProfiles:
    def test_profile_instruction_is_appended_not_substituted(self):
        """Base safety rules must survive whatever a profile asks for."""
        r = Refiner(api_key="k")
        prompt = r._system_prompt("Write shell commands only.")
        assert "Write shell commands only." in prompt
        assert "negation" in prompt.lower()

    def test_no_profile_leaves_prompt_unchanged(self):
        r = Refiner(api_key="k")
        assert r._system_prompt("") == r._system_prompt()

    def test_app_context_appears_in_input(self):
        r = Refiner(api_key="k")
        out = r._build_input("hello", None, None, "Slack \u00b7 [Edit]")
        assert "Slack" in out

    def test_restructure_flag_defaults_to_articulate(self):
        r = Refiner(api_key="")
        # No key -> deterministic cleanup, but the call must accept the args.
        assert asyncio.run(r.refine(
            "hello world", profile_instruction="x", allow_restructure=True
        )) == "Hello world."


# ── streaming ─────────────────────────────────────────────────────────────

class TestStreaming:
    def test_no_key_fails_cleanly(self):
        s = StreamingSession(api_key="")
        assert asyncio.run(s.open()) is False
        assert s.error and not s.ok

    def test_feed_before_open_is_safe(self):
        s = StreamingSession(api_key="k")
        s.feed(np.zeros(160, dtype=np.float32))  # must not raise

    def test_partial_supersedes_then_final_appends(self):
        """Vendor: "Within a turn, each Turn message supersedes the previous
        one. Render the latest transcript; do not append."" ("""
        s = StreamingSession(api_key="k", format_turns=True)
        s._handle_turn({"turn_order": 0, "transcript": "hello",
                        "end_of_turn": False, "turn_is_formatted": False})
        s._handle_turn({"turn_order": 0, "transcript": "hello there",
                        "end_of_turn": False, "turn_is_formatted": False})
        assert s._live_text() == "hello there"
        s._handle_turn({"turn_order": 1, "transcript": "and more",
                        "end_of_turn": False, "turn_is_formatted": False})
        assert s._live_text() == "hello there and more"

    def test_final_turn_is_the_text_of_the_result(self):
        s = StreamingSession(api_key="k", format_turns=True)
        s._handle_turn({"turn_order": 0, "transcript": "hello there",
                        "end_of_turn": True, "turn_is_formatted": True,
                        "words": [{"text": "hello", "confidence": 0.99},
                                  {"text": "there", "confidence": 0.98}]})
        r = s._result()
        assert r.text == "hello there"
        assert len(r.words) == 2, "words must be taken once per final turn"

    def test_unformatted_final_is_not_double_counted(self):
        """With format_turns on, each turn arrives twice: raw then formatted."""
        s = StreamingSession(api_key="k", format_turns=True)
        s._handle_turn({"turn_order": 0, "transcript": "hello there",
                        "end_of_turn": True, "turn_is_formatted": False})
        # Not a final -- but not thrown away either: if the socket died here
        # the user keeps their words and loses the formatting.
        assert not s._turns[0].final
        assert s._result().text == "hello there"
        s._handle_turn({"turn_order": 0, "transcript": "Hello there.",
                        "end_of_turn": True, "turn_is_formatted": True})
        assert s._result().text == "Hello there."

    def test_repeated_formatted_finals_do_not_duplicate(self):
        s = StreamingSession(api_key="k", format_turns=True)
        for _ in range(3):
            s._handle_turn({"turn_order": 0, "transcript": "One time.",
                            "end_of_turn": True, "turn_is_formatted": True})
        assert s._result().text == "One time."

    def test_turns_without_turn_order_still_work(self):
        seen = []
        s = StreamingSession(api_key="k", on_partial=seen.append)
        s._handle_turn({"transcript": "hello", "end_of_turn": False})
        s._handle_turn({"transcript": "hello there", "end_of_turn": True,
                        "turn_is_formatted": True})
        s._handle_turn({"transcript": "again", "end_of_turn": False})
        assert seen[-1] == "hello there again"
        assert s._result().text == "hello there"

    def test_result_collects_word_confidences(self):
        s = StreamingSession(api_key="k")
        s._handle_turn({"turn_order": 0, "transcript": "deploy now",
                        "end_of_turn": True, "turn_is_formatted": True,
                        "words": [{"text": "deploy", "confidence": 0.4},
                                  {"text": "now", "confidence": 0.99}]})
        r = s._result()
        assert [w.text for w in r.low_confidence_words()] == ["deploy"]

    def test_empty_stream_with_error_is_failure(self):
        s = StreamingSession(api_key="k")
        s.error = "connect failed"
        assert not s._result().ok

    def test_empty_stream_without_error_is_ok(self):
        """Genuine silence over a healthy socket is not an error."""
        assert StreamingSession(api_key="k")._result().ok

    # ── model pinning (audit C10) ─────────────────────────────────────────

    def test_connection_params_name_the_model(self):
        """Unrecognised params are ignored silently by the server, and an
        omitted speech_model means "whatever the account defaults to" -- so it
        has to be on the URL, matching the batch model we fall back to."""
        s = StreamingSession(api_key="k", batch_model="universal-3-5-pro")
        params = s.connection_params()
        assert params["speech_model"] == "universal-3-5-pro"
        assert params["sample_rate"] == 16000
        assert params["encoding"] == "pcm_s16le"
        assert params["format_turns"] == "true"

    def test_batch_only_models_map_to_a_realtime_id(self):
        from stt.streaming import STREAMING_MODELS, streaming_model_for

        for batch in ("universal-3-5-pro", "universal-3-pro", "slam-1",
                      "universal-2", "", "gibberish-typo"):
            assert streaming_model_for(batch) in STREAMING_MODELS, batch

    def test_begin_echo_mismatch_degrades_the_session(self):
        """"Always check that configuration.model matches the speech_model you
        requested" -- a mismatch must not be shown to the user as if it were
        the pinned model."""
        s = StreamingSession(api_key="k", speech_model="universal-3-5-pro")
        s._handle_begin({"id": "sess-1",
                         "configuration": {"model": "universal-streaming-english"}})
        assert s.degraded and s.session_id == "sess-1"
        assert "universal-streaming-english" in (s.error or "")

    def test_begin_records_session_id_when_it_matches(self):
        s = StreamingSession(api_key="k", speech_model="universal-3-5-pro")
        s._handle_begin({"id": "abc", "configuration": {"model": "universal-3-5-pro"}})
        assert not s.degraded and s.session_id == "abc"

    def test_termination_audio_duration_is_authoritative(self):
        """The server's count of processed audio beats our sent-byte estimate
        and is what support asks about."""
        s = StreamingSession(api_key="k")
        s._started = 0.0
        s._samples_sent = 16000 * 3
        # Termination is normally applied by the receive loop; exercise it.
        s.server_duration_s = 2.4
        s._handle_turn({"turn_order": 0, "transcript": "hi",
                        "end_of_turn": True, "turn_is_formatted": True})
        assert abs(s._result().audio_duration_s - 2.4) < 1e-9

    def test_keyterms_capped_for_streaming(self):
        s = StreamingSession(api_key="k",
                             keyterms=["x" * 60] + [f"term{i}" for i in range(200)])
        import json as _json
        sent = _json.loads(s.connection_params()["keyterms_prompt"])
        assert len(sent) == 100 and all(len(t) <= 50 for t in sent)

    def test_terminate_then_drain_order(self):
        """Termination must be *sent* and the socket read before it closes --
        closing on Terminate discards the last transcript (audit C11)."""
        import asyncio as _a

        sent = []

        class FakeWS:
            def __init__(self):
                self.closed = False

            async def send(self, payload):
                sent.append(payload)

            async def close(self):
                self.closed = True

        async def run():
            s = StreamingSession(api_key="k")
            s.ok = True
            ws = FakeWS()
            s._ws = ws
            s._recv_task = _a.create_task(_a.sleep(0))
            s._send_task = _a.create_task(_a.sleep(0))
            s.ok = True                      # a healthy socket being stopped
            await s.close_and_finalise(timeout=0.3)
            return s, ws

        s, ws = _a.run(run())
        texts = [t for t in sent if t.startswith("{")]
        assert any("ForceEndpoint" in t for t in texts), sent
        assert any("Terminate" in t for t in texts), sent
        assert ws.closed
        assert texts.index('{"type": "Terminate"}') < len(texts)

    def test_abort_still_terminates(self):
        sent = []

        class FakeWS:
            async def send(self, payload):
                sent.append(payload)

            async def close(self):
                pass

        async def run():
            s = StreamingSession(api_key="k")
            s.ok = True
            s._ws = FakeWS()
            await s.abort()
            return s

        s = asyncio.run(run())
        assert any("Terminate" in t for t in sent), sent
        assert s.degraded and not s.ok

    def test_send_frames_are_50ms(self):
        from stt.streaming import CHUNK_MS, CHUNK_SAMPLES, SAMPLE_RATE

        assert CHUNK_MS == 50
        assert CHUNK_SAMPLES == SAMPLE_RATE * CHUNK_MS // 1000 == 800

    def test_control_frame_never_raises_on_dead_socket(self):
        s = StreamingSession(api_key="k")

        class Boom:
            async def send(self, _p):
                raise RuntimeError("socket closed")

        s._ws = Boom()
        assert asyncio.run(s._send_control({"type": "Terminate"})) is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
