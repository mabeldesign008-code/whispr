"""The cleanup instruction builder and the deterministic local fallback."""

from refine import (
    BASE_INSTRUCTION,
    TONE_NAMES,
    TONES,
    basic_cleanup,
    build_instruction,
    default_tone,
)


class TestDefaultTone:
    def test_known_tones_pass_through(self):
        for t in TONES:
            assert default_tone(t) == t
            assert default_tone(t.upper()) == t  # case tolerant

    def test_legacy_aliases(self):
        assert default_tone("auto") == "general"
        assert default_tone("chat_acro-fromal") == "casual"  # the typo shipped
        assert default_tone("polished") == "formal"

    def test_unknown_and_empty_fall_back(self):
        assert default_tone("") == "general"
        assert default_tone("baroque") == "general"
        assert default_tone(None) == "general"


class TestBuildInstruction:
    def test_base_rules_always_present(self):
        for tone in TONES:
            out = build_instruction(tone)
            assert "filler" in out
            assert "self-corrections" in out
            assert out.startswith(BASE_INSTRUCTION[:40])

    def test_llm_instruction_replaces_default_prompt(self):
        """llm_instruction REPLACES the server's cleanup prompt, so the
        basic cleanup duties must be restated or they stop happening."""
        out = build_instruction("casual")
        assert "remove filler words" in out

    def test_tones_are_distinct(self):
        assert build_instruction("general") != build_instruction("formal")
        assert "professional" in build_instruction("formal")
        assert build_instruction("general") == BASE_INSTRUCTION

    def test_app_instruction_appended(self):
        out = build_instruction("general", "Keep identifiers exactly as spoken.")
        assert out.endswith(" Keep identifiers exactly as spoken.")

    def test_tone_names_match_tones(self):
        assert set(TONE_NAMES) == set(TONES)

    def test_nothing_over_2048_chars(self):
        out = build_instruction("formal", "x " * 5000)
        assert len(out) <= 2048


class TestBasicCleanup:
    def test_capitalises_and_terminates(self):
        assert basic_cleanup("hello world") == "Hello world."

    def test_existing_punctuation_kept(self):
        assert basic_cleanup("really?") == "Really?"
        assert basic_cleanup("wow!") == "Wow!"

    def test_brand_casing_survives(self):
        # Raising the first letter would corrupt iPhone/macOS/eBay.
        assert basic_cleanup("iPhone is great").startswith("iPhone")

    def test_dangling_comma_replaced(self):
        assert basic_cleanup("called John,") == "Called John."

    def test_trailing_dash_stripped(self):
        assert basic_cleanup("well—") == "Well"

    def test_whitespace_collapsed(self):
        assert basic_cleanup("  hello   there  ") == "Hello there."

    def test_empty(self):
        assert basic_cleanup("") == ""
        assert basic_cleanup("   ") == ""
