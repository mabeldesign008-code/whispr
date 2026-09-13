# Fix plan — WhisprFlow audit remediation

Branch `fix/audit-2026-09` off `68ef40d`. One commit per fix, each commit carrying its own
regression test. Research links are recorded per fix in `FIXLOG.md` and echoed in the commit
message.

Verification gate after every commit:
`python -m pytest -q` → `python eval/mock_api_test.py` → `xvfb-run python eval/smoke_test.py`
→ `xvfb-run python eval/import_probe.py` → `python -m pyflakes main.py audio ui stt refine context injector.py`.
Plus a re-measurement of the specific hot path the commit touches.

## Batch 1 — service contracts (unblocks everything else)  *(done)*

| Fix | Finding | Change |
|---|---|---|
| B1.1 ✅ | C1 | Model IDs env-overridable; non-200 surfaced + counted |
| B1.2 ✅ | C2 | Pin `speech_model` on the streaming socket; assert it |
| B1.3 ✅ | C5 | `Terminate` frame on every exit path |
| B1.4 ✅ | §3 | Sync API for the ≤2 min batch path; key validation on save; adaptive poll/finalise budgets |

## Batch 2 — stop losing people's text  *(done)*

| Fix | Finding | Change |
|---|---|---|
| B2.1 ✅ | C3.1/3.2 | Focus anchor: snapshot hwnd at `start_recording`, re-verify before `inject`; Command Mode re-verifies the selection |
| B2.2 ✅ | C3.3 | Check `inject()`'s return; clear `last_text_injected` on failure |
| B2.3 ✅ | C11 | Cancel (not stop) when extra keys are down at release |
| B2.4 ✅ | C4 | ~~Esc must never enter `pressed_keys`~~ (C4 retracted); shipped: Esc cancels mid-hold, `test_hotkey.py` pins the set is clean |

## Batch 3 — refinement safety

| Fix | Finding | Change |
|---|---|---|
| B3.1 | C6 | Read `finish_reason`; restructure-mode length floor; skip refinement on very long takes |
| B3.2 | C7 | Clear refinement history on cancel; key it per app |
| B3.3 | A3 | Guard: antonym pairs, word-boundary dictionary match, added-number rule, digit-in-identifier exclusion |
| B3.4 | A2 | High-pass + levelling on streaming frames; full keyterms to the streaming model |
| B3.5 | A1 | Remove the VAD discard gate |
| B3.6 | C13 | `basic_cleanup` punctuation rules |

## Batch 4 — performance

| Fix | Finding | Change |
|---|---|---|
| B4.1 | P2/P3 | Cursor-based `tail_since`; block pass-through in `_send_loop` |
| B4.2 | P1 | Overlay: 1× drawing, cached static layer, itemconfig instead of PIL rebuild, geometry diff |
| B4.3 | P4 | Warm `scipy.signal.lfilter` at import |
| B4.4 | Memory | int16 canonical take; bound `last_audio`; xrun reaction |
| B4.5 | P5/C9 | `_frame_energies(audio, sample_rate)` |

## Batch 5 — robustness, privacy, hygiene

| Fix | Finding | Change |
|---|---|---|
| B5.1 | C8 | Close abandoned `httpx` clients; real shutdown instead of `os._exit(0)`; atomic config writes |
| B5.2 | C12/S8 | UIA per-thread initializer + failure recovery + honest status; log `uia_failures` |
| B5.3 | S6 | Clipboard change-detection instead of fixed sleeps |
| B5.4 | S7 | `on_press`/`on_release` enqueue only |
| B5.5 | S2 | Remove the CWD `.env` override path; document |
| B5.6 | S4 | Learner retention: min-count floor, secret-shape filter, age cutoff, wipe action |
| B5.7 | S1 | Privacy section in README/docs + "what gets sent" disclosure |
| B5.8 | Q2 | README URLs/test count; `__version__` wired into title, log, fatal dialogs, spec |
| B5.9 | C10/U1/U2 | Pill: dead `if`, hit regions vs drawn glyphs, DPI scale factor, elapsed timer |
| B5.10 | Q1 | Replace the mirrored `_decide()` test with real state-machine tests; injector + selection tests |

## Deliberately not in this pass (needs a Windows box)

Real-hardware verification of the UIA/clipboard/hook changes, DPI behaviour on a 4K panel, the
PyInstaller EXE build, and a measured WER corpus — `eval/clips/` has no audio and no key here.
These get code + tests that pass on Linux stubs, and a written spec for what to check on Windows.
