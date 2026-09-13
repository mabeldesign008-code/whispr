# Getting started

## 1. Download

Grab **WhisprFlow.exe** from the
[latest release](https://github.com/mabeldesign008-code/flow/releases/latest).

No installer, no Python — one file. Put it anywhere, e.g. `C:\Tools\`.

## 2. First run

Double-click it. Windows will show:

> **Windows protected your PC** — Microsoft Defender SmartScreen prevented
> an unrecognised app from starting.

Click **More info → Run anyway**.

This appears because the binary isn't code-signed (a certificate costs
~$300/year). If you'd rather verify it yourself, the exact build is
reproducible from source — see [Build it yourself](#build-it-yourself).

## 3. Add an API key

The settings window opens on first launch.

**AssemblyAI** (required — this does the transcription)

1. Sign up at [assemblyai.com](https://www.assemblyai.com/dashboard/signup)
   — no credit card, **$50 free credit ≈ 185 hours**
2. Copy the key from the dashboard home page
3. Paste it into the app → **Save**

**Groq** (optional — AI cleanup: punctuation, filler removal, formatting)

1. Get a free key at [console.groq.com/keys](https://console.groq.com/keys)
2. Paste it → **Save**

Without Groq you still get accurate transcripts, just with basic
capitalisation instead of AI polish.

## 4. Dictate

Put your cursor in any text field — browser, VS Code, Slack, Word.

**Hold `Ctrl + Win`**, speak, release.

A pill appears at the bottom of the screen. Your words show up live while
you talk, then land at your cursor about half a second after you stop.

| Action | How |
|---|---|
| **Dictate (short)** | **Hold** `Ctrl + Win`, speak, release |
| **Command Mode** | Select text, `Ctrl + Shift + Win`, speak an instruction |
| **Dictate (long)** | **Tap** `Ctrl + Win`, speak hands-free, tap again to finish |
| Cancel while locked | `Esc` |
| Cancel mid-sentence | Click the **×** on the left of the pill |
| Stop early | Click the **red square** on the right |
| Undo | `Ctrl + Alt + Z` |
| Retry after an error | Click the pill |
| Settings | Double-click the tray icon |

### Hands-free mode

Holding the keys is fine for a sentence. For anything longer, **tap** the
hotkey instead of holding it:

- **Tap** `Ctrl + Win` (press and release quickly) → recording locks on.
  The stop button gains a blue ring so you can see it is still listening.
- Speak for as long as you like — hands completely free.
- **Tap** `Ctrl + Win` again to finish, or press `Esc` to throw it away.

Same keys, no new shortcut to remember. The app decides which you meant
from how long you held them: under 0.35 s is a tap, longer is a hold.

### Command Mode — edit text by voice

Select any text in any app, press **`Ctrl + Shift + Win`**, and speak what
you want done to it. The selection is rewritten in place.

| Say | Result |
|---|---|
| "make this more formal" | Casual note becomes professional prose |
| "turn this into bullet points" | Paragraph becomes a list |
| "translate to French" | Rewritten in French |
| "fix the grammar" | Typos and grammar only, nothing else changed |
| "add a docstring" | Docstring inserted into a selected function |
| "make it shorter" | Trimmed, every point kept |

The pill turns **violet** for Command Mode, so it is never confused with
dictation — one inserts text, the other replaces it.

This works in Gmail, VS Code, Slack, Word, Notion, browsers — anywhere you
can select text. It is the feature Wispr Flow charges $12/month for.

### Snippets — say a phrase, get canned text

Open **Snippets → Edit snippets** and add trigger phrases:

```json
{
  "snippets": [
    { "trigger": "my email address", "text": "you@example.com" },
    { "trigger": "standard signoff", "text": "Best regards,\nYour Name" }
  ]
}
```

Say the trigger while dictating and it expands. Triggers match regardless
of case or punctuation, and work mid-sentence: *"send it to my email
address please"* becomes *"send it to you@example.com please"*.

No AI involved, so it costs nothing and adds no latency.

### How long can I speak?

**Up to 30 minutes in one take.** That is roughly 4,500 spoken words.

The limit is local memory, not the service — audio is buffered in RAM at
about 3.8 MB per minute, so 30 minutes is ~115 MB. You will get a warning
in the activity log 30 seconds before the cap, and the app stops and
transcribes automatically rather than losing anything.

For reference, everything else in the chain is far more generous:

| Limit | Value |
|---|---|
| WhisprFlow buffer | 30 min |
| AssemblyAI streaming session | 3 hours |
| AssemblyAI upload | 10 hours / 2.2 GB |

Cost is about **$0.10 per hour** of audio, so the free $50 credit covers
roughly 185 hours.

The app lives in your system tray. Closing the window hides it; quit from
the tray menu.

---

## Make it accurate for *your* vocabulary

This is the single biggest improvement available, and it takes a minute.

Open **Dictionary** in the settings window and add names, product names,
acronyms and jargon — one per line:

```
Kubernetes
Grafana
Accra
mabeldesign
```

These go straight to the recogniser as keyterms, so it stops guessing.
The app also **suggests terms automatically**: if an unusual word keeps
coming up, it appears under *Suggested* with an **Add** button. Nothing is
added without your approval.

## Per-app formatting

WhisprFlow adapts to the app you're in — no configuration needed:

| Where you are | What you say | What you get |
|---|---|---|
| VS Code | "function called get user data" | `get_user_data` |
| Terminal | "git checkout dash b feature slash login" | `git checkout -b feature/login` |
| Slack | "hey can you look at the pr" | "hey can you look at the PR" |
| Outlook | "um so i wanted to follow up" | "I wanted to follow up." |

Add your own in `%APPDATA%\WhisprFlow\profiles.json`.

---

## Settings and privacy

Everything is stored locally in `%APPDATA%\WhisprFlow\`:

| File | Contents |
|---|---|
| `.env` | Your API keys |
| `user_dictionary.txt` | Your terms |
| `profiles.json` | Per-app formatting rules |
| `learned.json` | Auto-learn candidates |

**What leaves your machine:** the audio you dictate goes to AssemblyAI;
the resulting text goes to Groq if refinement is on. Nothing else.

The app reads your **foreground window's name and title** to pick a
formatting profile. It does **not** screenshot your screen, and it does
**not** log your keystrokes. Reading the focused text field's existing
content is off by default.

To disable context entirely, add to `%APPDATA%\WhisprFlow\.env`:

```
WHISPRFLOW_CONTEXT=0
```

Other toggles:

```
WHISPRFLOW_STREAMING=1    # live text while you speak
WHISPRFLOW_AUTOLEARN=1    # suggest dictionary terms
WHISPRFLOW_READ_FIELD=0   # read the focused field's text
WHISPRFLOW_DEBUG=0        # verbose logging
```

---

## Troubleshooting

**"No microphone"** — another app has exclusive control of the mic, or
Windows mic permission is off. Check *Settings → Privacy → Microphone*.

**"Check API key"** — the key was rejected. Re-copy it; keys have no
spaces and aren't wrapped in quotes.

**"Out of credit"** — the free $50 is used up. Add billing at AssemblyAI,
or switch to the cheaper model by adding `ASSEMBLYAI_MODEL=universal-2`
to your `.env`.

**Nothing pastes** — some apps block synthetic keystrokes. Try running
WhisprFlow as administrator if the target app is elevated.

**First syllable missing** — shouldn't happen; the mic runs continuously
with a 500 ms pre-roll. If it does, open an issue with your mic model.

**SmartScreen keeps blocking it** — right-click the EXE → *Properties* →
tick **Unblock** → *Apply*.

---

## Build it yourself

If you'd rather not trust a downloaded binary:

```bash
git clone https://github.com/mabeldesign008-code/flow.git
cd flow
pip install -r requirements.txt
pip install pyinstaller
pyinstaller WhisprFlow.spec --noconfirm --clean
```

The EXE lands in `dist/`. This is byte-for-byte what the release workflow
runs — see `.github/workflows/release.yml`.

Or just run from source:

```bash
python setup_stt.py    # configure your key
python main.py
```
