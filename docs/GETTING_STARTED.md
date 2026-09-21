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

**AssemblyAI** (required — this does the transcription *and* the cleanup:
filler removal, self-corrections, punctuation)

1. Sign up at [assemblyai.com](https://www.assemblyai.com/dashboard/signup)
   — no credit card, **$50 free credits** (then $0.62 per hour of audio)
2. Copy the key from the dashboard home page
3. Paste it into the app → **Save** (it is verified against your account)

**Groq** (optional — only if you want **Command Mode**, the
select-text-and-edit-by-voice feature)

1. Get a free key at [console.groq.com/keys](https://console.groq.com/keys)
2. Paste it → **Save**

No Groq key? Dictation works at full quality — Command Mode simply stays
off.

## 4. Dictate

Put your cursor in any text field — browser, VS Code, Slack, Word.

**Hold `Ctrl + Win`**, speak, release.

A pill appears at the bottom of the screen with a live waveform while you
talk; the finished text lands at your cursor about half a second to a
second after you stop.

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

**Up to about 1 minute 50 seconds in one take.** The Dictation API
accepts clips of at most 2 minutes, so the app stops you comfortably
before that, warns 30 seconds ahead, and transcribes what you have rather
than losing anything. For anything longer, use hands-free mode in
segments — each take pastes where the cursor is, so continuing is just
tapping the hotkey again.

**Cost:** the Dictation API is **$0.62 per hour of audio**, billed per
second, so ~80 hours of actual speech fits in the free $50 credits — for
most people that is months of dictation.

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
| `.env` | Your API keys and tone preference |
| `user_dictionary.txt` | Your terms |
| `profiles.json` | Per-app formatting rules |
| `snippets.json` | Voice-triggered text expansions |

**What leaves your machine:** each take's audio goes to AssemblyAI, which
returns both the verbatim transcript and the cleaned text in the same
response. Nothing else leaves for dictation. (Command Mode, if you add a
Groq key and use it, sends the *selected text* plus your spoken
instruction to Groq.)

The app reads your **foreground window's process name** (e.g.
`chrome.exe`) to pick a formatting profile. It does **not** read your
window titles, screenshot your screen, read field contents, or log your
keystrokes.

Toggles you can add to `%APPDATA%\WhisprFlow\.env`:

```
WHISPRFLOW_TONE=general      # general | casual | formal
WHISPRFLOW_MIC_DEVICE=       # pin a specific microphone
WHISPRFLOW_DEBUG=0           # verbose logging
```

---

## Troubleshooting

**"No microphone"** — another app has exclusive control of the mic, or
Windows mic permission is off. Check *Settings → Privacy → Microphone*.

**"Check API key"** — the key was rejected. Re-copy it; keys have no
spaces and aren't wrapped in quotes.

**"Out of credit"** — the free $50 is used up. Add billing in the
AssemblyAI dashboard (pay-as-you-go, no subscription; the Dictation API
is $0.62 per hour of audio).

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
