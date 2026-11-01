# airboard project instructions

## Assignment and scope

Build one shareable idea using AI. Submit a GitHub repository link on the first line of the online text, a Word or PDF report, and class presentation slides by **2 November 2026 at 08:00 Asia/Taipei**. No late submissions. The repository name and commit author must not contain the student's name or ID.

Airboard is a Python webcam whiteboard. Its main feature is drawing with hand gestures. Optional Claude and OpenAI APIs interpret a board image. API access is not needed for drawing or for the labeled mouse/interface demo.

## Read before editing

- Read this file first. Follow the user's current instructions over older role assignments.
- Work in a separate branch or copy when another tool is editing. Never edit the same file simultaneously.
- Antigravity supplied the initial draft. Claude's intended role is architecture/code review. ChatGPT/Codex performed an independent review and was subsequently asked by the user to repair the draft. These are roles, not proof of any unrecorded work.
- During this repair, Codex owns runtime/core/docs integration, a delegated evaluation task owns evaluation and provider tests, a report task owns the PDF report, and a slides task owns the presentation folder. Coordinate before changing ownership.

## Contracts

- `core/hand_tracking.py`: MediaPipe 0.10.14 legacy `mp.solutions` API, Python 3.11.
- `core/gestures.py`: upright hand; index only DRAW, index and middle ERASE, four extended fingers HOVER, continuous fist for one second CLEAR. Empty landmarks reset the hold timer.
- `core/smoothing.py`: EMA; alpha 1 is raw, alpha .5 reduces high-frequency changes but introduces lag. Reset at a new stroke.
- `core/canvas.py`: separate BGR stroke buffer; export a standard PNG on white using OpenCV's BGR encoder input.
- `agents.py`: each provider implements `ask(png) -> str`. `ask_both` returns two `AgentResult` values and an optional answer comparison. `None` means comparison unavailable. Matching final answers do not establish correctness.
- Missing credentials and provider failures are errors, not successful interpretations. Explicit mock mode is labeled DEMO and makes no image-understanding claim.
- Only a user pressing `a` or explicitly selecting a live benchmark should initiate API requests.

## Evidence rules

- Separate synthetic simulations, mocked software tests, manual camera trials, and actual API measurements.
- Never replace a failed request with a prepared correct answer. Never invent timings, camera accuracy, FPS, user feedback, or agent contributions.
- Preserve raw trial outcomes, full model replies, errors, configured model IDs, and measured elapsed times when collecting evidence.
- Do not report a bug fixed until a relevant check verifies it. Keep limitations in the report and slides.

## Keys and Git

Create `.gitignore` before adding files. Put keys only in a local `.env`. Commit blank `.env.example`. Ignore `.env*`, virtual environments, personal snapshots, and caches. Never log request headers or credentials.

Use a neutral commit identity. This repair copy uses `airboard contributor <airboard@example.invalid>`. Publishing, deletion, or changing GitHub visibility must follow the user's current request. The current repair is local until reviewed.

## Confirmed work log

| Date | Source | Confirmed action | Evidence |
| --- | --- | --- | --- |
| 30 October 2026 | User supplied draft attributed to Antigravity | Initial app, evaluation scripts, reports and slides | Original draft commit `e3b043a` |
| 30 October 2026 | ChatGPT/Codex | Independent offline review found six reproducible bugs | Review evidence described in `docs/repair_log.md` |
| 30 October 2026 | ChatGPT/Codex and delegated repair tasks | User authorized fixes to code and submission materials | Current repair branch, automated tests and honest evaluation outputs |

Earlier retrospective claims that Claude or ChatGPT had already reviewed specific changes, and exact FPS/API timings without artifacts, were removed because they were not verified. No claim is made that Claude performed an architectural review during this repair.
