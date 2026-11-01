# airboard

A webcam whiteboard for explaining an equation or sketch with hand movements. This is a university midterm prototype made with AI assistance.

The main feature is local drawing. Optional Claude and OpenAI vision requests can interpret the board. You can demonstrate the interface with a mouse and clearly labeled demo replies when no camera or API key is available.

## Start

Use Python 3.11 for the MediaPipe legacy `mp.solutions` API.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python airboard.py
```

Allow camera access to the application running Python when macOS asks. Keep the hand upright and visible. The classifier uses simple landmark relationships and can struggle with rotation, occlusion, distance or poor lighting.

For a camera-free demonstration:

```bash
python airboard.py --mouse --mock-ai
```

Drag the left mouse button to draw and the right button to erase. The header identifies **MOUSE DEMO**. Demo AI output confirms the interface works and does not interpret the image. If the camera cannot open, the app visibly switches to mouse mode.

## Gestures and controls

| Input | Action |
| --- | --- |
| Index finger extended | Draw |
| Index and middle fingers extended | Erase between the fingertips |
| Open hand with four fingers extended | Hover |
| Fist held continuously for one second | Clear once |
| `a` | Send the current white board PNG to both configured vision providers |
| Tab | Try another camera while preserving the current one if switching fails |
| `[` and `]` | Adjust smoothing alpha between 0.1 and 1.0 |
| `w` | Switch between video overlay and white board |
| `c` | Clear |
| `s` | Save the board PNG in ignored `snapshots/` |
| `1` to `5` | Change ink color |
| `x`, space, or click on the AI panel | Close the answers |
| `q` or Escape | Quit |

Only the board image, not the webcam background, is sent when you press `a`. Drawing pauses while the answer panel is open. Full replies are saved in ignored `snapshots/last_ai_answers.txt` if writable. The on-screen OpenCV text renderer supports basic ASCII; the saved text preserves the original reply.

Fixed EMA is the default. `--adaptive` enables the additional speed-dependent smoothing option from the later draft. For a fair alpha 1.0 versus 0.5 comparison, leave that option off.

## Notebook

Press `b` for notebook view, `n` for a new page, and `,` / `.` to change pages. Press `t` to recognize writing locally with Apple Vision on macOS. Review words beside the original ink. Press `e` to edit text, Escape to finish editing, `u` to undo a text change, and `j` / `k` to scroll. Press `f` to export a PDF. Pages autosave in ignored `notebooks/default/`. Recognition can misread handwriting; printed-text verification does not measure handwriting accuracy.

## Optional real AI replies

```bash
cp .env.example .env
```

Edit `.env` locally with your own provider keys. Separate API access is required. Never paste keys into code or commit them. Configure model IDs supported by your accounts. Current defaults are Claude `claude-sonnet-4-6` and OpenAI `gpt-4o`. See the official [Claude model lifecycle](https://platform.claude.com/docs/en/about-claude/model-deprecations), [Claude vision guide](https://platform.claude.com/docs/en/build-with-claude/vision), and [OpenAI vision guide](https://developers.openai.com/api/docs/guides/images-vision).

Missing keys, quota problems, timeouts, and provider failures appear in the relevant answer panel. A successful peer answer remains visible. Optional comparison uses normalized final-answer text, not numbers elsewhere in reasoning. Equivalent notation can compare unequal, and matching answers may both be wrong. The comparison is not a correctness score.

## What was checked

```bash
python -m pytest -q
python evaluation/evaluate.py
```

The tests cover state transitions, smoothing, canvas export, explicit errors, parallel dispatch, answer comparison and evaluation integrity. The default evaluation uses synthetic landmarks and generated trajectories; outputs explicitly identify those simulations. It does **not** measure actual webcam accuracy or live AI quality. Real camera trials and a live 15-board study are additional checks, documented in `evaluation/GRADING.md` and the evaluation command help.

The old draft's 100% AI/camera accuracy and precise timing/FPS claims were unsupported. They are not results of this repaired version. See [repair record](docs/repair_log.md).

## How AI helped

Antigravity supplied the first implementation and draft artifacts. ChatGPT/Codex independently reviewed that draft, reproduced six bugs, repaired the implementation and evaluation, and revised the submission materials. Claude was assigned an architecture/review role in the original plan; this repository does not claim an unverified review occurred. The student remains responsible for trying the camera, understanding the code, judging any AI answers and explaining limitations to the class.

## Submission files

- `deliverables/airboard_midterm_report.pdf`: what we made, who it is for, AI use, checks, limitations and next steps.
- `deliverables/slides/`: class demo slides.
- `SUBMISSION.txt`: draft online text. Put the actual accessible GitHub repository URL on its first line before submitting.

The assignment closes **2 November 2026 at 08:00 Asia/Taipei**. Confirm the final repository link works for the grader and contains the repaired code before submitting. Use a repository name such as `airboard` or `hello-python`, with no student name or ID. Keep the commit author anonymous as required.

## Next steps

Try a short drawing session with the real camera, then test recognition across lighting conditions and distances. Judge real AI responses against a fixed rubric if using that optional feature. Improve rotated-hand detection, answer scrolling and undo based on those observations.
