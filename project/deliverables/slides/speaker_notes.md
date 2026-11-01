## Slide 1

Introduce airboard as a university prototype that turns a webcam into a drawing input. It is meant for sharing a quick sketch in class. The pictured house is a generated sample board from the project, not a camera result or an AI answer. Source: evaluation/generate_boards.py and evaluation/test_boards/sk_02_house.png.

## Slide 2

The intended users are students and classroom presenters. These are intended use cases, not results from a user study. Ask the class to imagine explaining a diagram without switching to a drawing tablet. The project uses an ordinary webcam. Mouse mode is a fallback for a demonstration when the camera cannot start.

## Slide 3

Show each pose slowly with the palm facing the camera and fingers upright. The classifier uses image-space finger positions, so rotation, occlusion, lighting and distance can affect it. A continuously observed fist clears once after one second. Tracking loss restarts that hold. Source: core/gestures.py and airboard.py.

## Slide 4

Walk through the implementation in four stages. MediaPipe supplies 21 hand landmarks. The gesture classifier chooses a mode. The exponential moving average blends the current fingertip with the preceding smoothed point. OpenCV writes the strokes to a separate canvas and composites the display. Alpha 1.0 follows raw coordinates, while 0.5 blends equally. Smoothing trades responsiveness for reduced jitter; no real camera improvement percentage is claimed. Sources: core/hand_tracking.py, core/smoothing.py, core/canvas.py. MediaPipe Hands: https://chuoling.github.io/mediapipe/solutions/hands.html.

## Slide 5

Press b for notebook view. Press n for a new page and comma or period to turn pages. Press t for Apple Vision recognition on macOS. Original ink stays available beside editable text. Press e to edit, Escape to finish, u to undo a text change, and f to export a PDF. Recognition runs locally. A printed-text fixture passed. Handwriting quality still needs a real writing trial. Sources: core/notebook.py, core/handwriting.py, core/notebook_export.py and tests/test_notebook.py.

## Slide 6

Antigravity supplied the initial implementation and report/slides draft. ChatGPT independently inspected the code, reproduced errors, and helped repair them. Claude has a planned architecture-review role. The final log records only verified actions, so the deck does not credit unconfirmed work as completed. Sources: user conversation, independent review evidence, AGENTS.md after repair.

## Slide 7

Press a to submit the board PNG to Claude and OpenAI in parallel. Each provider needs its own API key in .env and account access to the configured model. Missing credentials and network errors are visible. --mock-ai exercises the interface only and makes no interpretation claim. A matching final answer is not proof of correctness. No live provider accuracy or latency is claimed in this project report. Sources: agents.py. OpenAI vision: https://developers.openai.com/api/docs/guides/images-vision?api-mode=chat. Claude vision: https://platform.claude.com/docs/en/build-with-claude/vision.

## Slide 8

The software regression tests check timer reset, smoothing, PNG export, API error handling, final-answer comparison and concurrent dispatch. Synthetic gesture and trajectory experiments are labeled simulations. Real camera accuracy, lighting/distance robustness and live-model quality still require field trials. These checks establish specific software behavior rather than empirical camera or model accuracy. Sources: tests/, evaluation/, and the repair verification output.

## Slide 9

Demo script: launch python airboard.py. Show draw, hover, erase, and a continuous one-second fist hold. Press w to show the whiteboard, then s to save. If camera permission or tracking is unavailable, use python airboard.py --mouse --mock-ai and clearly call it mouse/demo mode. Press a to show real provider output only if keys and live access have been checked; otherwise show the labeled demo or unavailable state. Next: record actual trials by lighting and distance, compare the same motion at alpha 1.0 and 0.5, and grade complete provider replies against a fixed rubric.