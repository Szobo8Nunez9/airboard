# Airboard project report

An AI assisted webcam whiteboard. Prepared for the class presentation on 2 November 2026.

## What I made

Airboard is a Python prototype for drawing on a screen with hand movements. OpenCV captures a webcam image, MediaPipe Hands supplies 21 landmarks, and simple rules map finger combinations to drawing tools. Ink stays on a separate canvas that can appear over the video or on a white background. The program can change ink colors and save PNG images. Mouse mode lets another student try the canvas when a webcam is unavailable.

Index finger extended means draw. Index and middle extended means erase. An open hand hovers. A fist held for one second clears the canvas once. An exponential moving average smooths the fingertip. Alpha 0.5 blends the new point with the prior smoothed point, while alpha 1.0 follows raw points. Ending a stroke resets the filter. Fixed EMA is the default. An optional adaptive mode varies the gain during motion; the recorded alpha comparison uses fixed EMA.

## Notebook

Notebook mode keeps multiple pages and original ink beside editable text. Apple Vision recognizes text locally. Spacing cleanup preserves recognized wording. Users review and correct words before exporting a PDF. A generated printed-text check passed. Real handwriting accuracy remains unmeasured.

## Who it is for

The intended users are students explaining quick equations or sketches to classmates or a study group. It is a small experiment with a normal webcam, rather than a validated replacement for a drawing tablet or an accessibility aid.

## How I made it with AI

Antigravity supplied the initial implementation, sample boards, tests, and draft artifacts. ChatGPT independently reviewed the source, reproduced errors, and helped repair the program and rewrite its supporting material. Claude was part of the intended architecture and review plan, but the earlier review actions attributed to Claude were not independently verified and are not counted as completed work here.

The optional runtime exports the same board PNG to Claude and an OpenAI vision model in parallel. Each provider has its own result and error state. Real requests require keys in a local .env file. Missing keys leave interpretation unavailable rather than producing a fictional answer. Explicit demo mode labels its panel examples and makes no model request.

A comparison uses explicit final-answer fields. Matching answers can both be wrong, and mathematically equivalent notation can compare unequal. Missing or uncertain responses produce an unassessed result.

## What I checked

- Automated regression suite: 80 passed, 0 failed. Command: `python -B -m pytest -q -p no:cacheprovider`.
- Synthetic gesture trials: generated landmarks check rule behavior, with IDLE and missing predictions preserved. They do not measure real lighting or distance robustness.
- Synthetic smoothing: replaying 360 generated points across three paths at alpha 1.0 and 0.5 reduced the jitter metric while increasing geometric error. It checks filter behavior, not user handwriting quality.
- Webcam field trials: User confirmed the Terminal launch worked on 1 November. Real camera quality and accuracy remain unmeasured.
- Live AI evaluation: Not measured; no provider requests made.

The independent review reproduced six initial errors: failed API calls became prepared correct benchmark answers; incidental shared numbers became agreement; a rejected correct candidate passed the correctness grader; IDLE predictions earned HOVER credit; lost tracking preserved the clear timer; and provider failures rendered blank panels. The revised flow separates failures from results, compares explicit answers, keeps evaluation provenance, resets state on tracking loss, and displays provider errors.

Six reproduced issues repaired and covered by regression checks.

The earlier 100% accuracy claims and unsupported review-credit entries are not evidence for this report. Software tests do not establish real camera accuracy, live-model quality, response speed, or presentation frame rate.

## What is next

A clean installation and the camera setup still need validation on the presentation computer. Collect 25 observed trials per gesture with actual lighting and distance labels. Compare alpha 1.0 and 0.5 on the same captured fingertip sequence, including lag and corner shape. Run the same 15 actual board images through both live providers, retain full responses/model IDs/errors/timing, and grade them against a fixed human rubric. Count matching wrong answers separately from matching correct answers.

For the demonstration, show draw, hover, erase, clear, white background, and PNG saving. Use mouse mode if camera access fails. Show real AI answers only if requests succeed; otherwise explain the labeled demo and remaining work.

## Sources

- [MediaPipe Hands legacy solution](https://chuoling.github.io/mediapipe/solutions/hands.html)
- [OpenAI images and vision](https://developers.openai.com/api/docs/guides/images-vision?api-mode=chat)
- [Anthropic vision input guide](https://platform.claude.com/docs/en/build-with-claude/vision)
- [OpenCV image encoding](https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html)

Configured draft repository: https://github.com/Szobo8Nunez9/airboard. The repository contains the source code and submission materials. Put the actual repository link first in the online submission text. Due 2 November 2026 at 08:00, Asia/Taipei.
