# Repair record

The initial Antigravity draft is preserved in Git history at `e3b043a`. The student asked ChatGPT/Codex to repair it against the actual assignment after an independent review.

## Reproduced issues and verification

| Issue in the draft | Repair | Verification |
| --- | --- | --- |
| Provider exceptions became prepared correct answers and fabricated latency values | Evaluation retains explicit errors and measured timings; offline results are not AI accuracy | Evaluation regression tests |
| Any shared number meant agreement | Compare explicit JSON final answers; return unavailable for ambiguous/unstructured replies | Contradictory-final-answer tests |
| An incorrect final answer passed because it mentioned a correct number | Fixed rubric with manual grading for real responses | Evaluation grading tests |
| IDLE predictions counted as HOVER | Preserve IDLE/missing predictions in the confusion matrix | IDLE-only classifier test |
| A fist timer survived lost hand tracking | Process empty landmarks on every missing-hand frame | Application test: fist, absent, fist must restart timer |
| Provider errors produced blank answer panels | Render each provider's failure while preserving the other's result | Panel rendering and async result tests |

Other repairs include an explicit mouse demo, lazy MediaPipe loading, a supported configurable Claude model default, bounded smoothing history, full-answer text export, and simpler secret-protection rules. The existing PNG color export already preserved the correct colors; the redundant BGR/RGB round trip was simplified rather than credited as a discovered color bug.

## Evidence limits

Automated tests exercise software behavior using scripted inputs and mocked network responses. Synthetic gesture and smoothing outputs are simulations. These do not measure actual camera accuracy, classroom lighting robustness, live provider answer quality, cost, FPS, or API latency. The report and slides must state those limits.

The original eleven independent review checks included four passes, six reproduced bugs, and one synthetic baseline. They are separate from the repaired project's regression-test count. Use the current test run record for the latter.

## Later draft changes preserved

The shared draft advanced to commits `a48ef0d` and `6da596b` during the repair. Camera switching, smoothing controls, round-cap strokes, distance-based finger checks and observed-pose debounce were reviewed and incorporated. Speed-dependent smoothing is an explicit `--adaptive` option, leaving fixed EMA as the default for reproducible comparisons. Tests check that alpha 1 remains raw, missing hand detections reset debounce, and a failed camera switch preserves a working camera.
