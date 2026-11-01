# airboard evaluation evidence

## Synthetic gesture classifier checks

100 generated landmark fixtures. Fixture accuracy: 100.0%.
The classifier receives the coordinates directly. No webcam or hand detector participates.
Lighting and distance labels select noise and scale in a simulation, not real environmental conditions.
IDLE and NO_HAND remain separate prediction outcomes. Raw trial rows and a confusion matrix are included.

## Synthetic smoothing comparison

The same 360 generated samples were replayed at alpha 1.0 and 0.5.
Reduction in the chosen second-difference jitter variance: 85.0%.
This number describes the generated trajectories only. It is not a measured improvement in human handwriting.
Geometric RMSE records the tradeoff between smoothing and following the intended trajectory.
The filter has a theoretical low-frequency delay of one sample at alpha 0.5. End-to-end camera latency is not measured.

| Synthetic path | Raw RMSE px | EMA RMSE px |
| --- | ---: | ---: |
| line | 4.915 | 7.417 |
| circle | 4.706 | 9.615 |
| corner | 4.909 | 5.654 |

## Optional AI board evaluation

Status: not_measured. Live APIs were not requested. No model quality measurement exists.
Model correctness is not automatically inferred from incidental numbers or matching model answers.
Claude accuracy: Not measured. ChatGPT accuracy: Not measured.
The generated sample set contains 5 equations, 5 sketches, and 5 messy boards. Human grading instructions are in evaluation/GRADING.md.

## Remaining checks

Run the real webcam trial logger, assess the resulting conditions, verify a live provider request if using the AI feature, and rehearse the classroom demo.
