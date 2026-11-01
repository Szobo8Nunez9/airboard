# Evaluation scope and grading

`python evaluation/evaluate.py` runs offline simulations only. Generated coordinates are classifier fixtures. Their scale and noise labels do not measure actual distance or room lighting. Generated test boards are samples, not captured student handwriting. No AI quality claim follows from an offline run.

## Real gesture trials

Run `python evaluation/collect_gestures.py`. It offers 100 trials, 25 per requested gesture, across six observer-labeled lighting/distance combinations. The first condition has five repetitions per gesture, and the other five have four. Press Space only after arranging the displayed condition and gesture. A timed trial starts, and the software records its prediction. Press Q or Escape to stop early. Completed trials remain in the CSV. No raw camera images are stored.

Room lighting and distance labels come from the observer. Record how you arranged those conditions in the report. They are not automatic lux or distance measurements. Every missed detection and IDLE prediction remains a failure in the matrix. Any CLEAR event becomes the recorded prediction, including an unwanted clear during another requested gesture. Otherwise, the logger uses the most common frame outcome, including missed detections.

Create a real-trial summary with `python evaluation/summarize_gestures.py evaluation/real_gesture_trials.csv`. Keep these outputs separate from the synthetic results.

## Optional live AI study

Only `python evaluation/test_ai_boards.py --live` requests model replies. Both API keys must be configured in `.env` and demo mode disabled. Requests may incur provider charges. No keys are included in output files. The JSON records full replies, failure states, measured elapsed time, configured model IDs, and board image hashes. Failures never receive replacement answers.

For each generated grading CSV, assess the complete reply against the supplied expected result and the corresponding image. Use `correct`, `incorrect`, or `unassessed` in the correctness column. Add reasons in grader_notes. Do not score a response from a matching incidental number or label alone.

- Equation: the final solution must be mathematically correct and correspond to the equation shown. Reject incorrect final answers even if the correct number occurs elsewhere in the explanation.
- Sketch: identify the main depicted object and the relationships relevant to the image. Do not accept a generic description that could fit unrelated boards.
- Messy board: assess the intended visible content and acknowledge ambiguity. A confident invented interpretation is not correct.
- Failed, demo, and unmeasured calls cannot receive a correct grade. Errors count as failures in end-to-end accuracy. Unmeasured or ungraded work leaves accuracy unavailable.

Summarize a completed grading sheet with `python evaluation/test_ai_boards.py --grade-csv PATH --benchmark-json evaluation/results/ai_benchmark_summary.json`. Match the CSV to its saved run. The summary counts agreement and correctness separately. Exact normalized final-answer comparison can miss equivalent notation; it is a comparison aid, not an independent grader.
