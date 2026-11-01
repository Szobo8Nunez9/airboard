"""Evidence integrity checks; fixture results must never masquerade as measurements."""
import csv
import json
from pathlib import Path
import pytest
from agents import AgentResult
from evaluation import test_ai_boards as ai
from evaluation import test_gestures as gestures
from evaluation.collect_gestures import trial_plan, observed_prediction
from evaluation.metrics import summarize_trials
from evaluation.summarize_gestures import summarize_recorded_csv


@pytest.fixture
def board_manifest(tmp_path, monkeypatch):
    image = tmp_path / 'board.png'
    image.write_bytes(b'\x89PNG\r\n\x1a\nfixture')
    manifest = [{'id': 'eq_01', 'category': 'equation', 'title': 'Fixture equation',
                 'ground_truth': 'x=6', 'path': str(image)}]
    monkeypatch.setattr(ai, 'generate_15_test_boards', lambda *args: manifest)
    return manifest


def test_idle_is_not_hover_credit():
    records = [{'ground_truth': label, 'predicted': 'IDLE'} for label in ('DRAW', 'HOVER', 'ERASE', 'CLEAR')]
    summary = summarize_trials(records, 'test_fixture')
    assert summary['overall_accuracy'] == 0
    assert summary['class_metrics']['HOVER']['recall'] == 0
    assert summary['confusion_matrix']['HOVER']['IDLE'] == 1


def test_missed_detections_remain_failures():
    summary = summarize_trials([{'ground_truth': 'DRAW', 'predicted': 'NO_HAND'}], 'webcam_observations')
    assert summary['overall_accuracy'] == 0
    assert summary['confusion_matrix']['DRAW']['NO_HAND'] == 1


def test_synthetic_output_has_provenance_and_raw_rows(tmp_path):
    summary = gestures.run_gesture_evaluation(tmp_path)
    assert summary['data_source'] == 'synthetic_landmarks'
    assert summary['camera_used'] is False
    with Path(summary['trials_csv']).open() as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 100
    assert {row['source'] for row in rows} == {'synthetic_landmarks'}
    assert 'NO_HAND' in summary['prediction_classes']


def test_observed_summary_rejects_simulated_csv(tmp_path):
    path = tmp_path / 'trials.csv'
    path.write_text('ground_truth,predicted,source\nDRAW,DRAW,synthetic_landmarks\n')
    with pytest.raises(ValueError, match='Only webcam_observation'):
        summarize_recorded_csv(path, tmp_path / 'observed')


def test_real_trial_plan_totals_25_per_gesture():
    plan = trial_plan()
    assert len(plan) == 100
    assert all(sum(row['ground_truth'] == label for row in plan) == 25 for label in ('DRAW', 'HOVER', 'ERASE', 'CLEAR'))
    assert len({(row['lighting'], row['distance']) for row in plan}) == 6


def test_clear_event_and_missing_frames_are_logged():
    assert observed_prediction('CLEAR', ['HOVER', 'CLEAR', 'IDLE', 'IDLE']) == 'CLEAR'
    assert observed_prediction('DRAW', ['NO_HAND', 'NO_HAND', 'DRAW']) == 'NO_HAND'
    assert observed_prediction('DRAW', []) == 'NO_HAND'
    assert observed_prediction('DRAW', ['DRAW', 'DRAW', 'CLEAR']) == 'CLEAR'


def test_offline_ai_benchmark_has_no_accuracy_or_fake_replies(tmp_path, board_manifest):
    summary = ai.run_ai_boards_benchmark(tmp_path)
    assert summary['status'] == 'not_measured'
    assert summary['claude_alone_accuracy'] is None
    assert summary['chatgpt_alone_accuracy'] is None
    assert summary['agreement_rate'] is None
    assert summary['board_results'][0]['claude'] is None
    assert summary['board_results'][0]['sha256']


def test_missing_live_credentials_skip_requests(tmp_path, board_manifest):
    summary = ai.run_ai_boards_benchmark(tmp_path, live=True)
    assert summary['status'] == 'not_measured'
    assert summary['data_source'] == 'not_measured'
    assert 'credentials' in summary['reason']


def test_failed_live_requests_never_become_canned_success(tmp_path, board_manifest):
    class FailedRunner:
        def run_parallel(self, *args):
            c = AgentResult('Claude', '', 0.0123, False, 'Deliberate timeout', 'fixture', 'error')
            o = AgentResult('ChatGPT', '', 0.0456, False, 'Deliberate timeout', 'fixture', 'error')
            return c, o, None
    summary = ai.run_ai_boards_benchmark(tmp_path, live=True, runner=FailedRunner())
    assert summary['claude_alone_accuracy'] is None
    record = summary['board_results'][0]
    assert record['claude']['success'] is False
    assert record['claude']['response_text'] == ''
    assert record['claude']['latency_seconds'] == 0.0123
    grades = ai.summarize_grades(summary['grading_csv'], tmp_path / 'ai_benchmark_summary.json')
    assert grades['providers']['claude']['correct_boards'] == 0
    assert grades['providers']['claude']['end_to_end_accuracy'] == 0


def test_grading_cannot_mark_failed_results_correct(tmp_path, board_manifest):
    class FailedRunner:
        def run_parallel(self, *args):
            result = AgentResult('Fixture', '', 0.1, False, 'timeout', 'fixture', 'error')
            return result, result, None
    summary = ai.run_ai_boards_benchmark(tmp_path, live=True, runner=FailedRunner())
    grading_path = Path(summary['grading_csv'])
    with grading_path.open() as file:
        rows = list(csv.DictReader(file))
    rows[0]['correctness'] = 'correct'
    with grading_path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=ai.GRADE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match='cannot be graded correct'):
        ai.summarize_grades(grading_path, tmp_path / 'ai_benchmark_summary.json')


def test_grading_is_explicit_and_requires_run_match(tmp_path, board_manifest):
    summary = ai.run_ai_boards_benchmark(tmp_path)
    path = Path(summary['grading_csv'])
    data = path.read_text().replace(summary['run_id'], 'different-run')
    path.write_text(data)
    with pytest.raises(ValueError, match='match this run'):
        ai.summarize_grades(path, tmp_path / 'ai_benchmark_summary.json')
