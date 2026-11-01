"""Optional live board capture and explicit human grading. Never invent replies."""
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys
import uuid
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents import ParallelDualAgentRunner
from evaluation.generate_boards import generate_15_test_boards

GRADE_FIELDS = ['run_id', 'board_id', 'provider', 'model', 'mode', 'success', 'ground_truth',
                'response_text', 'error', 'correctness', 'grader_notes']


def run_ai_boards_benchmark(output_dir='evaluation/results', *, live=False, runner=None, boards_dir=None):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    board_folder = Path(boards_dir) if boards_dir else Path('evaluation/test_boards')
    manifest = generate_15_test_boards(str(board_folder))
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
    summary = {
        'run_id': run_id, 'timestamp': datetime.now(timezone.utc).isoformat(),
        'data_source': 'live_api' if live else 'not_measured', 'status': 'not_measured',
        'total_boards': len(manifest), 'claude_alone_accuracy': None,
        'chatgpt_alone_accuracy': None, 'agreement_rate': None,
        'correctness_method': 'Human grading of full replies against a fixed rubric',
        'reason': 'Live APIs were not requested. No model quality measurement exists.',
        'board_results': [],
    }
    if live and runner is None:
        runner = ParallelDualAgentRunner()
        providers = [runner.claude, runner.openai]
        if any(p.mock_mode or not p.api_key or p.api_key.startswith('your_') for p in providers):
            summary['reason'] = 'Both live API credentials are required, and MOCK_VISION_APIS must be false. No calls made.'
            summary['data_source'] = 'not_measured'
            live = False
    if live:
        summary['status'] = 'awaiting_human_grading'
        summary['reason'] = 'Replies and failures are recorded. Correctness remains unassessed until human grading.'
    grading_rows = []
    for board in manifest:
        data = Path(board['path']).read_bytes()
        record = {**board, 'sha256': hashlib.sha256(data).hexdigest()}
        if live:
            claude, openai, agreement = runner.run_parallel(data)
            record.update({'claude': asdict(claude), 'chatgpt': asdict(openai), 'agreement': agreement})
        else:
            record.update({'claude': None, 'chatgpt': None, 'agreement': None})
        summary['board_results'].append(record)
        for provider in ('claude', 'chatgpt'):
            response = record[provider] or {}
            grading_rows.append({
                'run_id': run_id, 'board_id': board['id'], 'provider': provider,
                'model': response.get('model', ''), 'mode': response.get('mode', 'not_measured'),
                'success': str(response['success']).lower() if 'success' in response else '',
                'ground_truth': board['ground_truth'], 'response_text': response.get('response_text', ''),
                'error': response.get('error') or '', 'correctness': '', 'grader_notes': '',
            })
    if live:
        comparable = [b['agreement'] for b in summary['board_results'] if b['agreement'] is not None]
        summary['agreement_rate'] = sum(comparable) / len(comparable) if comparable else None
        summary['agreement_assessed_boards'] = len(comparable)
    grading_path = output / f'ai_grading_{run_id}.csv'
    with grading_path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=GRADE_FIELDS)
        writer.writeheader()
        writer.writerows(grading_rows)
    summary['grading_csv'] = grading_path.as_posix()
    (output / 'ai_benchmark_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f"AI evaluation: {summary['status']}. {summary['reason']}")
    return summary


def summarize_grades(grading_csv, benchmark_json):
    """Use saved provider results as authority; CSV edits cannot turn errors into answers."""
    benchmark = json.loads(Path(benchmark_json).read_text())
    with Path(grading_csv).open(newline='') as file:
        rows = list(csv.DictReader(file))
    expected = {(b['id'], provider): b[provider] for b in benchmark['board_results']
                for provider in ('claude', 'chatgpt')}
    seen = set()
    grades = {}
    for row in rows:
        key = row['board_id'], row['provider']
        if row['run_id'] != benchmark['run_id'] or key not in expected or key in seen:
            raise ValueError('Grading rows must match this run exactly, without duplicates')
        seen.add(key)
        grade = row['correctness'].strip().lower()
        if grade not in ('', 'correct', 'incorrect', 'unassessed'):
            raise ValueError('Correctness must be correct, incorrect, or unassessed')
        result = expected[key]
        eligible = result is not None and result['success'] and result['mode'] == 'live'
        if grade == 'correct' and not eligible:
            raise ValueError('An error, demo, or unmeasured result cannot be graded correct')
        grades[key] = (grade == 'correct') if grade in ('correct', 'incorrect') and eligible else (
            False if result is not None and result['mode'] == 'error' else None)
    if seen != set(expected):
        raise ValueError('Missing grading rows')
    output = {'run_id': benchmark['run_id'], 'correctness_method': 'Human grading', 'providers': {}}
    for provider in ('claude', 'chatgpt'):
        values = [grades[(b['id'], provider)] for b in benchmark['board_results']]
        assessed = [value for value in values if value is not None]
        output['providers'][provider] = {
            'total_boards': len(values), 'assessed_boards': len(assessed),
            'correct_boards': sum(assessed),
            'end_to_end_accuracy': sum(assessed) / len(values) if len(assessed) == len(values) else None,
        }
    taxonomy = {'matching_answers_both_correct': 0, 'matching_answers_at_least_one_incorrect': 0,
                'different_answers_at_least_one_correct': 0, 'different_answers_both_incorrect': 0,
                'unassessed': 0}
    for board in benchmark['board_results']:
        a, b = grades[(board['id'], 'claude')], grades[(board['id'], 'chatgpt')]
        agreement = board['agreement']
        if a is None or b is None or agreement is None:
            key = 'unassessed'
        elif agreement:
            key = 'matching_answers_both_correct' if a and b else 'matching_answers_at_least_one_incorrect'
        else:
            key = 'different_answers_at_least_one_correct' if a or b else 'different_answers_both_incorrect'
        taxonomy[key] += 1
    output['agreement_and_correctness'] = taxonomy
    output['interpretation'] = 'Matching final answers do not prove correctness. This is a small demonstration set.'
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true', help='Explicitly permit real provider requests (may incur API charges)')
    parser.add_argument('--output-dir', default='evaluation/results')
    parser.add_argument('--grade-csv', help='Summarize completed human grading instead of making API requests')
    parser.add_argument('--benchmark-json', default='evaluation/results/ai_benchmark_summary.json')
    args = parser.parse_args()
    if args.grade_csv:
        summary = summarize_grades(args.grade_csv, args.benchmark_json)
        destination = Path(args.output_dir) / 'ai_human_grading_summary.json'
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps(summary, indent=2))
    else:
        run_ai_boards_benchmark(args.output_dir, live=args.live)


if __name__ == '__main__':
    main()
