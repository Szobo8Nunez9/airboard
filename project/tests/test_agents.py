"""Regression checks for failure handling, answer comparison, and concurrency."""
import json
import threading
import pytest
import requests
import agents
from agents import (ClaudeVisionAgent, OpenAIVisionAgent, ParallelDualAgentRunner,
                    AgentResult, ask, display_response)
from core.canvas import AirCanvas


@pytest.fixture
def png():
    canvas = AirCanvas(100, 100)
    canvas.draw_segment((10, 10), (90, 90))
    return canvas.get_png_bytes()


def answer(final='x=6', observation='2x + 7 = 19', uncertain=False):
    return json.dumps({'kind': 'equation', 'observation': observation,
                       'final_answer': final, 'explanation': 'Step 1. Work checked.', 'uncertain': uncertain})


@pytest.mark.parametrize('provider', [ClaudeVisionAgent, OpenAIVisionAgent])
def test_missing_credentials_are_errors_not_automatic_demos(provider, png):
    agent = provider(api_key='')
    assert agent.mock_mode is False
    with pytest.raises(RuntimeError, match='Add .*_API_KEY'):
        agent.ask(png)


def test_explicit_demo_has_no_interpretation_or_consensus(monkeypatch, png):
    monkeypatch.setenv('MOCK_VISION_APIS', 'true')
    c, o, agreement = ParallelDualAgentRunner().run_parallel(png)
    assert c.success and o.success
    assert c.mode == o.mode == 'demo'
    assert agreement is None
    assert json.loads(c.response_text)['uncertain'] is True
    assert 'DEMO ONLY' in display_response(c)
    assert 'No API request was made' in display_response(o)


def test_contradictory_final_answers_cannot_share_equation_numbers():
    assert ParallelDualAgentRunner.check_agreement(answer('x=6'), answer('x=5')) is False


def test_matching_explicit_final_answers_compare_equal():
    assert ParallelDualAgentRunner.check_agreement(answer('x = 6'), answer('X=6')) is True


def test_unstructured_or_uncertain_responses_are_unassessed():
    assert ParallelDualAgentRunner.check_agreement('The answer is 6', 'The answer is 6') is None
    assert ParallelDualAgentRunner.check_agreement(answer(), answer(uncertain=True)) is None


def test_number_in_rejected_candidate_does_not_override_final_answer():
    wrong = answer('x=5', observation='Candidate x=6 was rejected; equation has 2, 7, 19.')
    assert ParallelDualAgentRunner.check_agreement(answer('x=6'), wrong) is False


def test_parallel_workers_reach_barrier(png):
    barrier = threading.Barrier(2)
    class Provider:
        name = 'Fixture'
        model = 'fixture-model'
        mock_mode = False
        def ask(self, *args):
            barrier.wait(timeout=2)
            return answer()
    c, o, agree = ParallelDualAgentRunner(Provider(), Provider()).run_parallel(png)
    assert c.success and o.success and agree is True
    assert c.model == 'fixture-model'


def test_one_provider_failure_preserves_other_answer_and_visible_error(png):
    class Broken:
        def ask(self, *args):
            raise requests.Timeout('Deliberate fixture timeout')
    class Working:
        def ask(self, *args):
            return answer()
    c, o, agree = ParallelDualAgentRunner(Broken(), Working()).run_parallel(png)
    assert not c.success and c.mode == 'error'
    assert o.success and 'x=6' in display_response(o)
    assert 'timed out' in display_response(c)
    assert agree is None


def test_unified_interface_validates_provider(monkeypatch, png):
    monkeypatch.setenv('MOCK_VISION_APIS', 'true')
    assert isinstance(ask(png, 'claude'), str)
    assert isinstance(ask(png, 'chatgpt'), str)
    with pytest.raises(ValueError):
        ask(png, 'unknown')


@pytest.mark.parametrize('provider', [ClaudeVisionAgent, OpenAIVisionAgent])
def test_png_validation_rejects_non_images(provider):
    with pytest.raises(ValueError, match='PNG'):
        provider(api_key='fixture-key').ask(b'not an image')


@pytest.mark.parametrize('provider', [ClaudeVisionAgent, OpenAIVisionAgent])
def test_http_payload_and_response_parsing_use_stub(provider, monkeypatch, png):
    seen = []
    class Response:
        def raise_for_status(self):
            pass
        def json(self):
            if provider is ClaudeVisionAgent:
                return {'content': [{'type': 'text', 'text': answer()}]}
            return {'choices': [{'message': {'content': answer()}}]}
    def post(url, **kwargs):
        seen.append((url, kwargs))
        return Response()
    monkeypatch.setattr(requests, 'post', post)
    result = provider(api_key='fixture-key', model='fixture-model').ask(png)
    assert json.loads(result)['final_answer'] == 'x=6'
    assert seen[0][1]['json']['model'] == 'fixture-model'
    assert seen[0][1]['timeout'] == (5, 30)
    content = seen[0][1]['json']['messages'][0]['content']
    assert any(part['type'] in ('image', 'image_url') for part in content)


def test_error_formatter_preserves_failure_message():
    result = AgentResult('Claude', '', 0.1, False, 'Missing API configuration')
    assert display_response(result) == 'Unavailable: Missing API configuration'
