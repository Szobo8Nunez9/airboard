"""Software tests never use credentials, the network, or real provider requests."""
import os
os.environ['OPENAI_API_KEY'] = ''
os.environ['ANTHROPIC_API_KEY'] = ''
os.environ['MOCK_VISION_APIS'] = 'false'
os.environ.setdefault('MPLBACKEND', 'Agg')

import pytest
import requests
import agents


@pytest.fixture(autouse=True)
def isolate_provider_configuration(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', '')
    monkeypatch.setenv('ANTHROPIC_API_KEY', '')
    monkeypatch.setenv('MOCK_VISION_APIS', 'false')
    monkeypatch.setattr(agents, '_runner_instance', None)
    def no_network(*args, **kwargs):
        raise AssertionError('A test tried to make a real HTTP request')
    monkeypatch.setattr(requests, 'post', no_network)
