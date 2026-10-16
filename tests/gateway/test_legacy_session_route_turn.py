"""#138912: a session a pre-0.22.1 surface wrote resumes on ITS route, with only that route's credential.

Rows are shaped exactly as v0.21.x persisted them: the TUI's ``_persist_live_session_runtime`` ->
``_runtime_model_config`` (model, provider, base_url, api_mode, reasoning_config, service_tier in
``model_config``), and the CLI's ``AIAgent._session_init_model_config`` (max_iterations, reasoning_config,
max_tokens) whose route lives only in the ``billing_*`` columns ``update_token_counts`` COALESCE-wrote.
The profile now targets another endpoint with its own key. Two loopback providers record what each
received on the adopted session's next turn.
"""
import asyncio
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.session_contract import Principal
from tests.fakes.fake_llm_provider import FakeLLMServer, Text

SID = '20261012_122515_721bf6'


def _adopt_and_resolve(home, legacy_route, source='tui', billing=None):
    from gateway.config import GatewayConfig
    from gateway.run_turn_prepare import _resolve_policy_agent_runtime
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_local_migration import bind_native_transport, resolve_local_target
    from gateway.session_policy import restore_policy
    from hermes_state_local import local_receipt

    store = SessionStore(home / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None, _adapter_for_source=lambda source: None)
    authority = asyncio.run(initialize_session_authority(runner, profile_id=str(home), instance_id='fixture'))
    actor = Principal('uid:1000', str(home), frozenset({'session:read', 'session:create'}), 'native')
    bind_native_transport(authority, actor, {'provider': 'local', 'profile_id': str(home), 'instance_id': 'fixture'})
    reasoning = {'enabled': True, 'effort': 'xhigh'}
    authority.db.create_session(SID, source=source, model='legacy-model', model_config={
        'model': 'legacy-model', **legacy_route, 'api_mode': 'chat_completions',
        'reasoning_config': reasoning, 'service_tier': 'priority'} if source == 'tui' else {
        'max_iterations': 90, 'reasoning_config': reasoning, 'max_tokens': None})
    if billing:
        authority.db.update_token_counts(SID, input_tokens=1, model='legacy-model', **billing)
    try:
        resolve_local_target(authority, actor, SID)
        policy = restore_policy(local_receipt(authority.db, SID)['policy'])
        return policy, _resolve_policy_agent_runtime(runner, policy)
    finally:
        store._db.close()


def _write_profile(home, profile_url, extra=''):
    home.mkdir(parents=True, exist_ok=True)
    (home / 'config.yaml').write_text(
        f'model:\n  default: profile-model\n  provider: custom\n  base_url: {profile_url}\n'
        '  api_key: sk-profile-only\n  context_length: 128000\n'
        'agent:\n  api_max_retries: 1\n  reasoning_effort: low\nplatform_toolsets:\n  tui: []\n' + extra)


@pytest.mark.parametrize('shape', ['tui-custom', 'tui-named', 'tui-provider-only', 'cli-billing'])
def test_adopted_legacy_session_turn_goes_to_its_route_with_its_own_key(monkeypatch, shape):
    from run_agent import AIAgent
    with FakeLLMServer([Text('profile')]) as profile, FakeLLMServer([Text('legacy ok')]) as legacy:
        home = Path(os.environ['HERMES_HOME'])  # the per-test home conftest isolates
        monkeypatch.setattr('gateway.run._hermes_home', home)
        _write_profile(home, profile.base_url, extra=(
            f'custom_providers:\n  - name: legacy-box\n    base_url: {legacy.base_url}\n    api_key: sk-legacy-own\n'))
        if shape == 'tui-provider-only':
            # A registry provider's route is its name: the endpoint and key are that provider's own.
            monkeypatch.setenv('OPENROUTER_BASE_URL', legacy.base_url)
            monkeypatch.setenv('OPENROUTER_API_KEY', 'sk-legacy-own')
            policy, (model, runtime) = _adopt_and_resolve(home, {'provider': 'openrouter'})
        elif shape == 'cli-billing':
            policy, (model, runtime) = _adopt_and_resolve(home, {}, source='cli', billing={
                'billing_provider': 'custom', 'billing_base_url': legacy.base_url})
        else:
            policy, (model, runtime) = _adopt_and_resolve(home, {
                'provider': 'custom' if shape == 'tui-custom' else 'custom:legacy-box', 'base_url': legacy.base_url})
        agent = AIAgent(model=model, **{k: runtime[k] for k in ('api_key', 'base_url', 'provider', 'api_mode')},
                        reasoning_config=policy.reasoning_config, quiet_mode=True, skip_context_files=True,
                        skip_memory=True, enabled_toolsets=['memory'])
        agent.run_conversation('hello again')

        assert profile.requests == []  # the profile's endpoint is never contacted ...
        sent = legacy.main_requests()
        assert [r['model'] for r in sent] == ['legacy-model']
        assert policy.reasoning_config == {'enabled': True, 'effort': 'xhigh'}
        if shape != 'tui-provider-only':  # OpenRouter sends effort only for models it knows reason
            assert sent[0]['reasoning_effort'] == 'xhigh'
        # ... and the profile's key never reaches the legacy endpoint: it gets the credential its own
        # provider resolves (the entry serving that URL, also for a bare ``custom`` route).
        assert {r['auth'] for r in legacy.requests} == {'Bearer sk-legacy-own'}


def test_adopted_cli_custom_session_keeps_custom_route_when_profile_moved_to_another_provider(monkeypatch):
    """A v0.21 CLI row's route is ``billing_provider='custom'`` + ``billing_base_url``. With the profile now
    on Anthropic, adoption must not pair ``anthropic`` (its wire mode and key) with the legacy endpoint."""
    from run_agent import AIAgent
    with FakeLLMServer([Text('legacy ok')]) as legacy:
        home = Path(os.environ['HERMES_HOME'])
        monkeypatch.setattr('gateway.run._hermes_home', home)
        monkeypatch.setenv('ANTHROPIC_API_KEY', 'sk-ant-profile-only')
        home.mkdir(parents=True, exist_ok=True)
        (home / 'config.yaml').write_text(
            'model:\n  default: claude-profile\n  provider: anthropic\n'
            'agent:\n  api_max_retries: 1\nplatform_toolsets:\n  tui: []\n'
            f'custom_providers:\n  - name: legacy-box\n    base_url: {legacy.base_url}\n    api_key: sk-legacy-own\n')
        policy, (model, runtime) = _adopt_and_resolve(home, {}, source='cli', billing={
            'billing_provider': 'custom', 'billing_base_url': legacy.base_url})
        agent = AIAgent(model=model, **{k: runtime[k] for k in ('api_key', 'base_url', 'provider', 'api_mode')},
                        reasoning_config=policy.reasoning_config, quiet_mode=True, skip_context_files=True,
                        skip_memory=True, enabled_toolsets=['memory'])
        agent.run_conversation('hello again')

        assert [r['model'] for r in legacy.main_requests()] == ['legacy-model']
        assert legacy.requests and all('sk-ant-profile-only' not in str(r['headers']) for r in legacy.requests)
        assert {r['auth'] for r in legacy.requests} == {'Bearer sk-legacy-own'}


def test_adopted_legacy_session_whose_provider_lost_credentials_is_refused_by_name(monkeypatch):
    with FakeLLMServer([Text('profile')]) as profile:
        home = Path(os.environ['HERMES_HOME'])
        monkeypatch.setattr('gateway.run._hermes_home', home)
        _write_profile(home, profile.base_url)  # the ``legacy-box`` entry is gone from config.yaml
        with pytest.raises(Exception) as caught:
            _adopt_and_resolve(home, {'provider': 'custom:legacy-box', 'base_url': 'http://127.0.0.1:9/v1'})
        assert 'custom:legacy-box' in str(caught.value)
        assert profile.requests == []
