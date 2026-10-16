"""A model pick on a session with a running turn lands for the next turn (RT-12, #138771)."""
import asyncio
import threading
from types import SimpleNamespace

import pytest

from tests.waiting import async_wait_until


def _owner(tmp_path, monkeypatch, handle):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import SessionAuthority
    from gateway.session_controls import AuthorityConnection
    from gateway.session_local import create_local_session
    from hermes_cli import model_switch
    import hermes_state_runtime as rt
    monkeypatch.setattr(run, '_load_gateway_config', lambda *a: {})
    monkeypatch.setattr(model_switch, 'switch_model', lambda **k: model_switch.ModelSwitchResult(
        success=True, new_model=k['raw_input'], target_provider='custom', base_url='http://127.0.0.1:9/v1'))
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    db = store._db
    runner = SimpleNamespace(session_store=store, _session_db=db, adapters={}, _draining=False,
                             _evict_cached_agent=lambda route: None, _handle_message=handle,
                             _resolve_session_agent_runtime=lambda **k: ('frozen', {}))
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = SessionAuthority(runner, profile_id='owned', instance_id='owner', db=db,
                                 epoch=rt.begin_runtime_epoch(db, instance_id='owner'))
    owner = AuthorityConnection(authority, object(), {'user_id': 'human'})
    ref = create_local_session(authority, owner.actor, dict(request_id='m', source='gui', cwd=str(tmp_path),
                                                            model='frozen', toolsets=[]))
    return store, db, authority, owner, ref


def _running_model(authority, ref):
    from gateway.config import Platform
    from gateway.session_local_recovery import local_adapter_map
    live = authority.sessions[ref.session_id]
    return local_adapter_map(authority)[Platform.LOCAL].policies[live.source.chat_id].model


@pytest.mark.asyncio
async def test_model_pick_during_a_running_turn_applies_to_the_next_turn(tmp_path, monkeypatch):
    from gateway.session_contract import Submission
    import hermes_state_runtime as rt
    started, release = threading.Event(), asyncio.Event()
    seen = []
    state = {}

    async def handle(event):
        seen.append((event.text, _running_model(state['authority'], state['ref'])))
        if event.text == 'RUNNING':
            started.set()
            await release.wait()
        return 'ok'
    store, db, authority, owner, ref = _owner(tmp_path, monkeypatch, handle)
    state.update(authority=authority, ref=ref)
    try:
        first = await authority.submit(owner.actor, Submission('running', ref, {'text': 'RUNNING'}, 'queue'))
        await async_wait_until(started.is_set, timeout=10, what='the first turn is running')
        snap = db.get_session(ref.session_id)
        reply = await owner.dispatch({'id': 1, 'method': 'session.mutate', 'params': {
            'session_id': ref.session_id, 'request_id': 'pick', 'expected_revision': snap['runtime_revision'],
            'expected_generation': snap['runtime_generation'], 'operation': 'model', 'payload': {'model': 'picked'}}})
        assert 'result' in reply, reply
        assert reply['result']['status'] == 'deferred'
        # The running turn keeps its route.
        assert _running_model(authority, ref) == 'frozen'
        release.set()
        await async_wait_until(
            lambda: rt.get_session_admission(db, admission_id=first.admission_id)['status'] == 'terminal',
            timeout=10, what='the running turn settled')
        await async_wait_until(lambda: _running_model(authority, ref) == 'picked',
                               timeout=10, what='the pick published at the turn boundary')
        second = await authority.submit(owner.actor, Submission('next', ref, {'text': 'NEXT'}, 'queue'))
        await async_wait_until(
            lambda: rt.get_session_admission(db, admission_id=second.admission_id)['status'] == 'terminal',
            timeout=10, what='the next turn ran')
        assert seen == [('RUNNING', 'frozen'), ('NEXT', 'picked')]
        assert db.get_session(ref.session_id)['model'] == 'picked'
    finally:
        release.set()
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_revision_conflict_refusal_carries_the_fresh_fences(tmp_path, monkeypatch):
    async def handle(event):
        return 'ok'
    store, db, _authority, owner, ref = _owner(tmp_path, monkeypatch, handle)
    try:
        snap = db.get_session(ref.session_id)
        reply = await owner.dispatch({'id': 1, 'method': 'session.mutate', 'params': {
            'session_id': ref.session_id, 'request_id': 'stale', 'expected_revision': snap['runtime_revision'] + 7,
            'expected_generation': snap['runtime_generation'], 'operation': 'model', 'payload': {'model': 'picked'}}})
        data = reply['error']['data']
        assert data['reason'] == 'revision_conflict'
        assert (data['revision'], data['execution_generation']) == (snap['runtime_revision'], snap['runtime_generation'])
        retry = await owner.dispatch({'id': 2, 'method': 'session.mutate', 'params': {
            'session_id': ref.session_id, 'request_id': 'retry', 'expected_revision': data['revision'],
            'expected_generation': data['execution_generation'], 'operation': 'model', 'payload': {'model': 'picked'}}})
        assert retry['result']['model'] == 'picked'
    finally:
        await owner.close()
        store.close_all_db_handles()
