"""A model pick on a session with a running turn lands for the next turn (RT-12).

The pick is authorized and validated when it arrives, kept durably in ``state_meta`` (latest pick
wins, survives an owner restart), and committed through ``session.mutate``'s own writer by the
drain at the next claim boundary, when nothing is executing. The running turn keeps its route.
"""
import json
import logging

from hermes_state_runtime import RuntimeStoreError

logger = logging.getLogger(__name__)
_PREFIX = 'gateway.deferred_pick.v1.'


def defer_pick(authority, actor, ref, params):
    """Record *params* (an authorized ``model`` mutation refused only because a turn is running)."""
    from hermes_state_mutations import validate_action
    from hermes_state_runtime import _json
    validate_action('model', params['payload'])
    value = _json({'principal_id': actor.subject, 'request_id': params['request_id'], 'payload': params['payload']})
    authority.db._execute_write(lambda conn: conn.execute(
        'INSERT INTO state_meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value',
        (_PREFIX + ref.session_id, value)))
    row = authority.db.get_session(ref.session_id)
    return {'session_id': ref.session_id, 'operation': 'model', 'status': 'deferred',
            'model': params['payload']['model'], 'revision': row['runtime_revision'],
            'execution_generation': row['runtime_generation']}


async def apply_deferred_pick(authority, ref, live):
    """Commit a deferred pick. The drain calls this under the claim gate and mutation lock, before
    claiming, so no turn is running. A pick that no longer resolves is dropped and announced."""
    import asyncio
    import uuid
    from gateway.session_mutations import write_mutation
    key = _PREFIX + ref.session_id
    with authority.db._read_ctx() as conn:
        stored = conn.execute('SELECT value FROM state_meta WHERE key=?', (key,)).fetchone()
    if stored is None:
        return
    pick = json.loads(stored[0])
    row = await asyncio.to_thread(authority.db.get_session, ref.session_id)
    params = {'session_id': ref.session_id, 'request_id': 'deferred-pick:' + uuid.uuid4().hex,
              'expected_revision': row['runtime_revision'], 'expected_generation': row['runtime_generation'],
              'operation': 'model', 'payload': pick['payload']}
    from hermes_state_mutation_guards import MUTATION_GUARD_REFUSALS
    try:
        result = await write_mutation(authority, ref, params, None, principal_id=pick['principal_id'])
    except MUTATION_GUARD_REFUSALS:
        return  # a worker still holds the transcript: the pick waits for the next boundary
    except RuntimeStoreError as exc:
        if exc.reason == 'session_busy':
            return
        result = {'session_id': ref.session_id, 'operation': 'model', 'status': 'refused', 'reason': exc.reason}
    authority.db._execute_write(lambda conn: conn.execute(
        'DELETE FROM state_meta WHERE key=? AND value=?', (key, stored[0])))
    if result.get('status') in {'refused', 'confirmation_required'}:
        logger.warning('Deferred model pick on %s not applied: %s', ref.session_id,
                       result.get('reason') or result['status'])
        live.event_stream.publish(ref.session_id, {**result, 'status': 'refused',
                                                   'reason': result.get('reason') or result['status']},
                                  event_type='session.updated')
