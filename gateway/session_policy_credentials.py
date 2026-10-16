"""Borrow frozen config credentials from the existing profile source, not a vault."""
import hashlib
import hmac
import json
import logging
from dataclasses import asdict
from pathlib import Path

from agent.credential_persistence import fingerprint_secret_value
from hermes_state_runtime import RuntimeStoreError

logger = logging.getLogger(__name__)
PREFIX = 'profile-config-v1:'
_ABSENT = object()


def _defined(root, path):
    """The value *root* defines at *path*, else ``_ABSENT`` when the key is simply not there."""
    value = root
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return _ABSENT
        value = value[key]
    return value


def _route_paths(policy):
    """Secret paths the session's own route authenticates with: its ``model:`` block, a direct alias's
    key (``api_key_from``) and the ``providers:`` / ``custom_providers:`` entry its provider names."""
    config = json.loads(policy.config_json)
    model = config.get('model') if isinstance(config.get('model'), dict) else {}
    paths = {('model', key) for key in model}
    if isinstance(model.get('api_key_from'), list):
        paths.add(tuple(model['api_key_from']))
    name = str(model.get('provider') or '').strip().lower()
    name = name.removeprefix('custom:')

    def named(value):
        return str(value or '').strip().lower().replace(' ', '-') == name.replace(' ', '-')
    providers = config.get('providers')
    for key, entry in (providers.items() if isinstance(providers, dict) and name else ()):
        if named(key) or isinstance(entry, dict) and named(entry.get('name')):
            paths |= {('providers', key, field) for field in (entry if isinstance(entry, dict) else ())}
    custom = config.get('custom_providers')
    for index, entry in (enumerate(custom) if isinstance(custom, list) and name else ()):
        if isinstance(entry, dict) and named(entry.get('name')):
            paths |= {('custom_providers', index, field) for field in entry}
    return paths


def policy_identity(policy):
    data = asdict(policy)
    data.pop('credential_ref')
    data.pop('config_secret_ref')
    # Fields added after release are hashed only when set, so an older session's reference still matches.
    if data.get('editor_mcp_json') is None:
        data.pop('editor_mcp_json', None)
    if not data.get('coding_posture'):
        data.pop('coding_posture', None)
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def _fingerprint(value):
    return fingerprint_secret_value(value if isinstance(value, str) else json.dumps(value, sort_keys=True))


def config_reference(authority, session_id, policy, secrets):
    home = Path(authority.db.db_path).resolve().parent
    return PREFIX + json.dumps({'home': str(home), 'profile': authority.profile_id,
        'session': session_id, 'policy': policy_identity(policy),
        'entries': [[list(path), _fingerprint(value)] for path, value in secrets.items()]},
        sort_keys=True)


def recover_config_secrets(authority, policy, skip=None):
    """The frozen config secrets; entries under a *skip* section are neither re-read nor checked. Every
    caller of a following session skips its followed sections: the live profile supplies them, so a key
    rotation there is not a revoked launch credential."""
    if skip is None:
        from gateway.session_policy import FOLLOW_SECTIONS
        skip = FOLLOW_SECTIONS if policy.follows_profile else ()
    ref = policy.config_secret_ref
    if not ref.startswith(PREFIX):
        values = getattr(authority, '_local_config_secrets', {}).get(ref)
        if values is None:
            raise RuntimeStoreError('launch_credentials_unavailable')
        return {path: value for path, value in values.items() if path[0] not in skip}
    try:
        source = json.loads(ref[len(PREFIX):])
        home = Path(authority.db.db_path).resolve().parent
        if (source['home'] != str(home) or source['profile'] != authority.profile_id
                or source['policy'] != policy_identity(policy)):
            raise ValueError('scope mismatch')
        cached = getattr(authority, '_local_config_secrets', {}).get(ref)
        if cached is not None:
            return {path: value for path, value in cached.items() if path[0] not in skip}
        from gateway.run import _load_gateway_config
        entries = [entry for entry in source['entries'] if entry[0][0] not in skip]
        config = _load_gateway_config(home / 'config.yaml') if any(path[0] is not None for path, _ in entries) else {}
        route = _route_paths(policy)
        values = {}
        for path, fingerprint in entries:
            value, lookup = config, path
            if path[0] is None:
                from tools.terminal_scope import build_profile_terminal_scope
                value, lookup = build_profile_terminal_scope(home), path[1:]
            value = _defined(value, lookup)
            actual = None if value is _ABSENT else _fingerprint(value)
            if actual is None or not hmac.compare_digest(actual, fingerprint):
                if tuple(path) in route:
                    raise RuntimeStoreError('launch_credentials_unavailable',
                        field='.'.join(str(part) for part in path if part is not None),
                        detail=('removed from' if value is _ABSENT else 'changed in')
                        + ' config.yaml since this session started')
                # Any other secret (an MCP env token, an unused provider's key) is simply no longer
                # supplied: the frozen tree holds None there, never the profile's new value.
                logger.warning('Session %s: frozen config secret %s no longer matches the profile '
                               'config; skipping it', source.get('session'), path)
                continue
            values[tuple(path)] = value
        return values
    except RuntimeStoreError:
        raise
    except (KeyError, TypeError, ValueError, AttributeError, OSError) as exc:
        raise RuntimeStoreError('launch_credentials_unavailable') from exc
