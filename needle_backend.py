"""Thin wrapper around the cactus-needle engine for ComfyUI.

The engine is a process-global singleton: `needle.Needle` keeps `_active` and
`_active_weights` at module level, and every completion re-binds the shared
C engine via `needle_init`. ComfyUI executes nodes on a worker thread and may
run several Needle nodes in one graph, so every call here goes through one
lock and agents are cached rather than rebuilt per execution.
"""

from __future__ import annotations

import json
import os
import threading

# Telemetry is on by default upstream and fires a network call per run.
# Opt back in with NEEDLE_COMFY_TELEMETRY=1 before ComfyUI starts.
if os.environ.get("NEEDLE_COMFY_TELEMETRY", "0") != "1":
    os.environ.setdefault("NEEDLE_TELEMETRY", "0")
    os.environ.setdefault("DO_NOT_TRACK", "1")

_LOCK = threading.RLock()
_AGENTS: dict[tuple, object] = {}
_IMPORT_ERROR: str | None = None
_needle = None

INSTALL_HINT = (
    "cactus-needle is not installed in ComfyUI's Python. Install it with:\n"
    r"  python_embeded\python.exe -m pip install cactus-needle"
)

# WinError 4551 / 577: Smart App Control or a WDAC policy refused to load the
# unsigned libneedle.dll. Nothing in this package can work around that.
_BLOCKED_HINT = (
    "Windows blocked libneedle.dll (Smart App Control / WDAC).\n"
    "The DLL downloaded fine but is unsigned, so Code Integrity refuses to "
    "load it into python.exe (event 3033/3077).\n"
    "Check with:  Get-MpComputerStatus | Select SmartAppControlState\n"
    "Smart App Control can only be turned off in Windows Security -> "
    "App & browser control. Note that turning it off is IRREVERSIBLE: it "
    "cannot be switched back on without reinstalling Windows."
)


class NeedleUnavailable(RuntimeError):
    """Raised when the engine cannot be reached, with actionable context."""


def _import_needle():
    global _needle, _IMPORT_ERROR
    if _needle is not None:
        return _needle
    if _IMPORT_ERROR is not None:
        raise NeedleUnavailable(_IMPORT_ERROR)
    try:
        import needle as _mod
    except ImportError as exc:
        _IMPORT_ERROR = f"{INSTALL_HINT}\n\n(import failed: {exc})"
        raise NeedleUnavailable(_IMPORT_ERROR) from exc
    if not hasattr(_mod, "Needle"):
        # This node pack lives in a directory that may also be called "needle".
        # ComfyUI itself registers directory nodes under their full path, but
        # anything that puts custom_nodes on sys.path would shadow the engine.
        _IMPORT_ERROR = (
            f"'import needle' resolved to {getattr(_mod, '__file__', '?')}, which is not "
            f"the cactus-needle engine - something is shadowing it on sys.path. "
            f"Rename this node directory (e.g. to comfyui-needle) and restart ComfyUI."
        )
        raise NeedleUnavailable(_IMPORT_ERROR)
    _needle = _mod
    return _mod


def _translate(exc: Exception) -> NeedleUnavailable:
    text = str(exc)
    if isinstance(exc, OSError) and ("4551" in text or "577" in text or "blockiert" in text):
        return NeedleUnavailable(f"{_BLOCKED_HINT}\n\n(original error: {text})")
    return NeedleUnavailable(text)


def engine_status() -> str:
    """Human-readable probe used by the status node and by error messages."""
    try:
        mod = _import_needle()
    except NeedleUnavailable as exc:
        return f"unavailable\n{exc}"
    version = getattr(mod, "__version__", "unknown")
    try:
        with _LOCK:
            mod.Needle(tools=[{
                "name": "ping",
                "description": "probe",
                "parameters": {"type": "object", "properties": {}},
            }])
        return f"ok\ncactus-needle {version}"
    except Exception as exc:  # noqa: BLE001 - surfaced verbatim to the user
        return f"unavailable\ncactus-needle {version}\n{_translate(exc)}"


# --------------------------------------------------------------------------
# schema helpers
# --------------------------------------------------------------------------

_SHORTHAND = {
    "string": "string", "str": "string", "text": "string",
    "int": "integer", "integer": "integer",
    "float": "number", "number": "number", "num": "number",
    "bool": "boolean", "boolean": "boolean",
    "list": "array", "array": "array",
    "dict": "object", "object": "object",
}


def normalize_schema(raw: str, default_name: str = "extract",
                     description: str = "Extract the requested fields from the text.") -> dict:
    """Accept the three shapes a user might reasonably type into a widget.

    1. a full needle tool schema   {"name": ..., "parameters": {...}}
    2. a bare JSON Schema object   {"type": "object", "properties": {...}}
    3. a shorthand field map       {"vendor": "string", "total": "number"}
    """
    text = (raw or "").strip()
    if not text:
        raise ValueError("schema is empty")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"schema is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("schema must be a JSON object")

    if "parameters" in data and isinstance(data["parameters"], dict):
        data.setdefault("name", default_name)
        return data

    if data.get("type") == "object" and isinstance(data.get("properties"), dict):
        return {"name": default_name, "description": description, "parameters": data}

    properties, required = {}, []
    for key, value in data.items():
        if isinstance(value, dict):
            properties[key] = value
            required.append(key)
            continue
        spec = str(value).strip()
        optional = spec.endswith("?")
        spec = spec.rstrip("?").strip()
        json_type = _SHORTHAND.get(spec.lower())
        if json_type is None:
            # treat an unknown string as a description of a string field
            properties[key] = {"type": "string", "description": spec}
        else:
            properties[key] = {"type": json_type}
        if not optional:
            required.append(key)
    parameters = {"type": "object", "properties": properties}
    if required:
        parameters["required"] = required
    return {"name": default_name, "description": description, "parameters": parameters}


def _agent(schema: dict, system: str, weights: str | None):
    key = (json.dumps(schema, sort_keys=True), system or "", weights or "")
    agent = _AGENTS.get(key)
    if agent is not None:
        return agent
    mod = _import_needle()
    try:
        agent = mod.Needle(tools=[schema], system=system or None, weights=weights or None)
    except Exception as exc:  # noqa: BLE001
        raise _translate(exc) from exc
    _AGENTS[key] = agent
    return agent


def complete(schema: dict, text: str, system: str = "", weights: str | None = None,
             max_new_tokens: int = 256) -> dict:
    """Run one turn and return the raw engine envelope.

    Keys: type, success, error, function_calls, reasoning, confidence,
    prefill_tps, decode_tps, peak_ram_mb.
    """
    with _LOCK:
        agent = _agent(schema, system, weights)
        try:
            # Every node execution is an independent turn. Agents are cached, and
            # the engine keeps a 256-token conversation window, so without this
            # the previous run's text bleeds into the next one - measurably: one
            # off-topic prompt makes every later extraction return {}.
            agent.reset()
            return agent.complete(text, max_new_tokens=max_new_tokens)
        except Exception as exc:  # noqa: BLE001
            raise _translate(exc) from exc


def complete_multi(tools: list, text: str, system: str = "", weights: str | None = None,
                   max_new_tokens: int = 256) -> dict:
    """Same as complete(), but offers several tools so the model picks one.

    Needle has a retrieval head that renders the top five tools per turn, so a
    long catalogue is allowed - but tools are pinned in the 256-token window,
    so a short list still decodes faster.
    """
    key = json.dumps(tools, sort_keys=True)
    with _LOCK:
        cache_key = (key, system or "", weights or "")
        agent = _AGENTS.get(cache_key)
        if agent is None:
            mod = _import_needle()
            try:
                agent = mod.Needle(tools=tools, system=system or None,
                                   weights=weights or None)
            except Exception as exc:  # noqa: BLE001
                raise _translate(exc) from exc
            _AGENTS[cache_key] = agent
        try:
            agent.reset()
            return agent.complete(text, max_new_tokens=max_new_tokens)
        except Exception as exc:  # noqa: BLE001
            raise _translate(exc) from exc


def strip_ungrounded(arguments: dict, envelope: dict) -> tuple[dict, list]:
    """Drop fields the engine could not ground in the input text.

    The engine reports these as ["<tool>.<field>", ...] under `validation`.
    They are values the model invented - for "a car, 768x512, 12 steps" it
    returned 600x600 and flagged width/height - so the caller's own default is
    strictly better than keeping them. Returns (kept, dropped_names).
    """
    flagged = (envelope.get("validation") or {}).get("ungrounded") or []
    names = {entry.split(".")[-1] for entry in flagged}
    if not names:
        return arguments, []
    kept = {k: v for k, v in arguments.items() if k not in names}
    return kept, sorted(names & set(arguments))


def extract_arguments(schema: dict, text: str, system: str = "",
                      weights: str | None = None, max_new_tokens: int = 256) -> tuple[dict, dict]:
    """Return (arguments, envelope). arguments is {} when nothing matched."""
    envelope = complete(schema, text, system, weights, max_new_tokens)
    calls = envelope.get("function_calls") or []
    if not calls:
        return {}, envelope
    return (calls[0].get("arguments") or {}), envelope
