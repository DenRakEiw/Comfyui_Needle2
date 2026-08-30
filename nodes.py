"""ComfyUI nodes backed by Cactus Needle 2 (45M params, CPU only, ~28 MB RAM).

Needle 2 is text-in / JSON-out. It does tool calling and structured extraction
and nothing else - there is no free-text fallback, so these nodes turn prompts
and captions into typed graph parameters. They never touch the GPU.
"""

from __future__ import annotations

import json

from . import needle_backend as backend

try:
    import comfy.samplers as _samplers
    SAMPLERS = list(_samplers.KSampler.SAMPLERS)
    SCHEDULERS = list(_samplers.KSampler.SCHEDULERS)
except Exception:  # noqa: BLE001 - keep the pack importable outside ComfyUI
    SAMPLERS = ["euler", "dpmpp_2m", "dpmpp_sde", "ddim"]
    SCHEDULERS = ["normal", "karras", "exponential", "simple"]

CATEGORY = "needle"


class AnyType(str):
    """A type that compares equal to every other ComfyUI type.

    ComfyUI's `validate_node_input` short-circuits on `not received != input`,
    which is why overriding `__ne__` is enough to make a slot accept anything.
    """

    def __ne__(self, other):  # noqa: D105
        return False


ANY = AnyType("*")


# One source of truth: these back both the widget defaults and the fallback
# used when a caller passes an empty string.
DEFAULT_SAMPLER_CHOICES = "euler, euler_ancestral, dpmpp_2m, dpmpp_2m_sde, dpmpp_3m_sde, ddim"
DEFAULT_SCHEDULER_CHOICES = "normal, karras, exponential, simple"


def _choices(raw: str, fallback: str, legal: list) -> list:
    """Parse a comma-separated choice list, keeping only names ComfyUI knows."""
    names = [n.strip() for n in (raw or "").split(",") if n.strip()]
    if not names:
        names = [n.strip() for n in fallback.split(",") if n.strip()]
    kept = [n for n in names if n in legal]
    return kept or legal[:6]


DEFAULT_SCHEMA = json.dumps({
    "subject": "string",
    "style": "string?",
    "width": "integer?",
    "height": "integer?",
}, indent=2)


def _fmt_ungrounded(envelope: dict) -> str:
    """The engine flags fields it could not ground in the input text.

    Undocumented but reliably present on successful calls, as
    {"ungrounded": ["render.steps"], "negation": false}.
    """
    validation = envelope.get("validation") or {}
    ungrounded = validation.get("ungrounded") or []
    parts = []
    if ungrounded:
        parts.append("ungrounded (model guessed): " + ", ".join(ungrounded))
    if validation.get("negation"):
        parts.append("negation detected in the input")
    return " | ".join(parts)


def _fmt_stats(envelope: dict) -> str:
    bits = []
    for key, label in (("prefill_tps", "prefill"), ("decode_tps", "decode")):
        value = envelope.get(key)
        if value:
            bits.append(f"{label} {value:.0f} tok/s")
    ram = envelope.get("peak_ram_mb")
    if ram:
        bits.append(f"peak {ram:.1f} MB")
    return ", ".join(bits)


# --------------------------------------------------------------------------


class NeedleExtract:
    """Free text -> JSON, against a schema you define in the widget."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "",
                                    "tooltip": "Text to extract from. Keep it short - "
                                               "the engine uses a 256-token window."}),
                "schema": ("STRING", {"multiline": True, "default": DEFAULT_SCHEMA,
                                      "tooltip": "Shorthand {\"field\": \"string\"}, a bare "
                                                 "JSON Schema, or a full needle tool schema. "
                                                 "Suffix a type with ? to make it optional."}),
            },
            "optional": {
                "system": ("STRING", {"multiline": True, "default": "",
                                      "tooltip": "Environment facts given to the model."}),
                "max_new_tokens": ("INT", {"default": 256, "min": 16, "max": 1024}),
                "drop_ungrounded": ("BOOLEAN", {"default": True,
                                                "tooltip": "Discard fields the engine flags as "
                                                           "not grounded in the input text."}),
                "weights": ("STRING", {"default": "",
                                       "tooltip": "Optional path to a finetuned .cact file. "
                                                  "The engine cannot unload weights, so once "
                                                  "one is loaded every base-model node in the "
                                                  "process will fail."}),
            },
        }

    RETURN_TYPES = ("NEEDLE_RESULT", "STRING", "FLOAT", "BOOLEAN", "STRING")
    RETURN_NAMES = ("result", "json", "confidence", "success", "info")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = __doc__

    def run(self, text, schema, system="", max_new_tokens=256, drop_ungrounded=True, weights=""):
        tool_schema = backend.normalize_schema(schema)
        arguments, envelope = backend.extract_arguments(
            tool_schema, text, system, weights or None, max_new_tokens)
        dropped = []
        if drop_ungrounded:
            arguments, dropped = backend.strip_ungrounded(arguments, envelope)
        confidence = envelope.get("confidence")
        success = bool(arguments) and envelope.get("success", True) is not False
        info = "\n".join(filter(None, [
            f"fields: {', '.join(arguments) if arguments else '(none matched)'}",
            f"reasoning: {envelope['reasoning']}" if envelope.get("reasoning") else "",
            f"dropped as ungrounded: {', '.join(dropped)}" if dropped else "",
            _fmt_ungrounded(envelope) if not drop_ungrounded else "",
            f"error: {envelope['error']}" if envelope.get("error") else "",
            _fmt_stats(envelope),
        ]))
        return (
            {"arguments": arguments, "envelope": envelope},
            json.dumps(arguments, indent=2, ensure_ascii=False),
            float(confidence) if confidence is not None else -1.0,
            success,
            info,
        )


class NeedleGetField:
    """Pull one field out of a NEEDLE_RESULT as a wildcard output."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "result": ("NEEDLE_RESULT",),
                "field": ("STRING", {"default": "subject",
                                     "tooltip": "Field name. Use dots for nesting: a.b"}),
                "cast": (["auto", "string", "int", "float", "boolean"], {"default": "auto"}),
                "default": ("STRING", {"default": "",
                                       "tooltip": "Used when the field is missing. Needle "
                                                  "omits fields it found no evidence for."}),
            },
        }

    RETURN_TYPES = (ANY, "STRING", "BOOLEAN")
    RETURN_NAMES = ("value", "string", "found")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = __doc__

    def run(self, result, field, cast="auto", default=""):
        arguments = (result or {}).get("arguments") or {}
        node, found = arguments, True
        for part in field.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                node, found = default, False
                break
        return (_cast(node, cast, default), _as_text(node), found)


def _as_text(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return "" if value is None else str(value)


def _cast(value, cast, default):
    if cast == "auto":
        return value
    text = _as_text(value)
    try:
        if cast == "int":
            return int(float(text))
        if cast == "float":
            return float(text)
        if cast == "boolean":
            return text.strip().lower() in ("1", "true", "yes", "on")
        return text
    except (TypeError, ValueError):
        return _cast(default, cast, "") if default else (0 if cast == "int" else
                                                         0.0 if cast == "float" else
                                                         False if cast == "boolean" else "")


# --------------------------------------------------------------------------


DEFAULT_ROUTES = """txt2img: generate a brand new image from a description
img2img: change or restyle an image the user already has
inpaint: replace or remove part of an existing image
upscale: enlarge an image or add detail without changing the content"""


class NeedleRouter:
    """Classify a request into one branch and gate it on the confidence score."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
                "routes": ("STRING", {"multiline": True, "default": DEFAULT_ROUTES,
                                      "tooltip": "One route per line, 'name: description'. "
                                                 "Tools are pinned in the 256-token window, "
                                                 "so keep the list short."}),
                "threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                        "tooltip": "Below this confidence the fallback wins. "
                                                   "0 disables the gate. Finetuned weights "
                                                   "report no confidence at all."}),
                "fallback_index": ("INT", {"default": 0, "min": 0, "max": 63}),
            },
            "optional": {
                "system": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("INT", "STRING", "FLOAT", "BOOLEAN", "STRING")
    RETURN_NAMES = ("index", "route", "confidence", "gated", "info")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = __doc__

    def run(self, text, routes, threshold=0.0, fallback_index=0, system=""):
        names, described = [], []
        for line in routes.splitlines():
            line = line.strip()
            if not line:
                continue
            name, _, desc = line.partition(":")
            name = name.strip()
            if not name:
                continue
            names.append(name)
            described.append(desc.strip() or name)
        if not names:
            raise ValueError("no routes defined")

        # One tool per route, not one enum-valued field. Needle is trained to
        # *select a tool*, and measured on six routing prompts the enum form
        # scored 0/6 while this form scored 4/6.
        tools = [{"name": name, "description": desc, "parameters": {"type": "object",
                                                                    "properties": {}}}
                 for name, desc in zip(names, described)]
        envelope = backend.complete_multi(tools, text, system)
        calls = envelope.get("function_calls") or []
        chosen = calls[0].get("name") if calls else None
        confidence = envelope.get("confidence")

        fallback = min(fallback_index, len(names) - 1)
        gated = False
        if chosen not in names:
            index, gated = fallback, True
        elif threshold > 0.0 and confidence is not None and confidence < threshold:
            index, gated = fallback, True
        else:
            index = names.index(chosen)

        info = "\n".join(filter(None, [
            f"picked: {chosen or '(nothing)'} -> index {index}"
            + (f" (gated to {names[index]})" if gated else ""),
            f"reasoning: {envelope['reasoning']}" if envelope.get("reasoning") else "",
            _fmt_stats(envelope),
        ]))
        return (index, names[index],
                float(confidence) if confidence is not None else -1.0, gated, info)


# --------------------------------------------------------------------------


def _snap(value, options, fallback):
    """Map whatever the model said onto a name ComfyUI actually accepts."""
    if not value:
        return fallback
    text = str(value).strip()
    if text in options:
        return text

    def norm(item):
        # "DPM++ 2M" and "dpmpp_2m" have to land on the same key, so the ++
        # convention is folded before punctuation is dropped.
        item = item.lower().replace("++", "pp")
        for char in "_- +.":
            item = item.replace(char, "")
        return item

    target = norm(text)
    for option in options:
        if norm(option) == target:
            return option
    # Longest overlap wins: a bare substring scan lets "dpm_2" swallow "dpmpp_2m".
    best = None
    for option in options:
        candidate = norm(option)
        if target and (target in candidate or candidate in target):
            if best is None or len(candidate) > len(norm(best)):
                best = option
    return best or fallback


class NeedlePromptParams:
    """Split a free-text prompt into sampler settings you can wire into KSampler."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True,
                                      "default": "a photo of a red fox in snow, "
                                                 "1024x1536, 30 steps, cfg 4.5, "
                                                 "dpmpp_2m karras, seed 1234"}),
            },
            "optional": {
                "sampler_choices": ("STRING", {
                    "default": DEFAULT_SAMPLER_CHOICES,
                    "tooltip": "Offered to the model as an enum. Keep it short - every tool "
                               "token competes for the 256-token window. The output is never "
                               "a sampler outside this list."}),
                "scheduler_choices": ("STRING", {"default": DEFAULT_SCHEDULER_CHOICES}),
                "default_width": ("INT", {"default": 1024, "min": 16, "max": 16384, "step": 8}),
                "default_height": ("INT", {"default": 1024, "min": 16, "max": 16384, "step": 8}),
                "default_steps": ("INT", {"default": 20, "min": 1, "max": 1000}),
                "default_cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "default_seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFF}),
                "default_sampler": (SAMPLERS, {"tooltip": "Used when the prompt names no sampler."}),
                "default_scheduler": (SCHEDULERS, {"tooltip": "Used when the prompt names no scheduler."}),
                "drop_ungrounded": ("BOOLEAN", {"default": True,
                                                "tooltip": "Discard values the engine flags as "
                                                           "not grounded in the prompt, and use "
                                                           "the defaults instead. The model does "
                                                           "invent numbers on sparse prompts."}),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "INT", "INT", "FLOAT",
                    SAMPLERS, SCHEDULERS, "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("positive", "width", "height", "steps", "cfg",
                    "sampler_name", "scheduler", "seed", "confidence", "info")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = __doc__

    def run(self, prompt, sampler_choices="", scheduler_choices="",
            default_width=1024, default_height=1024, default_steps=20,
            default_cfg=7.0, default_seed=0, default_sampler=None, default_scheduler=None,
            drop_ungrounded=True):
        samplers = _choices(sampler_choices, DEFAULT_SAMPLER_CHOICES, SAMPLERS)
        schedulers = _choices(scheduler_choices, DEFAULT_SCHEDULER_CHOICES, SCHEDULERS)

        schema = {
            "name": "render",
            "description": "Render an image with the settings the user asked for.",
            "parameters": {
                "type": "object",
                # Field order and bounds are both load-bearing, not cosmetic.
                # Both feed the byte-level decoding grammar:
                #   - without "maximum" on cfg, "cfg 4.5" is read as 45
                #   - with the enums declared last, scheduler comes back None
                #     (measured 0/4); declared first, sampler and scheduler are
                #     both correct 4/4.
                "properties": {
                    "sampler": {"type": "string", "enum": samplers},
                    "scheduler": {"type": "string", "enum": schedulers},
                    "subject": {"type": "string",
                                "description": "what to draw, without any technical settings"},
                    "width": {"type": "integer", "description": "image width in pixels",
                              "minimum": 64, "maximum": 8192},
                    "height": {"type": "integer", "description": "image height in pixels",
                               "minimum": 64, "maximum": 8192},
                    "steps": {"type": "integer", "description": "number of sampling steps",
                              "minimum": 1, "maximum": 200},
                    "cfg": {"type": "number", "minimum": 0, "maximum": 30,
                            "description": "guidance scale, typically 1 to 15"},
                    "seed": {"type": "integer", "description": "random seed", "minimum": 0},
                },
                "required": ["subject"],
            },
        }
        arguments, envelope = backend.extract_arguments(schema, prompt)
        dropped = []
        if drop_ungrounded:
            arguments, dropped = backend.strip_ungrounded(arguments, envelope)

        def as_int(key, fallback, lo=1, hi=0xFFFFFFFFFFFFFFF):
            try:
                return max(lo, min(hi, int(arguments[key])))
            except (KeyError, TypeError, ValueError):
                return fallback

        confidence = envelope.get("confidence")
        positive = str(arguments.get("subject") or prompt).strip()
        try:
            cfg = float(arguments["cfg"])
        except (KeyError, TypeError, ValueError):
            cfg = default_cfg

        info = "\n".join(filter(None, [
            f"extracted: {', '.join(arguments) if arguments else '(nothing - defaults used)'}",
            f"dropped as ungrounded, default used: {', '.join(dropped)}" if dropped else "",
            _fmt_ungrounded(envelope) if not drop_ungrounded else "",
            _fmt_stats(envelope),
        ]))
        return (
            positive,
            as_int("width", default_width, 16, 16384),
            as_int("height", default_height, 16, 16384),
            as_int("steps", default_steps, 1, 1000),
            max(0.0, min(100.0, cfg)),
            # Two stages on purpose. The model does not reliably honour the enum -
            # it has been observed returning "euler_cfg_pp" for an input of
            # "dpmpp_2m" - and that happens to be a real ComfyUI sampler, so a
            # single snap against the full list would pass the wrong answer
            # through silently. Constrain to what the user offered first, then
            # map that onto a name ComfyUI actually accepts.
            _snap(_snap(arguments.get("sampler"), samplers, default_sampler or samplers[0]),
                  SAMPLERS, default_sampler or SAMPLERS[0]),
            _snap(_snap(arguments.get("scheduler"), schedulers, default_scheduler or schedulers[0]),
                  SCHEDULERS, default_scheduler or SCHEDULERS[0]),
            as_int("seed", default_seed, 0),
            float(confidence) if confidence is not None else -1.0,
            info,
        )


# --------------------------------------------------------------------------


class NeedleStatus:
    """Report whether the Needle engine can actually load on this machine."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("STRING", "BOOLEAN")
    RETURN_NAMES = ("status", "ok")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True
    DESCRIPTION = __doc__

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")  # always re-probe

    def run(self):
        status = backend.engine_status()
        return (status, status.startswith("ok"))


NODE_CLASS_MAPPINGS = {
    "NeedleExtract": NeedleExtract,
    "NeedleGetField": NeedleGetField,
    "NeedleRouter": NeedleRouter,
    "NeedlePromptParams": NeedlePromptParams,
    "NeedleStatus": NeedleStatus,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NeedleExtract": "Needle Extract (text to JSON)",
    "NeedleGetField": "Needle Get Field (any)",
    "NeedleRouter": "Needle Router",
    "NeedlePromptParams": "Needle Prompt to Sampler Params",
    "NeedleStatus": "Needle Engine Status",
}
