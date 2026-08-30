"""Recover generation settings from a ComfyUI `prompt` graph embedded in a PNG.

This is the format that actually matters here: of the 1500 newest images in
this install, 1496 carry `prompt`/`workflow` chunks and 3 files on the whole
machine carry an A1111 `parameters` string.

The graph is not walkable by looking for "KSampler". Real workflows here are
overwhelmingly SamplerCustomAdvanced (1031) + KSamplerSelect (335), with plain
KSampler a distant third (51), plus one-off nodes from custom packs. So the
strategy is: walk the sampler chain when it is recognisable, and fall back to a
graph-wide scan for well-known input names when it is not.

Two things bite anyone writing this by hand:
  - `inputs` values are either literals or links `[node_id, slot]`, and a link
    can point at another link (CLIPTextEncode.text -> PrimitiveString.value)
  - node ids are strings and may be namespaced by subgraph: "29:39"
"""

from __future__ import annotations

import json

# Input names that carry each setting, in priority order. Collected from the
# node classes actually present in this install's outputs.
SCALAR_NAMES = {
    "seed": ("noise_seed", "seed"),
    "steps": ("steps",),
    "cfg": ("cfg", "guidance", "cfg_scale"),
    "sampler_name": ("sampler_name",),
    "scheduler": ("scheduler",),
    "denoise": ("denoise",),
    "width": ("width",),
    "height": ("height",),
    "model": ("unet_name", "ckpt_name", "model_name"),
}

# Nodes that hold the sampling call together.
SAMPLER_NODES = ("SamplerCustomAdvanced", "SamplerCustom", "KSampler",
                 "KSamplerAdvanced", "KSampler (Efficient)")
GUIDER_NODES = ("CFGGuider", "BasicGuider", "Guider_Basic")
TEXT_NODES = ("CLIPTextEncode", "CLIPTextEncodeSDXL", "CLIPTextEncodeFlux",
              "PrimitiveString", "PrimitiveStringMultiline", "String Literal",
              "Text _O", "ttN text", "ShowText|pysssss")


def is_prompt_graph(data) -> bool:
    return (isinstance(data, dict) and bool(data)
            and any(isinstance(v, dict) and "class_type" in v for v in data.values()))


def load_graph(text: str):
    """Return the node dict for a `prompt` chunk, or None."""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if is_prompt_graph(data):
        return data
    # a `workflow` chunk wraps the graph differently and carries no class_type
    # map we can walk; callers should prefer the `prompt` chunk
    return None


def _is_link(value) -> bool:
    return (isinstance(value, list) and len(value) == 2
            and isinstance(value[0], (str, int))
            and isinstance(value[1], int))


def resolve(graph: dict, value, depth: int = 0):
    """Follow [node_id, slot] links until a literal falls out."""
    if depth > 12 or not _is_link(value):
        return value
    node = graph.get(str(value[0]))
    if not isinstance(node, dict):
        return None
    inputs = node.get("inputs") or {}
    # A pass-through node (PrimitiveString, reroute, ShowText) carries exactly
    # one interesting literal; prefer the conventional names, else take the
    # single scalar it has.
    for name in ("value", "text", "string", "STRING", "text_0"):
        if name in inputs:
            return resolve(graph, inputs[name], depth + 1)
    scalars = [v for v in inputs.values() if isinstance(v, (str, int, float, bool))]
    if len(scalars) == 1:
        return scalars[0]
    for candidate in inputs.values():
        if _is_link(candidate):
            return resolve(graph, candidate, depth + 1)
    return None


OUTPUT_NODES = ("SaveImage", "PreviewImage", "SaveAnimatedWEBP", "SaveAnimatedPNG",
                "VHS_VideoCombine", "SaveVideo", "Image Save", "SaveImageWebsocket")


def _is_sampler(node: dict) -> bool:
    if node.get("class_type") in SAMPLER_NODES:
        return True
    inputs = set(node.get("inputs") or {})
    return {"sampler", "sigmas"} <= inputs or {"guider", "sigmas"} <= inputs


def all_samplers(graph: dict) -> list:
    return [node_id for node_id, node in graph.items()
            if isinstance(node, dict) and _is_sampler(node)]


def _find_sampler(graph: dict):
    """Pick the sampler that produced the saved image.

    Multi-pass workflows (hires fix, LTXV upscale) contain several samplers
    that share conditioning nodes, so "the first one in the JSON" is arbitrary
    and mixes settings across passes. Walking back from the output node gives
    the pass that actually made this file.
    """
    ids = all_samplers(graph)
    if not ids:
        return None, None
    if len(ids) > 1:
        for node_id, node in graph.items():
            if node.get("class_type") not in OUTPUT_NODES:
                continue
            # breadth-first back from the output: the first sampler we meet is
            # the last one that ran
            seen, queue = set(), [node]
            while queue:
                current = queue.pop(0)
                for value in (current.get("inputs") or {}).values():
                    if not _is_link(value):
                        continue
                    key = str(value[0])
                    if key in seen:
                        continue
                    seen.add(key)
                    nxt = graph.get(key)
                    if not isinstance(nxt, dict):
                        continue
                    if _is_sampler(nxt):
                        return key, nxt
                    queue.append(nxt)
    return ids[0], graph[ids[0]]


def _chain(graph: dict, node: dict):
    """Nodes reachable from the sampler, nearest first.

    Breadth-first, not depth-first, and the ordering is load-bearing: a
    multi-pass workflow (LTXV upscale, hires fix) reaches a second sampler's
    RandomNoise and sigmas through shared conditioning nodes. Nearest-first
    means the pass we were asked about wins.
    """
    seen, order, queue = set(), [], [node]
    while queue:
        current = queue.pop(0)
        for value in (current.get("inputs") or {}).values():
            if not _is_link(value):
                continue
            key = str(value[0])
            if key in seen:
                continue
            seen.add(key)
            nxt = graph.get(key)
            if isinstance(nxt, dict):
                order.append(nxt)
                queue.append(nxt)
    return order


def _scalar(nodes, names, graph=None):
    """First literal under any of `names`, following links when a graph is given."""
    for node in nodes:
        inputs = node.get("inputs") or {}
        for name in names:
            if name not in inputs:
                continue
            value = inputs[name]
            if graph is not None and _is_link(value):
                value = resolve(graph, value)
            if isinstance(value, (str, int, float)) and not isinstance(value, bool):
                return value
    return None


def _text_of(graph: dict, link):
    value = resolve(graph, link)
    if isinstance(value, str) and value.strip():
        return value.strip()
    # positive/negative usually points at a conditioning chain, so walk it
    if _is_link(link):
        node = graph.get(str(link[0]))
        if isinstance(node, dict):
            for candidate in _chain(graph, node) + [node]:
                text = (candidate.get("inputs") or {}).get("text")
                resolved = resolve(graph, text) if text is not None else None
                if isinstance(resolved, str) and resolved.strip():
                    return resolved.strip()
    return ""


def parse_graph(graph: dict) -> dict:
    """Return {"positive","negative","params":{...},"loras":[...],"notes":[...]}"""
    notes = []
    sampler_id, sampler = _find_sampler(graph)
    if sampler is not None:
        nodes = [sampler] + _chain(graph, sampler)
    else:
        nodes = list(graph.values())
        notes.append("no recognisable sampler node; scanned the whole graph")

    params = {}
    widened = []
    for key, names in SCALAR_NAMES.items():
        found = _scalar(nodes, names, graph)
        if found is None and sampler is not None:
            found = _scalar(list(graph.values()), names, graph)  # widen before giving up
            if found is not None:
                widened.append(key)
        if found is not None:
            params[key] = found
    if widened:
        notes.append("taken from outside the sampler chain: " + ", ".join(widened))

    # Sigma-driven workflows (ManualSigmas, SplitSigmas) have no `steps` input
    # at all - the step count is the length of the sigma schedule minus one.
    if "steps" not in params:
        for node in nodes:
            raw = (node.get("inputs") or {}).get("sigmas")
            if isinstance(raw, str) and "," in raw:
                count = len([p for p in raw.split(",") if p.strip()])
                if count > 1:
                    params["steps"] = count - 1
                    notes.append(f"steps derived from a {count}-value sigma schedule")
                    break

    positive = negative = ""
    if sampler is not None:
        guider = None
        for node in nodes:
            if node.get("class_type") in GUIDER_NODES or {"positive", "negative"} <= set(node.get("inputs") or {}):
                guider = node
                break
        if guider is not None:
            inputs = guider.get("inputs") or {}
            positive = _text_of(graph, inputs.get("positive"))
            negative = _text_of(graph, inputs.get("negative"))
        elif "positive" in (sampler.get("inputs") or {}):
            positive = _text_of(graph, sampler["inputs"].get("positive"))
            negative = _text_of(graph, sampler["inputs"].get("negative"))

    texts, clip_texts = [], []
    for node in graph.values():
        class_type = node.get("class_type", "")
        if class_type not in TEXT_NODES:
            continue
        inputs = node.get("inputs") or {}
        value = resolve(graph, inputs.get("text"))
        if not isinstance(value, str):
            value = resolve(graph, inputs.get("value"))
        if isinstance(value, str) and value.strip():
            texts.append(value.strip())
            if "CLIPTextEncode" in class_type:
                clip_texts.append(value.strip())
    if not positive:
        has_clip = any("CLIPTextEncode" in (n.get("class_type") or "") for n in graph.values())
        if clip_texts:
            positive = clip_texts[0]
            notes.append("positive prompt guessed from the first CLIP text encode")
        elif not has_clip and texts:
            # API and LLM workflows (Gemini, Nano Banana) carry the prompt in a
            # plain text node because they have no CLIP at all.
            positive = texts[0]
            notes.append("no CLIP in this graph; prompt taken from a text node")
        elif texts:
            # A CLIP encoder exists but its text is computed at runtime. Handing
            # back some other text node - they hold Python snippets and
            # filenames here - would be worse than handing back nothing.
            notes.append(f"no prompt recoverable; {len(texts)} text node(s) hold "
                         f"runtime-computed values - see Workflow Texts")

    loras = []
    for node in graph.values():
        inputs = node.get("inputs") or {}
        name = inputs.get("lora_name")
        if isinstance(name, str) and name:
            weight = inputs.get("strength_model", inputs.get("strength"))
            loras.append({"name": name,
                          "weight": weight if isinstance(weight, (int, float)) else None})

    passes = len(all_samplers(graph))
    if passes > 1:
        notes.append(f"{passes} sampler passes in this graph; reporting the one that "
                     f"feeds the output node")
    params["node_count"] = len(graph)
    params["sampler_passes"] = passes
    return {"positive": positive, "negative": negative, "params": params,
            "loras": loras, "texts": texts, "notes": notes}
