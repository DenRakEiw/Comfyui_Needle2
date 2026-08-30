"""Read a ComfyUI workflow back out of a PNG and turn it into wired values.

ComfyUI saves the executed graph into a `prompt` text chunk on almost every
image it writes, so the settings behind a picture can be read back exactly.

Nothing here uses Needle 2. A workflow graph is exact structured data, and
walking it is exact too; a 45M model reading JSON through a 256-token window
would be strictly worse.
"""

from __future__ import annotations

import json
import os

from . import comfy_graph
from .nodes import CATEGORY, SAMPLERS, SCHEDULERS, _snap

try:
    import folder_paths
except Exception:  # noqa: BLE001 - keep importable outside ComfyUI
    folder_paths = None

META_CATEGORY = CATEGORY + "/workflow"


class WorkflowFromImage:
    """Pull the raw `prompt` (or `workflow`) JSON out of an image in ComfyUI/input."""

    @classmethod
    def INPUT_TYPES(cls):
        files = []
        if folder_paths is not None:
            try:
                directory = folder_paths.get_input_directory()
                files = sorted(f for f in os.listdir(directory)
                               if os.path.isfile(os.path.join(directory, f)))
            except Exception:  # noqa: BLE001
                files = []
        return {
            "required": {
                "image": (files or ["<no files in input/>"], {"image_upload": True}),
                "chunk": (["prompt", "workflow"], {
                    "default": "prompt",
                    "tooltip": "`prompt` is the executed graph and the one worth parsing. "
                               "`workflow` is the editor layout."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "BOOLEAN")
    RETURN_NAMES = ("json", "source", "found")
    FUNCTION = "run"
    CATEGORY = META_CATEGORY
    DESCRIPTION = __doc__

    def run(self, image, chunk="prompt"):
        from PIL import Image

        path = image
        if folder_paths is not None and not os.path.isabs(path):
            try:
                path = folder_paths.get_annotated_filepath(image)
            except Exception:  # noqa: BLE001
                path = os.path.join(folder_paths.get_input_directory(), image)
        if not os.path.isfile(path):
            return ("", f"file not found: {image}", False)

        with Image.open(path) as handle:
            info = dict(handle.info)

        for key in (chunk, "prompt", "workflow"):
            value = info.get(key)
            if isinstance(value, bytes):
                value = value.decode("utf-8", "replace")
            if value:
                return (str(value), f"PNG text chunk: {key}", True)
        return ("", f"no ComfyUI metadata in {os.path.basename(path)} "
                    f"(chunks: {', '.join(info) or 'none'})", False)


class WorkflowParse:
    """ComfyUI workflow JSON -> prompts and sampler settings.

    Real graphs here are dominated by SamplerCustomAdvanced (1031 of the
    sampler nodes across this install) with KSamplerSelect, not plain KSampler
    (51), so the walk follows the sampler's links - guider, sigmas, noise -
    rather than looking for one node class. Anything not on that path is found
    by a graph-wide scan, and `missing` says which values were not in the graph
    at all, so a default never masquerades as a recovered setting.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_text": ("STRING", {"multiline": True, "default": "",
                                         "tooltip": "The `prompt` chunk. Wire in "
                                                    "Workflow From Image, or paste it."}),
            },
            "optional": {
                "default_steps": ("INT", {"default": 20, "min": 1, "max": 1000}),
                "default_cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "default_width": ("INT", {"default": 1024, "min": 16, "max": 16384, "step": 8}),
                "default_height": ("INT", {"default": 1024, "min": 16, "max": 16384, "step": 8}),
                "default_sampler": (SAMPLERS,),
                "default_scheduler": (SCHEDULERS,),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "FLOAT", SAMPLERS, SCHEDULERS,
                    "INT", "INT", "INT", "FLOAT", "STRING", "STRING",
                    "NEEDLE_RESULT", "BOOLEAN", "STRING", "STRING")
    RETURN_NAMES = ("positive", "negative", "steps", "cfg", "sampler_name", "scheduler",
                    "seed", "width", "height", "denoise", "model", "loras_json",
                    "result", "parsed", "missing", "info")
    FUNCTION = "run"
    CATEGORY = META_CATEGORY
    DESCRIPTION = __doc__

    def run(self, json_text, default_steps=20, default_cfg=7.0, default_width=1024,
            default_height=1024, default_sampler=None, default_scheduler=None):
        graph = comfy_graph.load_graph(json_text)
        if graph is None:
            hint = ("empty input" if not (json_text or "").strip()
                    else "not a ComfyUI prompt graph - this node wants the `prompt` "
                         "chunk, a JSON object of nodes with class_type")
            return ("", "", default_steps, default_cfg,
                    default_sampler or SAMPLERS[0], default_scheduler or SCHEDULERS[0],
                    0, default_width, default_height, 1.0, "", "[]",
                    {"arguments": {}, "envelope": {}}, False, "everything", hint)

        walked = comfy_graph.parse_graph(graph)
        found = walked["params"]
        missing = []

        def number(key, fallback, cast, lo=None, hi=None):
            if key not in found:
                missing.append(key)
                return fallback
            try:
                value = cast(found[key])
            except (TypeError, ValueError):
                missing.append(key)
                return fallback
            if lo is not None:
                value = max(lo, min(hi, value))
            return value

        steps = number("steps", default_steps, int, 1, 1000)
        cfg = number("cfg", default_cfg, float, 0.0, 100.0)
        seed = number("seed", 0, int, 0, 0xFFFFFFFFFFFFFFF)
        width = number("width", default_width, int, 16, 16384)
        height = number("height", default_height, int, 16, 16384)
        denoise = number("denoise", 1.0, float, 0.0, 1.0)

        for key in ("sampler_name", "scheduler", "model"):
            if key not in found:
                missing.append(key)
        if not walked["positive"]:
            missing.append("positive")

        info = "\n".join(filter(None, [
            f"{found.get('node_count', 0)} nodes, recovered "
            f"{len([k for k in found if k != 'node_count'])} settings",
            f"loras: {', '.join(e['name'] for e in walked['loras'])}" if walked["loras"] else "",
            f"{len(walked['texts'])} text node(s) in the graph" if walked["texts"] else "",
            *walked["notes"],
        ]))
        return (
            walked["positive"], walked["negative"], steps, cfg,
            _snap(found.get("sampler_name"), SAMPLERS, default_sampler or SAMPLERS[0]),
            _snap(found.get("scheduler"), SCHEDULERS, default_scheduler or SCHEDULERS[0]),
            seed, width, height, denoise,
            str(found.get("model", "") or ""),
            json.dumps(walked["loras"], ensure_ascii=False),
            # Same shape as NeedleExtract, so Needle Get Field reads any key.
            {"arguments": found, "envelope": {}},
            True,
            ", ".join(missing) if missing else "",
            info,
        )


class WorkflowTexts:
    """Every text/prompt string in the graph, for workflows the sampler walk cannot map.

    Custom prompt nodes are everywhere in real graphs - PrimitiveString, ttN
    text, VRGDG_PromptSplitter and so on - and a positive/negative split is not
    always recoverable. This lists what is there so you can pick.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_text": ("STRING", {"multiline": True, "default": ""}),
                "index": ("INT", {"default": 0, "min": 0, "max": 999,
                                  "tooltip": "Which text to return on the `text` output."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("text", "all_texts", "count")
    FUNCTION = "run"
    CATEGORY = META_CATEGORY
    DESCRIPTION = __doc__

    def run(self, json_text, index=0):
        graph = comfy_graph.load_graph(json_text)
        if graph is None:
            return ("", "", 0)
        texts = comfy_graph.parse_graph(graph)["texts"]
        picked = texts[index] if 0 <= index < len(texts) else ""
        joined = "\n\n---\n\n".join(f"[{i}] {t}" for i, t in enumerate(texts))
        return (picked, joined, len(texts))


NODE_CLASS_MAPPINGS = {
    "WorkflowFromImage": WorkflowFromImage,
    "WorkflowParse": WorkflowParse,
    "WorkflowTexts": WorkflowTexts,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WorkflowFromImage": "Workflow From Image",
    "WorkflowParse": "Workflow Parse (graph to values)",
    "WorkflowTexts": "Workflow Texts",
}
