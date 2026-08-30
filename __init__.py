"""Cactus Needle 2 nodes plus ComfyUI workflow-metadata nodes."""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .nodes_workflow import (NODE_CLASS_MAPPINGS as _WF_CLASSES,
                             NODE_DISPLAY_NAME_MAPPINGS as _WF_NAMES)

NODE_CLASS_MAPPINGS.update(_WF_CLASSES)
NODE_DISPLAY_NAME_MAPPINGS.update(_WF_NAMES)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
