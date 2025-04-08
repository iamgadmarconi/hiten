from __future__ import annotations
from typing import Optional

# Import custom logger
from log_config import logger


class Body(object):

    # Type hints for instance attributes
    name: str
    mass: float
    radius: float
    color: str
    parent: Body # A body's parent is always another Body instance (itself if it's the primary)

    def __init__(self, name: str, mass: float, radius: float, color: Optional[str] = None, parent: Optional[Body] = None):
        self.name = name
        self.mass = mass
        self.radius = radius
        self.color = color if color else "#000000" # Default color if none provided
        self.parent = parent if parent else self # If no parent, it orbits itself (primary body)

        # Log the creation of the body
        parent_name = self.parent.name if self.parent is not self else "None"
        logger.info(f"Created Body: name='{self.name}', mass={self.mass}, radius={self.radius}, color='{self.color}', parent='{parent_name}'")

    def __str__(self) -> str:
        parent_desc = f"orbiting {self.parent.name}" if self.parent is not self else "(Primary)"
        return f"{self.name} {parent_desc}"

    def __repr__(self) -> str:
        parent_repr = f"parent={self.parent.name!r}" if self.parent is not self else "parent=self"
        return f"Body(name={self.name!r}, mass={self.mass}, radius={self.radius}, color={self.color!r}, {parent_repr})"
