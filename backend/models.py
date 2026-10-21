from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SceneObject:
    id: str
    label: str
    confidence: float
    distance: float
    angle: float
    position: tuple[float, float, float]


@dataclass
class RobotState:
    position: tuple[float, float, float]
    rotation: float
    joints: dict[str, float]


@dataclass
class LLMCommand:
    action: str
    params: dict = field(default_factory=dict)


@dataclass
class LLMResponse:
    thought: str
    speech: str
    commands: list[LLMCommand]


@dataclass
class PerceptionData:
    robot: RobotState
    objects: list[SceneObject]
    timestamp: float
