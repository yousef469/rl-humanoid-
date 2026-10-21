"""Generate synthetic (scene + command -> command JSON) pairs for the T5 controller.

The target schema matches ``backend/controller.py``:
    walk:  {"action": "walk",  "params": {"distance_m": 2.0, "speed": 0.5}}
    turn:  {"action": "turn",  "params": {"degrees": 90}}
    wave:  {"action": "wave",  "params": {"arm": "right"}}
    stop:  {"action": "stop",  "params": {}}

The model is body-control only; it does not generate speech. Run:
    python training/generate_data.py
"""

import json
import os
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(BASE_DIR, "robot_training_data.jsonl")
NUM_SAMPLES = 50000

OBJECT_LABELS = ["tree", "rock", "house", "sign", "person", "car", "bush", "lamp"]
DIR_NAMES = ["NORTH", "NORTHWEST", "WEST", "SOUTHWEST", "SOUTH", "SOUTHEAST", "EAST", "NORTHEAST"]

USER_COMMANDS = [
    "what do you see?", "what's ahead?", "what's in front of me?",
    "describe the scene", "what objects are nearby?",
    "walk forward", "move ahead", "go forward 2 meters", "walk 5 meters",
    "turn left", "turn right", "turn around", "face north", "face east",
    "look at the tree", "look at the rock", "look at that",
    "point at the tree", "point at the house",
    "wave", "wave hello",
    "stop", "halt", "freeze",
    "go to the tree", "walk to the rock", "approach the house",
    "what is that thing?", "what's the closest object?",
    "say hello", "introduce yourself",
]


def random_objects(count=None):
    if count is None:
        count = random.randint(1, 5)
    objects = []
    for i in range(count):
        label = random.choice(OBJECT_LABELS)
        objects.append({
            "id": f"{label}_{i}",
            "label": label,
            "distance": round(random.uniform(1.0, 80.0), 1),
            "angle": round(random.uniform(-60, 60), 0),
        })
    objects.sort(key=lambda o: o["distance"])
    return objects


def build_scene_str(robot_pos, facing, objects):
    obj_str = " | ".join(
        f"{o['label']} {o['distance']}m conf={random.uniform(0.85, 0.99):.2f}"
        for o in objects[:5]
    )
    return f"Robot at ({robot_pos[0]},{robot_pos[1]}) facing {facing} | {obj_str}"


def find_object(command, objects):
    for obj in objects:
        if obj["label"] in command:
            return obj
    return None


def parse_distance(command, default=2.0):
    for word in command.replace(",", " ").split():
        try:
            value = float(word)
        except ValueError:
            continue
        if value > 0:
            return min(value, 20.0)
    return default


def facing_delta(facing, target):
    current = DIR_NAMES.index(facing)
    wanted = DIR_NAMES.index(target)
    steps = (wanted - current + 4) % 8 - 4
    return steps * 45


def generate_commands(command, objects, facing):
    """Return body commands only. Empty list means "no body action"."""
    text = command.lower()

    if any(word in text for word in ("stop", "halt", "freeze")):
        return [{"action": "stop", "params": {}}]

    if any(word in text for word in ("say hello", "introduce", "wave")):
        arm = random.choice(["left", "right"])
        return [{"action": "wave", "params": {"arm": arm}}]

    if "look at" in text or "point at" in text:
        target = find_object(text, objects)
        if target is None:
            return []
        return [{"action": "turn", "params": {"degrees": int(target["angle"])}}]

    if any(word in text for word in ("turn", "face", "rotate")):
        if "around" in text:
            return [{"action": "turn", "params": {"degrees": 180}}]
        if "left" in text:
            return [{"action": "turn", "params": {"degrees": 90}}]
        if "right" in text:
            return [{"action": "turn", "params": {"degrees": -90}}]
        for direction in DIR_NAMES:
            if direction.lower() in text:
                return [{"action": "turn", "params": {"degrees": facing_delta(facing, direction)}}]
        return []

    if any(word in text for word in ("walk", "move", "advance", "approach", "go to")):
        return [{"action": "walk", "params": {"distance_m": parse_distance(text), "speed": 0.5}}]

    if any(word in text for word in ("see", "ahead", "describe", "what", "object", "nearby", "closest")):
        if objects:
            return [{"action": "turn", "params": {"degrees": int(objects[0]["angle"])}}]
        return []

    # Unknown commands request no body motion; speech is handled elsewhere.
    return []


def main():
    random.seed(42)
    with open(OUTPUT_FILE, "w") as handle:
        for _ in range(NUM_SAMPLES):
            position = (0.0, 0.0, 0.0)
            facing = random.choice(DIR_NAMES)
            objects = random_objects()
            scene = build_scene_str(position, facing, objects)
            command = random.choice(USER_COMMANDS)
            commands = generate_commands(command, objects, facing)

            example = {
                "input": f"scene: {scene} | user: {command}",
                "output": json.dumps(commands),
            }
            handle.write(json.dumps(example) + "\n")

    print(f"Generated {NUM_SAMPLES} training examples -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
