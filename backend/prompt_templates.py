from models import PerceptionData


SYSTEM_PROMPT = """You control a humanoid robot. Respond with JSON only.
Format: {"thought":"...","speech":"...","commands":[{"action":"...","params":{}}]}
Actions:
- walk: {"action":"walk","params":{"distance_m":2.0,"speed":0.5}}
- turn: {"action":"turn","params":{"degrees":90}}  (positive = left, negative = right)
- wave: {"action":"wave","params":{"arm":"left"|"right"}}
- seek_red_ball: {"action":"seek_red_ball","params":{}}  (turn, walk to, and stop at the red ball)
- stop: {"action":"stop","params":{}}
- speak: {"action":"speak","params":{"text":"..."}}
Max 2 commands. Do not invent actions such as look_at or point_at."""  # noqa: E501

DIR_NAMES = ["NORTH", "NORTHWEST", "WEST", "SOUTHWEST", "SOUTH", "SOUTHEAST", "EAST", "NORTHEAST"]


def _facing_name(radians: float) -> str:
    idx = round(radians / (3.14159 / 4)) % 8
    return DIR_NAMES[idx]


def build_prompt(perception: PerceptionData, user_input: str) -> str:
    robot = perception.robot
    facing = _facing_name(robot.rotation)

    objects = perception.objects[:5]
    obj_lines = [f"  {o.id}: {o.label} {o.distance:.0f}m conf={o.confidence:.2f}"
                 for o in objects]

    scene = f"Robot at ({robot.position[0]:.0f},{robot.position[1]:.0f}) facing {facing}"
    obj_str = "\n".join(obj_lines) if obj_lines else "none"

    return f"{SYSTEM_PROMPT}\n\n{scene}\nObjects:\n{obj_str}\n\nUser: {user_input}"
