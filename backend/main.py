import asyncio
import logging
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from physics_world import PhysicsWorld
from perception import Perception
from controller import Controller
from models import RobotState
from config import OLLAMA_URL, LLM_MODEL, USE_TRAINED, WS_PORT, TICK_RATE, FRONTEND_DIR

CAMERA_INTERVAL = 5

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if USE_TRAINED:
    from trained_reasoner import TrainedReasoner as Reasoner, scene_to_str
    from speech_generator import SpeechGenerator
    logger.info("Using trained ONNX model + Gemma 1b for speech")
    speech_gen = SpeechGenerator(OLLAMA_URL)
else:
    from reasoner import Reasoner
    speech_gen = None
    logger.info("Using Ollama LLM")

app = FastAPI(title="Robot Control System")

physics_world = PhysicsWorld()
reasoner = Reasoner(OLLAMA_URL, LLM_MODEL)
perception = Perception(physics_world)
controller = Controller(physics_world)

if FRONTEND_DIR and os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="frontend")


@app.get("/")
async def index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.on_event("shutdown")
async def shutdown():
    perception.close()
    physics_world.close()
    close = getattr(reasoner, "close", None)
    if close:
        close()
    if speech_gen is not None:
        await speech_gen.close()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket client connected")

    stop_event = asyncio.Event()

    _cam_counter = 0

    async def state_stream():
        nonlocal _cam_counter
        try:
            while not stop_event.is_set():
                joints = controller.tick(1.0 / TICK_RATE)
                pos = controller.current_position
                rot = controller.current_rotation
                robot_state = _build_robot_state(pos, rot, joints)
                scene = perception.get_scene(robot_state)
                controller.set_scene_objects(scene.objects)
                controller.set_red_ball_target(perception.get_red_ball_target())

                speech = controller.get_speech()

                state_est = controller.state_est
                left_contact, right_contact = controller.foot_contacts
                await websocket.send_json({
                    "type": "state",
                    "joints": joints,
                    "robot_pos": list(pos),
                    "robot_rot": rot,
                    "robot_euler": list(state_est["base_orn"]) if state_est is not None else [0,0,0],
                    "robot_vel": list(state_est["base_vel_linear"]) if state_est is not None else [0,0,0],
                    "objects": [o.__dict__ for o in scene.objects],
                    "speech": speech,
                    "gait_phase": controller.gait_phase,
                    "walking": controller.walking,
                    "seeking_red_ball": controller.seeking_red_ball,
                    "red_ball_seen": perception.get_red_ball_target() is not None,
                    "left_contact": bool(left_contact),
                    "right_contact": bool(right_contact),
                    "obstacle": controller.obstacle_warning,
                })

                _cam_counter += 1
                if _cam_counter >= CAMERA_INTERVAL:
                    _cam_counter = 0
                    cam_b64, dets = perception.get_camera_frame()
                    if cam_b64:
                        await websocket.send_json({
                            "type": "camera",
                            "image": cam_b64,
                            "detections": dets,
                        })

                await asyncio.sleep(1.0 / TICK_RATE)
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.error(f"State stream error: {e}")

    async def command_listener():
        try:
            while not stop_event.is_set():
                data = await websocket.receive_json()
                if data.get("type") == "seek_red_ball":
                    logger.info("User requested red-ball seek")
                    controller.start_red_ball_seek()
                    continue

                if data.get("type") == "command":
                    user_text = data.get("text", "")
                    logger.info(f"User command: {user_text}")

                    if "red ball" in user_text.lower() or "redball" in user_text.lower():
                        logger.info("Natural-language red-ball request")
                        controller.start_red_ball_seek()
                        controller.set_speech("Looking for the red ball.")
                        continue

                    robot_state = _build_robot_state(
                        controller.current_position, controller.current_rotation, controller.current_joints,
                    )
                    scene = perception.get_scene(robot_state)
                    controller.set_scene_objects(scene.objects)

                    if USE_TRAINED:
                        commands = await reasoner.reason(scene, user_text)
                        controller.enqueue(commands)
                        logger.info(f"T5 commands: {[c.action for c in commands]}")

                        async def delayed_speech():
                            speech = await speech_gen.generate(
                                scene_to_str(scene), user_text
                            )
                            if speech:
                                controller.set_speech(speech)
                                logger.info(f"Speech: {speech}")

                        asyncio.create_task(delayed_speech())
                    else:
                        response = await reasoner.reason(scene, user_text)
                        controller.enqueue(response.commands)
                        if response.speech:
                            controller.set_speech(response.speech)
                        logger.info(f"LLM speech: {response.speech}")
                        logger.info(f"LLM commands: {[c.action for c in response.commands]}")
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.error(f"Command listener error: {e}")

    tasks = {asyncio.create_task(state_stream()), asyncio.create_task(command_listener())}
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            exc = task.exception()
            if exc and not isinstance(exc, WebSocketDisconnect):
                raise exc
    except WebSocketDisconnect:
        pass
    finally:
        stop_event.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("WebSocket client disconnected")


def _build_robot_state(position, rotation, joints):
    return RobotState(position=position, rotation=rotation, joints=joints)


def main():
    import uvicorn
    logger.info(f"Starting server on port {WS_PORT}")
    uvicorn.run(app, host="0.0.0.0", port=WS_PORT, log_level="info")


if __name__ == "__main__":
    main()
