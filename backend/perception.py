"""Camera + object perception for the simulated robot.

Vision uses two complementary sources:
  * a cheap 360-degree-ish raycast for physics objects (always available)
  * YOLO on the head camera for extra detections / the frontend overlay
"""

import base64
import io
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

from config import YOLO_MODEL_PATH, YOLO_FALLBACK_MODEL_PATH
from models import SceneObject, RobotState, PerceptionData

MAX_VISIBLE = 10
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
logger = logging.getLogger(__name__)


class Perception:
    def __init__(self, physics_world):
        self._pw = physics_world
        self._tracked: dict[str, tuple[SceneObject, int]] = {}
        self._tick = 0
        self._persist_ticks = 15
        self._yolo = None
        self._yolo_counter = 0
        self._yolo_interval = 10
        self._cached_frame = (None, [])
        self._camera_detections = []
        self._red_ball = None
        self._red_ball_tick = -999
        self._yolo_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="yolo")
        self._yolo_future = None
        self._yolo_ctx = None

    def close(self):
        self._yolo_pool.shutdown(wait=False, cancel_futures=True)

    def _get_yolo(self):
        if self._yolo is None:
            try:
                import os
                from ultralytics import YOLO
                path = YOLO_MODEL_PATH if os.path.exists(YOLO_MODEL_PATH) else YOLO_FALLBACK_MODEL_PATH
                self._yolo = YOLO(path)
            except Exception as exc:
                logger.warning("YOLO unavailable (%s); continuing with raycast perception", exc)
                self._yolo = False
        return self._yolo if self._yolo is not False else None

    def _unproject(self, u, v, depth, view_mat, proj_mat, w, h):
        vm = np.array(view_mat, dtype=np.float64).reshape(4, 4).T
        pm = np.array(proj_mat, dtype=np.float64).reshape(4, 4).T
        x_ndc = 2.0 * u / w - 1.0
        y_ndc = 1.0 - 2.0 * v / h
        z_ndc = 2.0 * depth - 1.0
        clip = np.array([x_ndc, y_ndc, z_ndc, 1.0])
        eye = np.linalg.inv(pm) @ clip
        eye = eye / eye[3]
        world = np.linalg.inv(vm) @ eye
        world = world / world[3]
        return (float(world[0]), float(world[1]), float(world[2]))

    def _refresh_camera(self):
        """Render the head camera once, then reuse it for scene and overlay."""
        img_w, img_h = CAMERA_WIDTH, CAMERA_HEIGHT
        cam_info = self._pw.get_head_camera_image(width=img_w, height=img_h)
        if len(cam_info) < 7:
            self._cached_frame = (None, [])
            self._camera_detections = []
            self._red_ball = None
            return

        raw_rgba = np.frombuffer(bytes(cam_info[2]), dtype=np.uint8).reshape(img_h, img_w, 4)
        depth_buf = np.array(cam_info[3], dtype=np.float64).reshape(img_h, img_w)
        view_mat = cam_info[5]
        proj_mat = cam_info[6]
        rgb = np.ascontiguousarray(raw_rgba[:, :, :3])

        overlay = []
        detections = []
        red_ball = None
        model = self._get_yolo()

        if self._yolo_future is not None and self._yolo_future.done():
            try:
                results = self._yolo_future.result()
                if self._yolo_ctx is not None:
                    overlay, detections, red_ball = self._process_yolo(
                        results, self._yolo_ctx, model
                    )
            except Exception as exc:
                logger.warning("YOLO inference failed: %s", exc)
            finally:
                self._yolo_future = None
                self._yolo_ctx = None

        if model is not None and self._yolo_future is None:
            self._yolo_ctx = (rgb.copy(), depth_buf, view_mat, proj_mat, img_w, img_h)
            self._yolo_future = self._yolo_pool.submit(
                model, rgb.copy(), verbose=False, imgsz=320
            )

        image = Image.fromarray(rgb)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=65)
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        self._cached_frame = (b64, overlay)
        self._camera_detections = detections
        if red_ball is not None:
            red_ball["source"] = "yolo"
            self._red_ball = red_ball
            self._red_ball_tick = self._tick

    def _process_yolo(self, results, ctx, model):
        rgb, depth_buf, view_mat, proj_mat, img_w, img_h = ctx
        overlay, detections, red_ball = [], [], None
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                label = model.names[int(box.cls[0])]
                is_red_ball = (
                    label in ("sports ball", "red_ball")
                    and confidence >= 0.15
                    and self._red_fraction(rgb, x1, y1, x2, y2) >= 0.02
                )
                if confidence < (0.15 if is_red_ball else 0.4):
                    continue
                display_label = "red ball" if is_red_ball else label
                overlay.append({
                    "label": display_label,
                    "confidence": round(confidence, 2),
                    "bbox": [x1, y1, x2, y2],
                })

                cx = max(0, min(img_w - 1, (x1 + x2) // 2))
                cy = max(0, min(img_h - 1, (y1 + y2) // 2))
                depth = float(depth_buf[cy, cx])
                if depth >= 0.99:
                    continue
                position = self._unproject(cx, cy, depth, view_mat, proj_mat, img_w, img_h)
                if is_red_ball:
                    position = (position[0], position[1], 0.0)
                detection = {
                    "label": "red_ball" if is_red_ball else label,
                    "confidence": round(confidence, 2),
                    "position": position,
                    "bbox": (x1, y1, x2, y2),
                }
                detections.append(detection)
                if is_red_ball and (red_ball is None or confidence > red_ball["confidence"]):
                    red_ball = detection
        return overlay, detections, red_ball

    @staticmethod
    def _red_fraction(rgb, x1, y1, x2, y2):
        h, w = rgb.shape[:2]
        x1 = max(0, min(w - 1, x1))
        x2 = max(x1 + 1, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(y1 + 1, min(h, y2))
        crop = rgb[y1:y2, x1:x2].astype(np.int16)
        if crop.size == 0:
            return 0.0
        r, g, b = crop[:, :, 0], crop[:, :, 1], crop[:, :, 2]
        red = (r > 90) & (r - g > 40) & (r - b > 40) & (r > g * 1.5) & (r > b * 1.5)
        return float(red.mean())

    def get_red_ball_target(self, max_age_ticks=30):
        """Return the most recent YOLO/colour-verified red ball, if fresh."""
        if self._red_ball is None or self._tick - self._red_ball_tick > max_age_ticks:
            return None
        target = dict(self._red_ball)
        target["age_ticks"] = self._tick - self._red_ball_tick
        return target

    def get_camera_frame(self):
        if self._cached_frame[0] is None:
            self._refresh_camera()
        return self._cached_frame

    @staticmethod
    def _label_match(yolo_label, physics_label):
        mapping = {
            "person": "person", "car": "car", "truck": "car",
            "traffic light": "lamp", "stop sign": "sign",
            "chair": "sign", "couch": "sign", "tv": "sign",
            "sports ball": "red_ball",
        }
        return yolo_label == physics_label or mapping.get(yolo_label) == physics_label

    def get_scene(self, robot: RobotState) -> PerceptionData:
        self._tick += 1

        pb_detected = self._pw.raycast_vision(num_rays=32, max_distance=100.0)
        ray_target = next((obj for obj in pb_detected if obj["label"] == "red_ball"), None)
        if ray_target is not None:
            previous = self._red_ball or {}
            fresh_yolo = (
                previous.get("source") in ("yolo", "yolo+raycast")
                and self._tick - self._red_ball_tick <= self._yolo_interval
            )
            center = ray_target.get("center", ray_target["position"])
            self._red_ball = {
                "label": "red_ball",
                "confidence": max(0.95, previous.get("confidence", 0.0)) if fresh_yolo else 0.95,
                "position": (center[0], center[1], 0.0),
                "bbox": previous.get("bbox"),
                "source": "yolo+raycast" if fresh_yolo else "raycast",
            }
            self._red_ball_tick = self._tick

        fresh_camera = False
        self._yolo_counter += 1
        if self._yolo_counter >= self._yolo_interval:
            self._yolo_counter = 0
            self._refresh_camera()
            fresh_camera = True

        merged = {}
        for pb in pb_detected:
            merged[pb["id"]] = SceneObject(
                id=pb["id"],
                label=pb["label"],
                confidence=pb.get("confidence", 0.9),
                distance=pb["distance"],
                angle=pb["angle"],
                position=pb["position"],
            )

        camera_detections = self._camera_detections if fresh_camera else []
        self._camera_detections = []

        for yd in camera_detections:
            yp = yd["position"]
            for obj in merged.values():
                dx = yp[0] - obj.position[0]
                dy = yp[1] - obj.position[1]
                if math.hypot(dx, dy) < 3.0 and self._label_match(yd["label"], obj.label):
                    obj.confidence = max(obj.confidence, yd["confidence"])
                    break
            else:
                dx = yp[0] - robot.position[0]
                dz = yp[1] - robot.position[1]
                dist = math.hypot(dx, dz)
                angle_rad = math.atan2(dz, dx) - robot.rotation
                new_id = f"yolo_{len(merged)}_{self._tick}"
                merged[new_id] = SceneObject(
                    id=new_id,
                    label=yd["label"],
                    confidence=yd["confidence"],
                    distance=round(dist, 1),
                    angle=round(math.degrees(angle_rad), 1),
                    position=(round(yp[0], 2), round(yp[1], 2), 0.0),
                )

        for oid, (obj, last_seen) in list(self._tracked.items()):
            if oid not in merged and self._tick - last_seen < self._persist_ticks:
                merged[oid] = obj

        for oid, obj in merged.items():
            self._tracked[oid] = (obj, self._tick)

        self._tracked = {k: v for k, v in self._tracked.items()
                         if self._tick - v[1] < self._persist_ticks * 2}

        if ray_target is not None and self._red_ball is not None:
            center = ray_target.get("center", ray_target["position"])
            self._red_ball["position"] = (center[0], center[1], 0.0)
            if self._red_ball.get("source") == "yolo":
                self._red_ball["source"] = "yolo+raycast"

        objects = sorted(merged.values(), key=lambda obj: obj.distance)[:MAX_VISIBLE]
        return PerceptionData(robot=robot, objects=objects, timestamp=time.time())
