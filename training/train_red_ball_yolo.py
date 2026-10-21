"""Generate synthetic red-ball data from the PyBullet head camera and train a YOLO detector."""

import argparse
import math
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import pybullet as p
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from physics_world import PhysicsWorld  # noqa: E402


def _bbox_from_mask(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) < 20:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def generate(out_dir: Path, count: int, negatives: int):
    images = out_dir / "images"
    labels = out_dir / "labels"
    for split in ("train", "val"):
        (images / split).mkdir(parents=True, exist_ok=True)
        (labels / split).mkdir(parents=True, exist_ok=True)

    pw = PhysicsWorld()
    ball = pw.objects["red_ball"]
    ball_body = ball["body"]
    rng = random.Random(7)
    made = 0
    attempts = 0

    def render():
        info = pw.get_head_camera_image(320, 240)
        rgb = np.frombuffer(bytes(info[2]), dtype=np.uint8).reshape(240, 320, 4)[:, :, :3]
        seg = np.array(info[4], dtype=np.int32).reshape(240, 320)
        return rgb, seg

    def place(ball_visible: bool):
        robot_pos, _ = pw.get_robot_position()
        yaw = rng.uniform(-math.pi, math.pi)
        pw.reset_robot_position(robot_pos[0], robot_pos[1], robot_pos[2], yaw)
        if ball_visible:
            bearing = yaw + math.radians(rng.uniform(-38, 38))
            dist = rng.uniform(1.3, 8.0)
        else:
            bearing = yaw + math.radians(rng.uniform(95, 180)) * rng.choice((-1, 1))
            dist = rng.uniform(1.5, 8.0)
        bx = robot_pos[0] + dist * math.cos(bearing)
        by = robot_pos[1] + dist * math.sin(bearing)
        p.resetBasePositionAndOrientation(ball_body, [bx, by, 0.18], [0, 0, 0, 1],
                                          physicsClientId=pw.client)
        p.resetBaseVelocity(ball_body, [0, 0, 0], [0, 0, 0], physicsClientId=pw.client)
        for name, value in (("neck_yaw", rng.uniform(-0.35, 0.35)),
                            ("neck_pitch", rng.uniform(0.0, 0.45))):
            idx = pw.joint_indices[name]
            p.resetJointState(pw.robot_id, idx, value, 0.0, physicsClientId=pw.client)
        pw.step(1.0 / 60.0, steps=3)

    while made < count:
        attempts += 1
        if attempts > count * 8:
            raise RuntimeError("could not generate enough positive frames")
        place(ball_visible=True)
        rgb, seg = render()
        bbox = _bbox_from_mask(seg == ball_body)
        if bbox is None or (bbox[2] - bbox[0]) < 8 or (bbox[3] - bbox[1]) < 8:
            continue
        split = "val" if made % 8 == 0 else "train"
        stem = f"red_ball_{made:04d}"
        Image.fromarray(rgb).save(images / split / f"{stem}.jpg", quality=92)
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2 / 320, (y1 + y2) / 2 / 240
        bw, bh = (x2 - x1) / 320, (y2 - y1) / 240
        (labels / split / f"{stem}.txt").write_text(
            f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"
        )
        made += 1

    for i in range(negatives):
        place(ball_visible=False)
        rgb, seg = render()
        if _bbox_from_mask(seg == ball_body) is not None:
            continue
        split = "val" if i % 8 == 0 else "train"
        stem = f"no_ball_{i:04d}"
        Image.fromarray(rgb).save(images / split / f"{stem}.jpg", quality=92)
        (labels / split / f"{stem}.txt").write_text("")

    (out_dir / "data.yaml").write_text(
        f"path: {out_dir.resolve()}\ntrain: images/train\nval: images/val\n"
        "names:\n  0: red_ball\n"
    )
    pw.close()
    print(f"generated {made} positives + {negatives} negatives in {out_dir}")


def train(data_dir: Path, epochs: int, imgsz: int, batch: int):
    from ultralytics import YOLO

    model = YOLO(str(ROOT / "models" / "yolo11n.pt"))
    results = model.train(
        data=str(data_dir / "data.yaml"),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device="cpu",
        workers=0,
        project=str(data_dir.parent / "red_ball_runs"),
        name="train",
        exist_ok=True,
        pretrained=True,
        verbose=True,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    dest = ROOT / "models" / "red_ball_yolo.pt"
    shutil.copyfile(best, dest)
    print(f"copied {best} -> {dest}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "training" / "red_ball_dataset")
    ap.add_argument("--count", type=int, default=240)
    ap.add_argument("--negatives", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--skip-generate", action="store_true")
    args = ap.parse_args()
    if not args.skip_generate:
        if args.out.exists():
            shutil.rmtree(args.out)
        generate(args.out, args.count, args.negatives)
    train(args.out, args.epochs, args.imgsz, args.batch)
