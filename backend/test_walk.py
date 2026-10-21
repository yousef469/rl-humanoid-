"""Test walking stability with deterministic environment."""
import sys, os, time, math, random
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from physics_world import PhysicsWorld
from controller import Controller

TICK_RATE = 60
FALL_PITCH = 0.6
FALL_ROLL = 0.5
FALL_HEIGHT = 0.3

class CleanWorld(PhysicsWorld):
    """Physics world with no random objects for deterministic testing."""
    def _spawn_objects(self, count=0):
        pass

def run_speed_test(speed, duration=300, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    pw = CleanWorld()
    ctrl = Controller(pw)
    ctrl.vx_des = speed * 0.35
    ctrl._walk_enabled = True

    dt = 1.0 / TICK_RATE
    total_steps = int(duration * TICK_RATE)
    max_pitch = max_roll = 0.0
    min_height = 999.0
    fall_step = None
    last_yaw = 0.0
    total_yaw_drift = 0.0
    start_pos = None

    for i in range(total_steps):
        ctrl.tick(dt)
        pos, yaw = pw.get_robot_position()
        euler = pw.get_robot_orientation_euler()
        pitch, roll = abs(euler[0]), abs(euler[1])
        height = pos[2]
        if start_pos is None:
            start_pos = pos[:2]

        max_pitch = max(max_pitch, pitch)
        max_roll = max(max_roll, roll)
        min_height = min(min_height, height)

        if i % TICK_RATE == 0:
            yaw_delta = yaw - last_yaw
            if yaw_delta > math.pi: yaw_delta -= 2*math.pi
            if yaw_delta < -math.pi: yaw_delta += 2*math.pi
            total_yaw_drift += abs(yaw_delta)
            last_yaw = yaw

        if pitch > FALL_PITCH or roll > FALL_ROLL or height < FALL_HEIGHT:
            fall_step = i
            if i > TICK_RATE * 3:
                break

    dx = pos[0] - start_pos[0]
    dy = pos[1] - start_pos[1]
    dist = math.sqrt(dx*dx + dy*dy)

    pw.close()
    alive_time = (fall_step or total_steps) / TICK_RATE
    return {
        "speed": speed,
        "alive": alive_time,
        "distance": round(dist, 2),
        "max_pitch": round(max_pitch, 3),
        "max_roll": round(max_roll, 3),
        "min_height": round(min_height, 3),
        "yaw_drift_deg": round(math.degrees(total_yaw_drift), 1),
        "fell": fall_step is not None and fall_step > TICK_RATE * 3,
    }

def run_trials(speed, duration, trials=3):
    results = []
    for t in range(trials):
        r = run_speed_test(speed, duration, seed=t)
        results.append(r)
        status = "FELL" if r["fell"] else "OK"
        print(f"  trial {t}: {status} at {r['alive']:.0f}s  pitch={r['max_pitch']} yaw_drift={r['yaw_drift_deg']}°")
    return results

if __name__ == "__main__":
    speed = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    dur = float(sys.argv[2]) if len(sys.argv) > 2 else 120
    trials = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    print(f"\n=== Walking speed={speed} for {dur:.0f}s ({trials} trials) ===")
    t0 = time.time()
    if trials > 1:
        rs = run_trials(speed, dur, trials)
        avg_alive = sum(r["alive"] for r in rs) / len(rs)
        fell_count = sum(1 for r in rs if r["fell"])
        avg_drift = sum(r["yaw_drift_deg"] for r in rs) / len(rs)
        pitch_max = max(r["max_pitch"] for r in rs)
        print(f"  Avg alive: {avg_alive:.0f}s  falls: {fell_count}/{trials}")
        print(f"  Max pitch: {pitch_max}  avg yaw drift: {avg_drift:.0f}°/trial")
    else:
        r = run_speed_test(speed, dur)
        status = "❌ FELL" if r["fell"] else "✅ OK"
        print(f"  {status} at {r['alive']:.0f}s  pitch={r['max_pitch']} roll={r['max_roll']}")
        print(f"  dist={r['distance']}m  yaw_drift={r['yaw_drift_deg']}°  min_h={r['min_height']}")
    print(f"  wall time: {time.time()-t0:.1f}s")
