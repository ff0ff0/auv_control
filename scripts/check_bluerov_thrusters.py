#!/usr/bin/env python3
"""Check BlueROV Heavy thruster transforms loaded from USD against the URDF."""

from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
THRUSTER_DYNAMICS_PATH = (
    REPO_ROOT / "source/auv_control/auv_control/tasks/direct/warpauv/thruster_dynamics.py"
)
DEFAULT_USD_PATH = (
    "/home/linc/project/RL/isaac_underwater/isaac_navigation_task/assets/BlueRov/"
    "bluerov_heavy/bluerov_heavy/bluerov_heavy.usd"
)
DEFAULT_URDF_PATH = (
    "/home/linc/project/RL/isaac_underwater/isaac_navigation_task/assets/BlueRov/"
    "bluerov_heavy/bluerov_heavy.urdf"
)

THRUSTER_PRIM_NAMES = [
    "Thruster_R_FORWARD_FRONT",
    "Thruster_L_FORWARD_FRONT",
    "Thruster_R_FORWARD_REAR",
    "Thruster_L_FORWARD_REAR",
    "Thruster_R_UP_FRONT",
    "Thruster_L_UP_FRONT",
    "Thruster_R_UP_REAR",
    "Thruster_L_UP_REAR",
]


def load_thruster_reader():
    spec = importlib.util.spec_from_file_location("warpauv_thruster_dynamics", THRUSTER_DYNAMICS_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load thruster dynamics module from: {THRUSTER_DYNAMICS_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_thruster_com_and_orientations


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd", default=DEFAULT_USD_PATH, help="Path to the BlueROV USD asset.")
    parser.add_argument("--urdf", default=DEFAULT_URDF_PATH, help="Path to the BlueROV URDF asset.")
    parser.add_argument("--pos_tol", type=float, default=1e-3, help="Position tolerance in meters.")
    parser.add_argument("--dir_tol_deg", type=float, default=3.0, help="Direction tolerance in degrees.")
    return parser.parse_args()


def parse_urdf_thrusters(urdf_path: str) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    thruster_map: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    for joint in root.findall("joint"):
        name = joint.attrib.get("name", "")
        short_name = name.split("/")[-1]
        if short_name not in THRUSTER_PRIM_NAMES:
            continue

        origin = joint.find("origin")
        if origin is None:
            raise ValueError(f"Joint '{name}' has no origin element in URDF.")

        xyz = [float(v) for v in origin.attrib.get("xyz", "0 0 0").split()]
        rpy = [float(v) for v in origin.attrib.get("rpy", "0 0 0").split()]
        thruster_map[short_name] = (
            torch.tensor(xyz, dtype=torch.float32),
            rpy_to_x_axis(*rpy),
        )

    missing = [name for name in THRUSTER_PRIM_NAMES if name not in thruster_map]
    if missing:
        raise ValueError(f"Missing thrusters in URDF: {missing}")

    return thruster_map


def rpy_to_x_axis(roll: float, pitch: float, yaw: float) -> torch.Tensor:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rot = torch.tensor(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=torch.float32,
    )
    x_axis = rot @ torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
    return x_axis / torch.linalg.norm(x_axis)


def quat_to_x_axis(quat_wxyz: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat_wxyz.tolist()
    rot = torch.tensor(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=torch.float32,
    )
    x_axis = rot @ torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
    return x_axis / torch.linalg.norm(x_axis)


def main():
    args = parse_args()
    get_thruster_com_and_orientations = load_thruster_reader()

    urdf_thrusters = parse_urdf_thrusters(args.urdf)
    usd_positions, usd_quats = get_thruster_com_and_orientations(
        device=torch.device("cpu"),
        usd_path=args.usd,
        thruster_prim_names=THRUSTER_PRIM_NAMES,
    )

    print(f"USD : {args.usd}")
    print(f"URDF: {args.urdf}")
    print()

    all_ok = True
    for idx, name in enumerate(THRUSTER_PRIM_NAMES):
        urdf_pos, urdf_dir = urdf_thrusters[name]
        usd_pos = usd_positions[idx].cpu()
        usd_dir = quat_to_x_axis(usd_quats[idx].cpu())

        pos_err = torch.linalg.norm(usd_pos - urdf_pos).item()
        cos_sim = torch.clamp(torch.dot(usd_dir, urdf_dir), -1.0, 1.0).item()
        dir_err_deg = math.degrees(math.acos(cos_sim))

        pos_ok = pos_err <= args.pos_tol
        dir_ok = dir_err_deg <= args.dir_tol_deg
        status = "PASS" if pos_ok and dir_ok else "FAIL"
        all_ok = all_ok and pos_ok and dir_ok

        print(f"[{status}] {name}")
        print(f"  usd_pos  = {[round(v, 6) for v in usd_pos.tolist()]}")
        print(f"  urdf_pos = {[round(v, 6) for v in urdf_pos.tolist()]}")
        print(f"  pos_err  = {pos_err:.6f} m")
        print(f"  usd_dir  = {[round(v, 6) for v in usd_dir.tolist()]}")
        print(f"  urdf_dir = {[round(v, 6) for v in urdf_dir.tolist()]}")
        print(f"  dir_err  = {dir_err_deg:.4f} deg")
        print()

    print("Overall:", "PASS" if all_ok else "FAIL")
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
