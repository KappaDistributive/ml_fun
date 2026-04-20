#!/usr/bin/env uv run mjpython
import time
import xml.etree.ElementTree as ET
from math import pi
from pathlib import Path

import mujoco
import mujoco.viewer
from dm_control import mjcf


def convert_to_dictionary(qpos) -> dict[str, float]:
    return {
        "shoulder_pan": qpos[0] * 180.0 / pi,  # convert to degrees
        "shoulder_lift": qpos[1] * 180.0 / pi,  # convert to degrees
        "elbow_flex": qpos[2] * 180.0 / pi,  # convert to degrees
        "wrist_flex": qpos[3] * 180.0 / pi,  # convert to degrees
        "wrist_roll": qpos[4] * 180.0 / pi,  # convert to degrees
        "gripper": qpos[5] * 100 / pi,  # convert to 0-100 range
    }


def convert_to_list(data: dict[str, float]) -> list[float]:
    return [
        data["shoulder_pan"] * pi / 180.0,
        data["shoulder_lift"] * pi / 180.0,
        data["elbow_flex"] * pi / 180.0,
        data["wrist_flex"] * pi / 180.0,
        data["wrist_roll"] * pi / 180.0,
        data["gripper"] * pi / 100.0,
    ]


def send_position_command(d, position_dict: dict):
    pos = convert_to_list(position_dict)
    d.ctrl = pos


starting_position = {
    "shoulder_pan": 20.0,  # in degrees
    "shoulder_lift": 1.2,
    "elbow_flex": 1.0,
    "wrist_flex": 2.0,
    "wrist_roll": 3.0,
    "gripper": 10.0,  # 0-100 range
}

if __name__ == "__main__":
    root_model_path = Path(__file__).parent / "sim/Simulation/SO101/scene.xml"
    spec = mujoco.MjSpec.from_file(str(root_model_path.absolute()))
    box = spec.worldbody.add_body()
    box.name = "box"
    box.pos = [0.25, 0.0, 0.02]

    box.add_freejoint()

    geom = box.add_geom()
    geom.name = "box_geom"
    geom.type = mujoco.mjtGeom.mjGEOM_BOX
    geom.size = [0.02, 0.02, 0.02]
    geom.rgba = [0.8, 0.2, 0.2, 1]
    geom.condim = 4
    geom.friction = [2.0, 0.05, 0.01]
    # geom.solimp=[0.99, 0.999, 0.001]
    geom.solref = [0.004, 1.0]
    geom.mass = 0.005

    model = spec.compile()
    data = mujoco.MjData(model)
    # with mujoco.viewer.launch_passive(mode0l, data) as viewer:
    with mujoco.viewer.launch(model, data) as viewer:
        while True:
            starting_position["shoulder_pan"] += 0.1

            send_position_command(data, starting_position)
            mujoco.mj_step(model, data)
            viewer.sync()
