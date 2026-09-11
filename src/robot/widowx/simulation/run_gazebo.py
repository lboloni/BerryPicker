#!/usr/bin/env python3
"""Launch the Interbotix WidowX Gazebo simulation with a rendered camera."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


DEFAULT_IMAGE_TOPIC = "/berrypicker/gazebo/camera_0/image_raw"
DEFAULT_CAMERA_MODEL = Path(__file__).with_name("fixed_camera.sdf")
DEFAULT_BRIDGE = Path(__file__).with_name("interbotix_gazebo_bridge.py")


def _positive_float(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("value must be positive and finite")
    return result


def _require_ros2():
    ros2 = shutil.which("ros2")
    if ros2 is None:
        raise RuntimeError("Unable to find ros2; source the ROS environment first")
    return ros2


def _require_camera_plugin(ros2):
    result = subprocess.run(
        [ros2, "pkg", "prefix", "gazebo_plugins"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "gazebo_plugins is unavailable; install ros-humble-gazebo-plugins"
        )
    plugin = Path(result.stdout.strip(), "lib", "libgazebo_ros_camera.so")
    if not plugin.is_file():
        raise RuntimeError(f"Gazebo ROS camera plugin is missing: {plugin}")
    return plugin


def _prepend_path(environment, name, value):
    existing = environment.get(name)
    environment[name] = f"{value}{os.pathsep}{existing}" if existing else str(value)


def _gazebo_environment():
    """Supply paths normally added by Gazebo's setup script."""
    gazebo = shutil.which("gazebo")
    if gazebo is None:
        raise RuntimeError("Unable to find gazebo; install Gazebo Classic first")
    prefix = Path(gazebo).resolve().parents[1]
    shares = sorted(
        path for path in Path(prefix, "share").glob("gazebo-*")
        if Path(path, "media").is_dir()
    )
    if not shares:
        raise RuntimeError(f"Unable to find Gazebo resources under {prefix}")
    share = shares[-1]
    plugin_directories = [
        path for path in Path(prefix, "lib").glob("*/gazebo-*/plugins")
        if Path(path, "libCameraPlugin.so").is_file()
    ]
    if not plugin_directories:
        raise RuntimeError(f"Unable to find Gazebo plugins under {prefix}")

    environment = os.environ.copy()
    _prepend_path(environment, "GAZEBO_RESOURCE_PATH", share)
    _prepend_path(environment, "GAZEBO_MODEL_PATH", Path(share, "models"))
    # The Interbotix launch file appends to this value without adding a separator.
    environment["GAZEBO_MODEL_PATH"] += os.pathsep
    for plugin_directory in plugin_directories:
        _prepend_path(environment, "GAZEBO_PLUGIN_PATH", plugin_directory)
        _prepend_path(environment, "LD_LIBRARY_PATH", plugin_directory)
    ogre_directories = sorted(Path(prefix, "lib").glob("*/OGRE-*"))
    if ogre_directories:
        _prepend_path(environment, "OGRE_RESOURCE_PATH", ogre_directories[-1])
    return environment


def _ros_names(ros2, kind):
    result = subprocess.run(
        [ros2, kind, "list", "--no-daemon", "--spin-time", "0.2"],
        check=False,
        capture_output=True,
        text=True,
        timeout=3.0,
    )
    if result.returncode != 0:
        return set()
    return set(result.stdout.splitlines())


def _wait_for_ros_name(ros2, kind, name, timeout, launch_process):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        returncode = launch_process.poll()
        if returncode is not None:
            raise RuntimeError(
                f"Simulation process exited with status {returncode} while waiting for {name}"
            )
        try:
            if name in _ros_names(ros2, kind):
                return
        except subprocess.TimeoutExpired:
            pass
        time.sleep(0.2)
    raise RuntimeError(f"Timed out waiting for ROS {kind} {name}")


def _stop_process(process):
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGINT)
    try:
        process.wait(timeout=10.0)
        return
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5.0)


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-model", default="wx250s")
    parser.add_argument("--robot-name", default="wx250s")
    parser.add_argument("--sdk-robot-name", default="wx250s_sdk")
    parser.add_argument("--camera-model", type=Path, default=DEFAULT_CAMERA_MODEL)
    parser.add_argument("--image-topic", default=DEFAULT_IMAGE_TOPIC)
    parser.add_argument("--startup-timeout", type=_positive_float, default=30.0)
    parser.add_argument(
        "--gazebo-gui", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--rviz", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def run(args):
    ros2 = _require_ros2()
    plugin = _require_camera_plugin(ros2)
    environment = _gazebo_environment()
    if "/spawn_entity" in _ros_names(ros2, "service"):
        raise RuntimeError(
            "A Gazebo server is already running; stop it before starting this simulation"
        )
    camera_model = args.camera_model.resolve()
    if not camera_model.is_file():
        raise RuntimeError(f"Gazebo camera model does not exist: {camera_model}")
    if not DEFAULT_BRIDGE.is_file():
        raise RuntimeError(f"Gazebo command bridge does not exist: {DEFAULT_BRIDGE}")
    print(f"Using Gazebo camera plugin: {plugin}")

    launch_command = [
        ros2,
        "launch",
        "interbotix_xsarm_sim",
        "xsarm_gz_classic.launch.py",
        f"robot_model:={args.robot_model}",
        f"robot_name:={args.robot_name}",
        "hardware_type:=gz_classic",
        f"use_gazebo_gui:={str(args.gazebo_gui).lower()}",
        f"use_rviz:={str(args.rviz).lower()}",
    ]
    processes = []
    try:
        launch_process = subprocess.Popen(
            launch_command, start_new_session=True, env=environment
        )
        processes.append(launch_process)
        _wait_for_ros_name(
            ros2, "service", "/spawn_entity", args.startup_timeout, launch_process
        )
        spawn = subprocess.run(
            [
                ros2,
                "run",
                "gazebo_ros",
                "spawn_entity.py",
                "-entity",
                "berrypicker_fixed_camera",
                "-file",
                str(camera_model),
            ],
            check=False,
        )
        if spawn.returncode != 0:
            raise RuntimeError(
                f"Unable to spawn Gazebo camera; command exited with {spawn.returncode}"
            )
        _wait_for_ros_name(
            ros2, "topic", args.image_topic, args.startup_timeout, launch_process
        )
        _wait_for_ros_name(
            ros2,
            "topic",
            f"/{args.robot_name}/arm_controller/state",
            args.startup_timeout,
            launch_process,
        )

        sdk_command = [
            ros2,
            "launch",
            "interbotix_xsarm_control",
            "xsarm_control.launch.py",
            f"robot_model:={args.robot_model}",
            f"robot_name:={args.sdk_robot_name}",
            "use_sim:=true",
            "use_rviz:=false",
        ]
        sdk_process = subprocess.Popen(
            sdk_command, start_new_session=True, env=environment
        )
        processes.append(sdk_process)
        _wait_for_ros_name(
            ros2,
            "service",
            f"/{args.sdk_robot_name}/get_robot_info",
            args.startup_timeout,
            sdk_process,
        )

        bridge_process = subprocess.Popen(
            [
                sys.executable,
                str(DEFAULT_BRIDGE),
                "--sdk-robot-name",
                args.sdk_robot_name,
                "--gazebo-robot-name",
                args.robot_name,
            ],
            start_new_session=True,
            env=environment,
        )
        processes.append(bridge_process)
        _wait_for_ros_name(
            ros2,
            "node",
            "/berrypicker_widowx_gazebo_bridge",
            args.startup_timeout,
            bridge_process,
        )

        print(
            f"Gazebo camera ready on {args.image_topic}; "
            f"Interbotix controller ready as {args.sdk_robot_name}"
        )
        while True:
            for process in processes:
                returncode = process.poll()
                if returncode is not None:
                    raise RuntimeError(
                        f"Simulation process {process.args} exited with status {returncode}"
                    )
            time.sleep(0.5)
    finally:
        for process in reversed(processes):
            _stop_process(process)


def main():
    try:
        run(_arguments())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
