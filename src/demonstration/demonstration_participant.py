"""Uniform participants used while collecting a demonstration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import copy
from dataclasses import dataclass, field
import math

from exp_run_config import Config
from robot.al5d import PositionController, SimulatedPositionController
from robot.widowx import (
    PositionController as WidowXPositionController,
    SimulatedPositionController as SimulatedWidowXPositionController,
    WidowXCommand,
    WidowXPose,
)


@dataclass
class DemonstrationSample:
    """One participant's contribution to a recorded timestep."""

    action: dict | None = None
    telemetry: dict = field(default_factory=dict)
    images: dict = field(default_factory=dict)
    annotations: dict = field(default_factory=dict)


class DemonstrationContext:
    """Mutable state shared by participants during one collection run."""

    def __init__(self):
        self.commands = {}
        self.values = {}
        self.stop_requested = False

    def request_stop(self):
        self.stop_requested = True


class DemonstrationParticipant(ABC):
    """Common lifecycle for an input, robot, or sensor in a collection run."""

    def __init__(self, name, spec, exp):
        self.name = name
        self.spec = spec
        self.exp = exp

    def bind(self, participants):
        """Resolve references to other named participants after construction."""

    def start(self, context):
        """Acquire resources before the collection loop starts."""

    @abstractmethod
    def update(self, context, dt):
        """Advance this participant by one collection tick."""

    def sample(self, context):
        """Return this participant's recordable output for the current tick."""
        return DemonstrationSample()

    def stop(self, context):
        """Release resources after collection has stopped."""


class AL5DParticipant(DemonstrationParticipant):
    """Apply an AL5D target and expose the resulting demonstration action."""

    def __init__(self, name, spec, exp, simulated):
        super().__init__(name, spec, exp)
        self.controller = (
            SimulatedPositionController(exp) if simulated else PositionController(exp)
        )
        self.simulated = simulated
        self.command_name = spec["command"]
        self.target = self.controller.get_position()

    def start(self, context):
        if not self.simulated:
            self.controller.start_robot()

    def update(self, context, dt):
        if self.command_name not in context.commands:
            return
        self.target = copy(context.commands[self.command_name])
        self.controller.move(self.target)

    def sample(self, context):
        data = {"rc-position-target": copy(self.target.values)}
        if isinstance(self.controller, PositionController):
            data["rc-angle-target"] = self.controller.angle_controller.as_dict()
            data["rc-pulse-target"] = self.controller.pulse_controller.as_dict()
        return DemonstrationSample(
            action=data,
            telemetry={"position": copy(self.controller.get_position().values)},
        )

    def stop(self, context):
        if not self.simulated:
            self.controller.stop_robot()


class WidowXParticipant(DemonstrationParticipant):
    """Apply native WidowX commands and record target and observed robot state."""

    def __init__(self, name, spec, exp, simulated):
        super().__init__(name, spec, exp)
        self.controller = (
            SimulatedWidowXPositionController(exp)
            if simulated
            else WidowXPositionController(exp)
        )
        self.command_name = spec["command"]
        self.target = WidowXCommand(WidowXPose(exp))

    def start(self, context):
        self.controller.start_robot()
        self.target = WidowXCommand(self.controller.get_target())

    def update(self, context, dt):
        if self.command_name not in context.commands:
            return
        command = context.commands[self.command_name]
        if isinstance(command, WidowXPose):
            command = WidowXCommand(command)
        if not isinstance(command, WidowXCommand):
            raise TypeError(
                f"WidowX participant {self.name} received an invalid command"
            )
        self.target = copy(command)
        self.controller.move(
            self.target,
            moving_time=self.spec.get("moving_time", self.exp.get("moving_time")),
            blocking=self.spec.get("blocking", False),
        )

    def sample(self, context):
        return DemonstrationSample(
            action={"widowx-command": self.target.as_dict()},
            telemetry=self.controller.get_state(),
        )

    def stop(self, context):
        self.controller.stop_robot()


class FixedCameraParticipant(DemonstrationParticipant):
    """Capture all configured fixed cameras once per collection tick."""

    def __init__(self, name, spec, exp):
        super().__init__(name, spec, exp)
        self.controller = None

    def start(self, context):
        from camera.camera_controller import CameraController

        self.controller = CameraController(self.exp)
        self.controller.visualize = self.exp.get("visualize", True)

    def update(self, context, dt):
        context.values["key"] = self.controller.update()

    def sample(self, context):
        return DemonstrationSample(images=dict(self.controller.images))

    def stop(self, context):
        self.controller.stop()


class _AL5DLeaderParticipant(DemonstrationParticipant):
    """Base class for participants that produce targets for an AL5D participant."""

    def __init__(self, name, spec, exp):
        super().__init__(name, spec, exp)
        self.command_name = spec["emits"]
        self.target_robot_name = spec["target_robot"]
        self.target_robot = None

    def bind(self, participants):
        try:
            self.target_robot = participants[self.target_robot_name]
        except KeyError as error:
            raise ValueError(
                f"Participant {self.name} refers to unknown target robot "
                f"{self.target_robot_name}"
            ) from error
        if not isinstance(self.target_robot, AL5DParticipant):
            raise TypeError(
                f"Participant {self.name} target {self.target_robot_name} is not an AL5D participant"
            )


class XboxLeaderParticipant(_AL5DLeaderParticipant):
    """Use an XBox controller to emit AL5D targets."""

    def __init__(self, name, spec, exp):
        super().__init__(name, spec, exp)
        self.controller = None
        self.resource = None
        self.joystick = None

    def bind(self, participants):
        super().bind(participants)
        try:
            from remote_control.gamepad_controller import GamepadController
        except ModuleNotFoundError as error:
            raise RuntimeError("XBox collection requires the approxeng input package") from error
        self.controller = GamepadController(self.exp, self.target_robot.controller)

    def start(self, context):
        from approxeng.input.selectbinder import ControllerResource

        self.resource = ControllerResource()
        self.joystick = self.resource.__enter__()
        if self.joystick is None:
            raise RuntimeError("Unable to acquire configured XBox controller")

    def update(self, context, dt):
        if not self.joystick.connected:
            context.request_stop()
            return
        self.controller.last_interval = dt
        self.controller.poll_controller(self.joystick)
        if self.controller.exit_control:
            context.request_stop()
            return
        context.commands[self.command_name] = copy(self.controller.pos_target)

    def sample(self, context):
        return DemonstrationSample(
            telemetry={"target": copy(self.controller.pos_target.values)}
        )

    def stop(self, context):
        if self.resource is not None:
            self.resource.__exit__(None, None, None)
            self.resource = None
            self.joystick = None


class WidowXXboxLeaderParticipant(DemonstrationParticipant):
    """Use an Xbox-style controller to emit native WidowX commands."""

    def __init__(self, name, spec, exp):
        super().__init__(name, spec, exp)
        self.command_name = spec["emits"]
        self.target_robot_name = spec["target_robot"]
        self.target_robot = None
        self.controller = None
        self.resource = None
        self.joystick = None

    def bind(self, participants):
        try:
            self.target_robot = participants[self.target_robot_name]
        except KeyError as error:
            raise ValueError(
                f"Participant {self.name} refers to unknown target robot "
                f"{self.target_robot_name}"
            ) from error
        if not isinstance(self.target_robot, WidowXParticipant):
            raise TypeError(
                f"Participant {self.name} target {self.target_robot_name} "
                "is not a native WidowX participant"
            )
        from remote_control.widowx_gamepad_controller import (
            WidowXGamepadController,
        )

        self.controller = WidowXGamepadController(
            self.exp, self.target_robot.controller
        )

    def start(self, context):
        if self.resource is not None:
            raise RuntimeError("WidowX Xbox controller is already acquired")
        try:
            from approxeng.input.selectbinder import (
                ControllerNotFoundError,
                ControllerResource,
            )
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "WidowX Xbox collection requires the approxeng input package"
            ) from error

        self.resource = ControllerResource()
        try:
            self.joystick = self.resource.__enter__()
        except ControllerNotFoundError as error:
            self.resource = None
            raise RuntimeError("Unable to acquire configured Xbox controller") from error
        if self.joystick is None:
            self.resource.__exit__(None, None, None)
            self.resource = None
            raise RuntimeError("Unable to acquire configured Xbox controller")

    def update(self, context, dt):
        if not self.joystick.connected:
            context.request_stop()
            return
        if not self.controller.synchronized:
            self.controller.synchronize(self.target_robot.controller.get_position())
        command = self.controller.poll_controller(self.joystick, dt)
        if self.controller.exit_control:
            context.request_stop()
            return
        context.commands[self.command_name] = copy(command)

    def sample(self, context):
        return DemonstrationSample(telemetry=self.controller.get_state())

    def stop(self, context):
        if self.resource is not None:
            self.resource.__exit__(None, None, None)
            self.resource = None
        self.joystick = None


class KeyboardLeaderParticipant(_AL5DLeaderParticipant):
    """Use the key captured by the fixed-camera participant to emit AL5D targets."""

    def bind(self, participants):
        super().bind(participants)
        from remote_control.keyboard_controller import KeyboardController

        self.controller = KeyboardController(self.exp, self.target_robot.controller)

    def update(self, context, dt):
        self.controller.last_interval = dt
        self.controller.process_key(context.values.get("key", -1))
        if self.controller.exit_control:
            context.request_stop()
            return
        context.commands[self.command_name] = copy(self.controller.pos_target)

    def sample(self, context):
        return DemonstrationSample(
            telemetry={"target": copy(self.controller.pos_target.values)}
        )


class AutoMoveLeaderParticipant(_AL5DLeaderParticipant):
    """Emit the waypoint targets generated by the existing AutoMove controller."""

    def bind(self, participants):
        super().bind(participants)
        from remote_control.automove_controller import AutoMoveController

        self.controller = AutoMoveController(self.exp, self.target_robot.controller)
        self.controller.generate_waypoints()
        self.autonomous_countdown = 0

    def update(self, context, dt):
        self.controller.last_interval = dt
        self.controller.pos_current = self.target_robot.controller.get_position()
        self.controller.max_timesteps -= 1
        target = self.controller.next_pos()
        if target is None or self.controller.max_timesteps <= 0:
            context.request_stop()
            return
        self.autonomous_countdown -= 1
        if self.controller.interactive_confirm and self.autonomous_countdown <= 0:
            proceed = input(f"Proposed next target: {target}. Proceed? [stop/y/<number>]")
            if proceed == "stop":
                context.request_stop()
                return
            if proceed.isdigit():
                self.autonomous_countdown = int(proceed)
        context.commands[self.command_name] = copy(target)

    def sample(self, context):
        return DemonstrationSample(
            telemetry={
                "target": copy(self.controller.pos_target.values),
                "automove_type": self.controller.automove_type,
                "random_seed": self.controller.random_seed,
            }
        )


class WidowXAutoMoveLeaderParticipant(DemonstrationParticipant):
    """Emit reachable native-pose commands for a WidowX robot participant."""

    def __init__(self, name, spec, exp):
        super().__init__(name, spec, exp)
        self.command_name = spec["emits"]
        self.target_robot_name = spec["target_robot"]
        self.target_robot = None
        self.controller = None
        self.autonomous_countdown = 0

    def bind(self, participants):
        try:
            self.target_robot = participants[self.target_robot_name]
        except KeyError as error:
            raise ValueError(
                f"Participant {self.name} refers to unknown target robot "
                f"{self.target_robot_name}"
            ) from error
        if not isinstance(self.target_robot, WidowXParticipant):
            raise TypeError(
                f"Participant {self.name} target {self.target_robot_name} "
                "is not a native WidowX participant"
            )
        from remote_control.widowx_automove_controller import WidowXAutoMoveController

        self.controller = WidowXAutoMoveController(
            self.exp, self.target_robot.controller
        )
        self.controller.generate_waypoints()

    def update(self, context, dt):
        if self.controller.max_timesteps <= 0:
            context.request_stop()
            return
        command = self.controller.next_command(dt)
        if command is None:
            context.request_stop()
            return
        self.autonomous_countdown -= 1
        if self.controller.interactive_confirm and self.autonomous_countdown <= 0:
            proceed = input(
                f"Proposed next WidowX target: {command.pose}. "
                "Proceed? [stop/y/<number>]"
            )
            if proceed == "stop":
                context.request_stop()
                return
            if proceed.isdigit():
                self.autonomous_countdown = int(proceed)
        context.commands[self.command_name] = copy(command)
        self.controller.max_timesteps -= 1

    def sample(self, context):
        return DemonstrationSample(telemetry={
            "target": self.controller.pos_target.as_dict(),
            "automove_type": self.controller.automove_type,
            "random_seed": self.controller.random_seed,
            "remaining_waypoints": len(self.controller.waypoints),
        })


class WidowXLeaderParticipant(_AL5DLeaderParticipant):
    """Backdriven WidowX leader that emits an AL5D target."""

    def __init__(self, name, spec, exp):
        super().__init__(name, spec, exp)
        self.bot = None
        self.robot_shutdown = None

    def start(self, context):
        from interbotix_common_modules.common_robot.robot import robot_startup, robot_shutdown
        from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS

        self.bot = InterbotixManipulatorXS(
            robot_model=self.exp.get("robot_model", "wx250s"),
            group_name=self.exp.get("group_name", "arm"),
            gripper_name=self.exp.get("gripper_name", "gripper"),
        )
        self.robot_shutdown = robot_shutdown
        robot_startup()

    def update(self, context, dt):
        target = self.target_robot.controller.get_position()
        joints = self.bot.arm.get_joint_positions()
        ee_pose = self.bot.arm.get_ee_pose()
        target["heading"] = joints[0] * -50.0
        target["height"] = 2.0 + 10.0 * ee_pose[2, 3]
        target["distance"] = 2.0 + 10.0 * math.sqrt(
            ee_pose[0, 3] ** 2 + ee_pose[1, 3] ** 2
        )
        target["wrist_angle"] = -45.0 - 15.0 * joints[4]
        target["wrist_rotation"] = 75.0 + 40.0 * joints[5]

        keycode = context.values.get("key", -1) & 0xFF
        if keycode == ord(self.exp["exit_control_ord"]):
            context.request_stop()
            return
        if keycode == ord(self.exp["home_ord"]):
            target = copy(self.target_robot.controller.get_position())
        if keycode == self.exp["close_gripper_kc"]:
            target["gripper"] += 100
        if keycode == self.exp["open_gripper_kc"]:
            target["gripper"] -= 100
        if keycode == self.exp["closer_gripper_kc"]:
            target["gripper"] += 50.0 * dt
        if keycode == self.exp["wider_gripper_kc"]:
            target["gripper"] -= 50.0 * dt
        context.commands[self.command_name] = target

    def sample(self, context):
        return DemonstrationSample(
            telemetry={"target": copy(context.commands[self.command_name].values)}
        )

    def stop(self, context):
        if self.bot is not None:
            self.robot_shutdown(self.bot.core.get_node())
            self.bot = None


class WidowXObserverParticipant(DemonstrationParticipant):
    """Record a WidowX state without using it as the demonstration leader."""

    def __init__(self, name, spec, exp):
        super().__init__(name, spec, exp)
        self.bot = None
        self.robot_shutdown = None

    def start(self, context):
        from interbotix_common_modules.common_robot.robot import robot_startup, robot_shutdown
        from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS

        self.bot = InterbotixManipulatorXS(
            robot_model=self.exp.get("robot_model", "wx250s"),
            group_name=self.exp.get("group_name", "arm"),
            gripper_name=self.exp.get("gripper_name", "gripper"),
        )
        self.robot_shutdown = robot_shutdown
        robot_startup()

    def update(self, context, dt):
        pass

    def sample(self, context):
        return DemonstrationSample(
            telemetry={
                "joint_positions": self.bot.arm.get_joint_positions().tolist(),
                "ee_pose": self.bot.arm.get_ee_pose().tolist(),
            }
        )

    def stop(self, context):
        if self.bot is not None:
            self.robot_shutdown(self.bot.core.get_node())
            self.bot = None


class MobileCameraParticipant(DemonstrationParticipant):
    """Move a mobile camera after the controlled robot has moved."""

    def __init__(self, name, spec, exp):
        super().__init__(name, spec, exp)
        self.target_robot_name = spec["target_robot"]
        self.target_robot = None
        self.controller = None

    def bind(self, participants):
        try:
            self.target_robot = participants[self.target_robot_name]
        except KeyError as error:
            raise ValueError(
                f"Participant {self.name} refers to unknown target robot "
                f"{self.target_robot_name}"
            ) from error
        if not isinstance(self.target_robot, AL5DParticipant):
            raise TypeError(
                f"Participant {self.name} target {self.target_robot_name} is not an AL5D participant"
            )

    def start(self, context):
        from mobile_camera.mobile_camera_controller import MobileCamera

        self.controller = MobileCamera(self.exp, self.target_robot.controller)

    def update(self, context, dt):
        self.controller.update()

    def stop(self, context):
        self.controller.stop()


def _load_participant_experiment(spec, binding):
    exp_name = spec.get("exp", binding.get("exp"))
    run_name = spec.get("run", binding.get("run"))
    if exp_name is None or run_name is None:
        raise ValueError(f"Participant {spec['name']} must define exp/run in its spec or binding")
    return Config().get_experiment(exp_name, run_name)


def create_participants(collection_exp, machine_exp):
    """Create and validate participants from a collection recipe and machine profile."""
    try:
        bindings = machine_exp["bindings"]
        specs = collection_exp["participants"]
    except KeyError as error:
        raise ValueError("Collection recipe and machine profile must define participants/bindings") from error

    factories = {
        "al5d_hardware": lambda name, spec, exp: AL5DParticipant(name, spec, exp, False),
        "al5d_simulated": lambda name, spec, exp: AL5DParticipant(name, spec, exp, True),
        "fixed_cameras": FixedCameraParticipant,
        "xbox_leader": XboxLeaderParticipant,
        "widowx_xbox_leader": WidowXXboxLeaderParticipant,
        "keyboard_leader": KeyboardLeaderParticipant,
        "automove_leader": AutoMoveLeaderParticipant,
        "widowx_hardware": lambda name, spec, exp: WidowXParticipant(
            name, spec, exp, False
        ),
        "widowx_simulated": lambda name, spec, exp: WidowXParticipant(
            name, spec, exp, True
        ),
        "widowx_automove_leader": WidowXAutoMoveLeaderParticipant,
        "widowx_leader": WidowXLeaderParticipant,
        "widowx_observer": WidowXObserverParticipant,
        "mobile_camera": MobileCameraParticipant,
    }
    participants = {}
    resources = set()
    for spec in specs:
        try:
            name = spec["name"]
            binding_name = spec["binding"]
        except KeyError as error:
            raise ValueError(f"Invalid participant specification: {spec}") from error
        if name in participants:
            raise ValueError(f"Duplicate participant name: {name}")
        if binding_name not in bindings:
            raise ValueError(f"Participant {name} requests unknown machine binding {binding_name}")
        binding = bindings[binding_name]
        if not binding.get("available", False):
            raise RuntimeError(f"Machine binding {binding_name} is unavailable")
        factory_name = binding.get("factory")
        if factory_name not in factories:
            raise ValueError(f"Unsupported participant factory {factory_name}")
        for resource in binding.get("resources", []):
            if resource in resources:
                raise RuntimeError(f"Machine resource {resource} is assigned more than once")
            resources.add(resource)
        binding_exp = _load_participant_experiment(spec, binding)
        if factory_name in ("automove_leader", "widowx_automove_leader"):
            tick_interval = collection_exp.get("tick_interval", 0.1)
            if binding_exp["robot_interval"] != tick_interval:
                raise ValueError(
                    f"Automove robot_interval {binding_exp['robot_interval']} must equal "
                    f"collection tick_interval {tick_interval}"
                )
        participants[name] = factories[factory_name](name, spec, binding_exp)
    for participant in participants.values():
        participant.bind(participants)
    return list(participants.values())
