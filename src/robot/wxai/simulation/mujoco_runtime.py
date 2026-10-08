"""One MuJoCo simulation of the WidowX AI scene, shared by the robot and the cameras."""

import pathlib

import cv2
import mujoco

from robot.wxai.kinematics import model_path


SIMULATION_DIR = pathlib.Path(__file__).resolve().parent
_RUNTIMES = {}


class MujocoRuntime:
    """The model and data of a scene with the robot attached at robot_mount."""

    def __init__(self, exp):
        scene = mujoco.MjSpec.from_file(str(SIMULATION_DIR / exp["scene"]))
        robot = mujoco.MjSpec.from_file(str(model_path(exp)))
        scene.frame("robot_mount").attach_body(robot.body("base_link"), "", "")
        scene.option.timestep = exp["physics_timestep"]
        self.model = scene.compile()
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.renderers = {}

    def step(self, dt):
        """Advance the physics by dt seconds."""
        for _ in range(max(1, round(dt / self.model.opt.timestep))):
            mujoco.mj_step(self.model, self.data)

    def render(self, camera, width, height):
        """The BGR image of a named camera, as the real camera controllers return it."""
        if (width, height) not in self.renderers:
            self.renderers[(width, height)] = mujoco.Renderer(self.model, height, width)
        renderer = self.renderers[(width, height)]
        renderer.update_scene(self.data, camera=camera)
        return cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)

    def close(self):
        for renderer in self.renderers.values():
            renderer.close()
        self.renderers = {}


def create_runtime(exp):
    """Create the runtime of the scene of a robot exprun; one per scene and process."""
    if exp["scene"] in _RUNTIMES:
        raise RuntimeError(f"A MuJoCo runtime for {exp['scene']} already exists")
    _RUNTIMES[exp["scene"]] = MujocoRuntime(exp)
    return _RUNTIMES[exp["scene"]]


def get_runtime(scene):
    """The runtime of a scene, created by the robot; the cameras attach to it."""
    if scene not in _RUNTIMES:
        raise RuntimeError(
            f"No MuJoCo runtime for {scene}: the MuJoCo WidowX AI must be created before its cameras")
    return _RUNTIMES[scene]


def release_runtime(scene):
    _RUNTIMES.pop(scene).close()
