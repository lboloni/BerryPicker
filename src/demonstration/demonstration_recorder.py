"""
demonstration_recorder.py

Code that helps in recording ongoing demonstrations

"""
import sys
sys.path.append("..")



# from robotcontrol.gamepad_controller import GamepadController
# from robotcontrol.keyboard_controller import KeyboardController
# from robotcontrol.program_controller import ProgramController
from robot.al5d_position_controller import PositionController, RobotPosition
from camera.camera_controller import CameraController
import pathlib
import cv2
import json
import copy


class DemonstrationRecorder:
    """Record demonstration data collected from various controllers, sensors etc."""

    def __init__(self, controller, robot_controller: PositionController, camera_controller: CameraController, task_name = "unknown", save_dir = None):
        self.save_dir = save_dir
        self.remote_control = controller
        self.robot_controller = robot_controller
        self.camera_controller = camera_controller
        self.counter = 1
        self.task_name = task_name
        pass

    def save(self):
        """
        Write the data from the various sources with a common prefix
        """
        save_prefix = f"{self.counter:05d}"
        self.counter += 1
        data = {}
        data["rc-position-target"] = copy.copy(self.remote_control.pos_target.values)
        data["rc-angle-target"] = self.robot_controller.angle_controller.as_dict()
        data["rc-pulse-target"] = self.robot_controller.pulse_controller.as_dict()
        data["reward"] = 0.0 # placeholder for the reward
        data["annotation"] = "" # placeholder for annotation

        json_file = pathlib.Path(self.save_dir, f"{save_prefix}.json") 
        print(f"Saving into json file {json_file}")
        with open(json_file,"w") as f:
            json.dump(data, f)
        # save the captured images
        if self.camera_controller is not None:
            for index in self.camera_controller.images:
                filename = pathlib.Path(self.save_dir, f"{save_prefix}_{index}.jpg")
                cv2.imwrite(str(filename), self.camera_controller.images[index])

    def stop(self):
        """FIXME: This might be used in the future to save final stuff etc."""
        pass
