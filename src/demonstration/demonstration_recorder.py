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
    """Record demonstration data collected from various controllers, sensors etc.
    
    FIXME: this will need to be changed to directly create a demonstration object
    """

    def __init__(self, demonstration, remote_control, robot_controller: PositionController, camera_controller: CameraController, save_dir = None):
        self.demonstration = demonstration
        self.save_dir = save_dir
        self.remote_control = remote_control
        self.robot_controller = robot_controller
        self.camera_controller = camera_controller
        self.counter = 0
        pass

    def save(self):
        """
        Write the data from the various sources with a common prefix
        """
        save_prefix = f"{self.counter:05d}"
        # writing the robot data if the robot is available
        if self.robot_controller:
            assert len(self.demonstration.action) == self.counter
            assert self.demonstration.metadata["maxsteps"] == self.counter
            data = {}
            data["rc-position-target"] = copy.copy(self.remote_control.os_target.values)
            data["rc-angle-target"] = self.robot_controller.angle_controller.as_dict()
            data["rc-pulse-target"] = self.robot_controller.pulse_controller.as_dict()
            self.demonstration.actions.append(data)
            self.demonstration.annotations.append({})
            #data = {}
            #data["rc-position-target"] = copy.copy(self.remote_control.pos_target.values)
            #data["rc-angle-target"] = self.robot_controller.angle_controller.as_dict()
            #data["rc-pulse-target"] = self.robot_controller.pulse_controller.as_dict()
            #data["reward"] = 0.0 # placeholder for the reward
            #data["annotation"] = "" # placeholder for annotation

            #json_file = pathlib.Path(self.save_dir, f"{save_prefix}.json") 
            #print(f"Saving into json file {json_file}")
            #with open(json_file,"w") as f:
            #    json.dump(data, f)

        # save the captured images
        if self.camera_controller is not None:
            for index in self.camera_controller.images:
                filename = pathlib.Path(self.save_dir, f"{save_prefix}_{index}.jpg")
                cv2.imwrite(str(filename), self.camera_controller.images[index])
        self.counter += 1
        self.demonstration.metadata["maxsteps"] = self.counter

    def stop(self):
        self.demonstration.save_metadata()

