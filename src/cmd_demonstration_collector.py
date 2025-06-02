from pathlib import Path
from datetime import datetime
import cv2
import json
from copy import copy
import random
# import sys

from helper import ui_choose_task
from robotcontrol.gamepad_controller import GamepadController
from robotcontrol.keyboard_controller import KeyboardController
from robotcontrol.program_controller import ProgramController
from robot.al5d_position_controller import PositionController, RobotPosition
from camera.camera_controller import CameraController

from exp_run_config import Config
Config.PROJECTNAME = "BerryPicker"

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class DemonstrationRecorder:
    """
    FIXME: obsolete. see the new code in demonstration...
    
    Record demonstration data collected from various controllers, sensors etc."""

    def __init__(self, controller, robot_controller: PositionController, camera_controller: CameraController, task_name = "unknown", save_dir = None):
        self.save_dir = save_dir
        self.remote_control = controller
        self.robot_controller = robot_controller
        self.camera_controller = camera_controller
        self.counter = 0
        self.task_name = task_name
        pass

    def save(self):
        """
        Write the data from the various sources with a common prefix
        """
        save_prefix = f"{self.counter:05d}"
        self.counter += 1
        data = {}
        data["rc-position-target"] = copy(self.remote_control.pos_target.values)
        data["rc-angle-target"] = self.robot_controller.angle_controller.as_dict()
        data["rc-pulse-target"] = self.robot_controller.pulse_controller.as_dict()
        data["reward"] = 0.0 # placeholder for the reward
        data["annotation"] = "" # placeholder for annotation

        json_file = Path(self.save_dir, f"{save_prefix}.json") 
        logger.info(f"Saving into json file {json_file}")
        with open(json_file,"w") as f:
            json.dump(data, f)
        # save the captured images
        if self.camera_controller is not None:
            for index in self.camera_controller.images:
                filename = Path(self.save_dir, f"{save_prefix}_{index}.jpg")
                cv2.imwrite(str(filename), self.camera_controller.images[index])

    def stop(self):
        """FIXME: This might be used in the future to save final stuff etc."""
        pass


def main():
    print("======= Demonstration collector =======")
    # find the demonstration path
    _, task_dir = ui_choose_task(offer_task_creation=True)

    demoname = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
    demo_dir = Path(task_dir, demoname)
    demo_dir.mkdir(parents=False, exist_ok=False)
    print(f"Ok, demonstration with go into {demo_dir}")

    # the demonstration dictionary
    description = {"name": demoname, "task": task_dir.name}
    description["name"] = demoname
    description["task"] = task_dir.name    
    description["trim-from"] = 1 # counter number from where we need to take the data, default from beginning
    description["trim-to"] = -1 # counter number to where we need to take the demonstration, default -1 means to the end
    description["success"] = False # was the demonstration successful
    description["quality"] = 0.0 # numerical quality metric of the demonstration, 0.0 total failure, 1.0 perfect success
    description["text-annotation"] = "" # text annotation

    # Write the overall file
    file_overall = Path(demo_dir, "_demonstration.json")
    with open(file_overall, "w") as f:
        #str_description = json.dumps(description)
        #f.write(str_description)
        json.dump(description, f)

    # start the demonstration
        
    print("====== Starting the demonstration ========")

    # the robot position controller
    robot_controller = PositionController(Config()["robot"]["usb_port"]) 

    img_size = Config()["robot"]["saved_image_size"]
    # (256, 256) # was (128, 128)
    camera_controller = CameraController(devices = Config()["robot"]["active_camera_list"], img_size = img_size)
    camera_controller.visualize = True
    #cameratracker = None
    # the XBox controller - we are using the control loop from this one
    # controller = "program"
    controller = "xbox"
    if controller == "xbox":
        gamepad_controller = GamepadController(
            robot_controller=robot_controller, camera_controller=camera_controller)
        demo_recorder = DemonstrationRecorder(
            camera_controller=camera_controller, controller=gamepad_controller, robot_controller=
                                    robot_controller, save_dir=demo_dir, task_name=task_dir.name)
        # dr = None
        gamepad_controller.demonstration_recorder = demo_recorder
        gamepad_controller.control()
        print("====== Demonstration terminated and recorded successfully, bye. ======")
    if controller == "kb":
        kb_controller = KeyboardController(
            robot_controller=robot_controller, camera_controller=camera_controller)
        demo_recorder = DemonstrationRecorder(camera_controller=camera_controller, controller=kb_controller, robot_controller=
                                    robot_controller, save_dir=demo_dir, task_name=task_dir.name)
        # dr = None
        kb_controller.demonstration_recorder = demo_recorder
        kb_controller.control()
        print("====== Demonstration terminated and recorded successfully, bye. ======")

    if controller == "program":
        program_controller = ProgramController(
            robot_controller=robot_controller, camera_controller=camera_controller)
        
        # create wpcount 
        waypoints = []
        wpcount = 10
        while True:
            norm = [0] * 6
            for df in range(6):
                norm[df] = random.random()                
            rp = RobotPosition.from_normalized_vector(norm)
            if RobotPosition.limit(rp):
                print(f"added waypoint {rp}")
                waypoints.append(rp)
            if len(waypoints) >= wpcount:
                break

        program_controller.waypoints = waypoints
        program_controller.interactive_confirm = False


        demo_recorder = DemonstrationRecorder(camera_controller=camera_controller, controller=program_controller, robot_controller=
                                    robot_controller, save_dir=demo_dir, task_name=task_dir.name)

        program_controller.demonstration_recorder = demo_recorder
        program_controller.control()

if __name__ == "__main__":
    main()