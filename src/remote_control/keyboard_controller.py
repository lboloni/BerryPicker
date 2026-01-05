"""
keyboard_controller.py

Keyboard based controller for the AL5D robot
"""
from robot.al5d_position_controller import RobotPosition, PositionController
from .abstract_controller import AbstractController

import time
# import serial 
from copy import copy

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

class KeyboardController(AbstractController):
    """
    A controller to control an AL5D robot with the keyboard. The assumption is that the keys are read out by the OpenCV camera controller.

    These are the keys for typical FPS games
    
    WSAD for move.
    Ctrl for crouch or sneak.
    Left click for primary attack, Right click for secondary attack.
    Shift for run.
    E or F for "activate"
    "R" for reload.

    The keys to be used here:
    W 
    S
    A 
    D 
    up 82
    down 84
    left  81
    right 83
    Q
    pgup 85
    pgdown 86
    left-shift  225
    right-shift  226
    left-alt 233
    right-alt 234
    """

    def __init__(self, exp, robot_controller: PositionController = None, camera_controller = None, demonstration_recorder = None):
        super().__init__(robot_controller, camera_controller, demonstration_recorder)
        self.exp = exp
    

    def control(self):
        """The main control loop"""
        self.exit_control = False
        while True:
            start_time = time.time() 
            key = self.camera_controller.update() 
            self.process_key(key)
            # if we are exiting, call the stopping of the robot, of the recording and the vision
            if self.exit_control:
                self.stop()
                break;
            print(key)
            self.control_robot()
            self.update()
            end_time = time.time() 
            execution_time = end_time - start_time 
            self.last_interval = execution_time
            time_to_sleep = max(0.0, self.controller_interval - execution_time) 
            time.sleep(time_to_sleep) 

    def process_key(self, key):
        """Sets the target location based on the key pressed"""
        keycode = key & 0xFF
        # distance: s and a 
        delta_distance = 0
        if keycode == ord(self.exp["forward_ord"]): # forward
            delta_distance = self.v_distance * self.last_interval
        if keycode == ord(self.exp["backward_ord"]): # backward
            delta_distance = - self.v_distance * self.last_interval
        # height: w and z
        delta_height = 0
        if keycode == ord(self.exp["up_ord"]): # up
            delta_height = self.v_height * self.last_interval
        if keycode == ord(self.exp["down_ord"]): # down
            delta_height = - self.v_height * self.last_interval
        # rotation: left-right FIXME: maybe this should go on pgup 85/86
        delta_heading = 0
        if keycode == self.exp["heading_left_kc"]: # left -> rotate-left
            delta_heading = self.v_heading * self.last_interval
        if keycode == self.exp["heading_right_kc"]: # right -> rotate-right
            delta_heading = - self.v_heading * self.last_interval

        # wrist angle: pg-up pg-down FIXME: maybe this should go on up down
        delta_wrist_angle = 0
        if keycode == self.exp["wrist_up_kc"]: # pgup - wrist angle up
            delta_wrist_angle = self.v_wrist_angle * self.last_interval 
        if keycode == self.exp["wrist_down_kc"]: # pgdn - wrist angle down
            delta_wrist_angle = - self.v_wrist_angle * self.last_interval 

        # wrist rotation: left-right
        delta_wrist_rotation = 0
        if keycode == self.exp["wrist_left_kc"]: # left --> wrist-rotate-left
            delta_wrist_rotation = self.v_wrist_rotation * self.last_interval 
        if keycode == self.exp["wrist_right_kc"]: # right --> write-rotate-right
            delta_wrist_rotation = - self.v_wrist_rotation * self.last_interval 

        # gripper open-close: right alt / shift 226 / 234
        delta_gripper = 0
        # the right alt/shift immediately closes and opens the gripper
        if keycode == self.exp["close_gripper_kc"]:
            delta_gripper = 100
        if keycode == self.exp["open_gripper_kc"]:
            delta_gripper = -100
        # the left alt/shift opens/closes it gradually 225/233
        if keycode == self.exp["closer_gripper_kc"]:
            delta_gripper += self.v_gripper * self.last_interval
        if keycode == self.exp["wider_gripper_kc"]:
            delta_gripper += - self.v_gripper * self.last_interval

        # square aka x - exit control
        if keycode == ord(self.exp["exit_control_ord"]):
            self.exit_control = True
            return
        # home h  
        if keycode == ord(self.exp["home_ord"]):
            self.pos_target = copy(self.pos_home)
            return
        # applying the changes 
        self.pos_target["distance"] += delta_distance
        self.pos_target["height"] += delta_height
        self.pos_target["heading"] += delta_heading
        self.pos_target["wrist_angle"] += delta_wrist_angle
        self.pos_target["wrist_rotation"] += delta_wrist_rotation
        self.pos_target["gripper"] += delta_gripper
        # FIXME: applying a safety reset which prevents us going out of range
        ok = RobotPosition.limit(self.robot_controller.exp, self.pos_target)
        if not ok:
            logger.warning(f"DANGER! exceeded range! {self.pos_target}")
        logger.warning(f"Target: {self.pos_target}")
