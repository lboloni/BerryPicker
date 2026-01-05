"""
abstract_controller.py

Abstract based class for the controller for the AL5D robot
"""
from abc import ABC, abstractmethod
from copy import copy

from robot.al5d_position_controller import RobotPosition, PositionController

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

class AbstractController:
    """An abstract class representing the ancestor of all classes that are used to perform high-level control the robot. Most of the classes here are something like remote control"""

    def __init__(self, robot_controller: PositionController = None, camera_controller = None, demonstration_recorder = None):
        """Initialize the XBox controller, possibly connected to a real robot."""
        self.robot_controller = robot_controller
        if robot_controller is not None:
            self.pos_current = self.robot_controller.get_position()
        else:
            # FIXME: this won't work without at least the exp here.
            self.pos_current = RobotPosition(None)
        self.pos_target = copy(self.pos_current)
        # the home position
        self.pos_home = copy(self.pos_current)
        # the interval at which we are polling the controller
        self.controller_interval = 0.1
        # the actual interval
        self.last_interval = self.controller_interval
        # the interval at which we are updating the robot
        self.robot_interval = 0.1
        # the velocities corresponding to maximum push
        self.v_distance = 1.0 # inch / second
        self.v_height = 1.0 # inch / second
        self.v_heading = 15.0 # angle degree / second
        self.v_gripper = 50.0 # percentage / second
        self.v_wrist_angle = 15.0 # angle degree / second
        self.v_wrist_rotation = 0.1 # angle degree / second
        self.camera_controller = camera_controller
        self.demonstration_recorder = demonstration_recorder

    def stop(self):
        """Stops the controller and all the other subcomponents"""
        if self.robot_controller:
            self.robot_controller.stop_robot()
        if self.demonstration_recorder:
            self.demonstration_recorder.stop()
        if self.camera_controller:
            self.camera_controller.stop()

    def update(self):
        """Updates the state of the various components"""
        logger.info(f"***AbstractController***: Update started")
        if self.camera_controller:
            self.camera_controller.update()
        if self.demonstration_recorder:
            self.demonstration_recorder.save()
        logger.info(f"***AbstractController***: Update done")
        
    def control_robot(self):
        """Control the robot by sending a command to move towards the target"""
        if self.robot_controller:
            logger.info(f"***AbstractController***: Control robot: move to position {self.pos_target}")
            self.robot_controller.move(self.pos_target)
            logger.info("***AbstractController***: Control robot done.")
