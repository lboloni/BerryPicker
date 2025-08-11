"""
program_controller.py

Preprogrammed controller for the AL5D robot
"""
from robot.al5d_position_controller import RobotPosition, PositionController
from .abstract_controller import AbstractController

import time
# import serial 
from copy import copy

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


def move_towards(current, target, max_velocity):
    if abs(target - current) <= max_velocity:
        # If the distance to the target is less than the max velocity, snap to target
        return target
    elif target > current:
        # Move up towards the target
        return current + max_velocity
    else:
        # Move down towards the target
        return current - max_velocity

def move_position_towards(current: RobotPosition, 
    target: RobotPosition, ctrl, exp):
    """Move a position towards the target with specific velocities"""
    rv = RobotPosition(exp)
    rv["height"] = move_towards(current["height"], target["height"], 
                             ctrl.v_height * ctrl.robot_interval)
    rv["distance"] = move_towards(current["distance"], target["distance"], 
                               ctrl.v_distance * ctrl.robot_interval)
    rv["heading"] = move_towards(current["heading"], target["heading"], 
                             ctrl.v_heading * ctrl.robot_interval)
    rv["wrist_angle"] = move_towards(
        current["wrist_angle"], target["wrist_angle"],
        ctrl.v_wrist_angle * ctrl.robot_interval)
    rv["wrist_rotation"] = move_towards(current["wrist_rotation"], 
                                     target["wrist_rotation"], 
                                     ctrl.v_wrist_rotation * ctrl.robot_interval)
    rv["gripper"] = move_towards(current["gripper"], target["gripper"],
                              ctrl.v_gripper * ctrl.robot_interval)
    return rv


class ProgramController(AbstractController):
    """A programmed robot controller that works by reaching a set of waypoints with the robot. """

    def __init__(self, robot_controller: PositionController = None, camera_controller = None, demonstration_recorder = None):
        super().__init__(robot_controller, camera_controller, demonstration_recorder)
        self.max_timesteps = 1000
        self.interactive_confirm = True
        # try to fix the wrist_rotation from here
        self.v_wrist_rotation = 5.0
        self.v_wrist_angle = 15.0

    def set_waypoints(self, waypoints):
        """Sets the waypoints the robot needs to as list of positions"""
        self.waypoints = waypoints

    def next_pos(self):
        """Gets the next position in the path into pos_target"""
        if self.waypoints is None or self.waypoints == []:
            return None
        wp = self.waypoints[0]
        dist = self.pos_current.empirical_distance(self.robot_controller.exp, wp)
        print(f"wp {wp}")
        print(f"Distance to wp {dist}")
        if self.pos_current.empirical_distance(self.robot_controller.exp, wp) <= 0.001:
            # current waypoint was reached get next
            del self.waypoints[0]
            if self.waypoints == []:
                print("Finished waypoints")
                return None
            wp = self.waypoints[0]
            print(f"New waypoint: {wp}")
        self.pos_target = move_position_towards(self.pos_current, wp, self, self.robot_controller.exp)
        return self.pos_target

    def control(self):
        """The main control loop"""
        self.exit_control = False
        while True:
            start_time = time.time() 
            key = self.camera_controller.update() 
            if self.exit_control:
                self.stop()
                break;            
            # reached, self.pos_target = self.move_to_waypoint()
            self.max_timesteps -= 1
            if self.next_pos() is None or self.max_timesteps <= 0:
                self.stop()
                break
            if self.interactive_confirm:
                dist = self.pos_current.empirical_distance(self.pos_target, self.robot_controller.exp)
                print(f"Proposed next target: {self.pos_target} which is at distance {dist} from current")
                proceed = input("Proceed? ") in ["y", "Y", ""]
                if not proceed:
                    self.stop()
                    break
            self.control_robot()
            self.pos_current = self.pos_target
            self.update()
            end_time = time.time() 
            execution_time = end_time - start_time 
            self.last_interval = execution_time
            time_to_sleep = max(0.0, self.controller_interval - execution_time) 
            time.sleep(time_to_sleep) 


