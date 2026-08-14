"""
automove_controller.py

Automove controller for the AL5D robot. Generates moving patterns useful for demonstrations, such as various kinds of random. 
"""
from robot.al5d_position_controller import RobotPosition, PositionController
from robot.helper_al5d_move import move_position_towards
from .abstract_controller import AbstractController
from exp_run_config import Experiment

import time
import logging
import random
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class AutoMoveController(AbstractController):
    """A programmed robot controller that works by reaching a set of waypoints with the robot. 
    The parameters are described by an exp of type automove. 
    The waypoints are not determined here, they are passed into this controller
    """
    def __init__(self, exp: Experiment, robot_controller: PositionController = None, camera_controller = None, demonstration_recorder = None):
        super().__init__(robot_controller, camera_controller, demonstration_recorder)
        self.robot_controller = robot_controller
        self.automove_type = exp["automove_type"]
        self.max_timesteps = exp["max_timesteps"]
        self.interactive_confirm = exp["interactive_confirm"]
        # try to fix the wrist_rotation from here
        self.v_wrist_rotation = exp["v_wrist_rotation"]
        self.v_wrist_angle = exp["v_wrist_angle"]

    def generate_waypoints(self):
        if self.automove_type == "random_waypoint_6D":
            self.generate_waypoints_automove_6d()
        else:
            raise Exception(f"Automove generate_waypoints not supported for {self.automove_type}")

    def generate_waypoints_automove_6d(self):
        """Generates the robot path by adding random waypoints along uniformly distributed along the six dimensions of the robot control."""
        # create wpcount 
        self.waypoints = []
        wpcount = 10
        while True:
            norm = [0] * 6
            for df in range(6):
                norm[df] = random.random()                
            rp = RobotPosition.from_normalized_vector(self.robot_controller.exp, norm)
            if RobotPosition.limit(self.robot_controller.exp, rp):
                print(f"added waypoint {rp}")
                self.waypoints.append(rp)
            if len(self.waypoints) >= wpcount:
                break

    def set_waypoints(self, waypoints):
        """Sets the waypoints externally"""
        self.waypoints = waypoints

    def next_pos(self):
        """Gets the next position in the path into pos_target"""
        if self.waypoints is None or self.waypoints == []:
            return None
        wp = self.waypoints[0]
        dist = self.pos_current.empirical_distance(self.robot_controller.exp, wp)
        print(f"*** next_pos: next waypoint is {wp} at distance {dist}")
        if self.pos_current.empirical_distance(self.robot_controller.exp, wp) <= 0.001:
            # current waypoint was reached get next
            del self.waypoints[0]
            if self.waypoints == []:
                print("Finished waypoints")
                return None
            wp = self.waypoints[0]
            print(f"New waypoint: {wp}")
        max_steps = {
            field: getattr(self, f"v_{field}") * self.robot_interval
            for field in RobotPosition.FIELDS
        }
        self.pos_target = move_position_towards(
            self.robot_controller.exp, self.pos_current, wp, max_steps)
        return self.pos_target

    def control(self):
        """The main control loop"""
        self.exit_control = False
        self.autonomous_countdown = 0
        while True:
            self.autonomous_countdown -= 1
            start_time = time.time() 
            # Ensures that we can stop it by pressing q on the image display
            key = self.camera_controller.update() 
            if self.exit_control:
                self.stop()
                break;            
            # reached, self.pos_target = self.move_to_waypoint()
            self.max_timesteps -= 1
            if self.next_pos() is None or self.max_timesteps <= 0:
                self.stop()
                break
            if self.interactive_confirm and self.autonomous_countdown <= 0:
                dist = self.pos_current.empirical_distance(self.robot_controller.exp, self.pos_target)
                print(f"Proposed next target: {self.pos_target} which is at distance {dist} from current")
                proceed = input("Proceed? [stop/y/<number>]")
                if proceed == "stop":
                    self.stop()
                    break
                if proceed.isdigit():
                    self.autonomous_countdown = int(proceed)
            self.control_robot()
            self.pos_current = self.pos_target
            self.update()
            end_time = time.time() 
            execution_time = end_time - start_time 
            self.last_interval = execution_time
            time_to_sleep = max(0.0, self.controller_interval - execution_time) 
            time.sleep(time_to_sleep) 

