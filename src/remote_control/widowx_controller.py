"""
widowx_controller.py

A controller for the AL5D robot based on a backdriven WidowX robot
"""
from robot.al5d_position_controller import RobotPosition, PositionController
from .abstract_controller import AbstractController

import logging
import time

from interbotix_common_modules.common_robot.robot import robot_shutdown, robot_startup
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
import interbotix_common_modules.angle_manipulation as ang


logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

class WidowXController(AbstractController):
    """A controller based on a backdriven WidowX robot"""    

    def __init__(self, exp, robot_controller: PositionController = None, camera_controller = None, demonstration_recorder = None):
        super().__init__(robot_controller, camera_controller, demonstration_recorder)
        self.exp = exp
        # connecting the robot controller to the widowx
        self.bot = InterbotixManipulatorXS(
            robot_model='wx250s',
            group_name='arm',
            gripper_name='gripper',
        )

        robot_startup()

    def control(self):
        """The main control loop. 
        FIXME: this is just a proof of concept. We will need to figure out how to ensure that I have all the components"""
        self.exit_control = False
        while True:
            start_time = time.time() 
            # get the key from the opencv control loop
            key = self.camera_controller.update() 
            if key:
                self.process_key(key) 

            self.process_widowx()

            # FIXME: applying a safety reset which prevents us going out of range
            ok = RobotPosition.limit(self.robot_controller.exp, self.pos_target)
            if not ok:
                logger.warning(f"DANGER! exceeded range! {self.pos_target}")
            logger.warning(f"Target: {self.pos_target}")
            self.control_robot()
            self.update()
            end_time = time.time() 
            execution_time = end_time - start_time 
            self.last_interval = execution_time
            time_to_sleep = max(0.0, self.controller_interval - execution_time) 
            time.sleep(time_to_sleep) 


    def process_widowx(self):
        """Processes the movement of the widowx"""
        # get the joint positions from the WidowX
        joints = self.bot.arm.get_joint_positions()
        jo_heading = joints[0] # supposedly "waist" ????
        #jo_shoulder = joints[1] # supposedly "shoulder"
        #jo_elbow = joints[2] # supposedly "elbow"
        jo_wrist_rotate = joints[5]
        jo_wrist_angle = joints[4]

        val = self.bot.arm.get_ee_pose()
        #x, y, z, roll, pitch, yaw = self.bot.arm.#get_ee_pose_components()
        x = val[0, 3]
        y = val[1, 3]
        z = val[2, 3]
        print(f"x={x} y={y} z={z}")

        print(f'WidowX heading joint {jo_heading} current on al5d {self.pos_target["heading"]}')

        self.pos_target["heading"] = jo_heading * (-50.0)
        #print(f"jo_shoulder {jo_shoulder} pos_height {self.pos_target['height']}")

        self.pos_target["height"] = 2.0 + 10.0 * z
        import math
        self.pos_target["distance"] = 2.0 + 10.0 * math.sqrt(x*x+y*y)

        #print(f"jo_elbow {jo_elbow} pos_distance {self.pos_target['distance']}")
        #self.pos_target["height"] = jo_shoulder * 1.0
        #self.pos_target["distance"] = jo_elbow * 1.0
         # this will be commented out for the time being
        print(f"jo_wrist_angle={jo_wrist_angle} pos_wrist_angle={self.pos_target['wrist_angle']}")
        print(f"jo_wrist_rotate={jo_wrist_rotate} pos_wrist_rotation={self.pos_target['wrist_rotation']}")
        self.pos_target["wrist_angle"] = -45 - 15 * jo_wrist_angle
        self.pos_target["wrist_rotation"] = 75 + 40 * jo_wrist_rotate
    
    def process_key(self, key):
        """Some of the components of the keyboard based controllers are replicated here"""
        keycode = key & 0xFF

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
        self.pos_target["gripper"] += delta_gripper

        # square aka x - exit control
        if keycode == ord(self.exp["exit_control_ord"]):
            self.exit_control = True
            return
        # home h  
        if keycode == ord(self.exp["home_ord"]):
            self.pos_target = copy.copy(self.pos_home)
            return
