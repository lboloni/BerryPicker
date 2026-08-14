"""
al5d_pulse_controller.py

A low-level position controller for the al5d robot
"""

from exp_run_config import Config, Experiment
Config.PROJECTNAME = "BerryPicker"

import numpy as np
import serial
import sys
import time
from .al5d_helper import RobotHelper
from .al5d_constants import SERVO_COUNT
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class PulseController:
    """A robot controller for the AL5D that operates at the level of the pulse. It interfaces directly to the hardware controller. Keeps the state of the robot in terms of the current pulse values of the servos. It is intended to be wrapped with more advanced controllers.

    https://wiki.lynxmotion.com/info/wiki/lynxmotion/view/servo-erector-set-system/ses-electronics/ses-modules/ssc-32/ssc-32-manual/#qpwidth
    """

    def __init__(self, exp):
        self.exp = exp
        # the position taken by every motor when we are starting up
        # the robot
        self.pulse_position_default = exp["pulse_position_default"]
        # the position taken by every motor when we are starting up
        # the robot
        # self.pulse_position_zero = 0
        # tracking the positions of the robot in terms of the pulse positions at the servos
        self.positions_pulse = np.full(
            SERVO_COUNT, self.pulse_position_default, dtype=int)
        # if sp == None:
        #     if device == None:
        #         raise Exception("No device specified")                
        #     try:
        #         self.sp = serial.Serial(device, 9600)
        #     except serial.serialutil.SerialException as spex:
        #         print(spex)
        #         sys.exit(1)            
        # else:
        #     self.sp = sp
        try:
            self.sp = serial.Serial(exp["device"], 9600, timeout=1)
            self.command_finished = True
        except serial.SerialException as se:
            print(f"Try out the backup {exp['device_backup']}")
            self.sp = serial.Serial(exp["device_backup"], 9600, timeout=1)
            self.command_finished = True

    def as_dict(self):
        """Returns the pulse configuration as a dictionary of lists, to be put into a saved yaml file"""
        retval = {}
        for i in range(SERVO_COUNT):
            retval[i] = self.positions_pulse[i].item()
        return retval
            
    def start_robot(self, speed=100):
        """ Starts the robot and brings all the motors to the default position"""
        for servo in range(SERVO_COUNT):
            self.control_servo_pulse(servo, self.pulse_position_default, speed)

    def stop_robot(self):
        """ Brings the robot relatively quickly to the default position. 
        Then, sets all motors to idle/unpowered (pulse = 0) """
        self.start_robot(self.exp["robot_start_speed"])
        time.sleep(5)
        self.command_finished = False
        for servo in range(SERVO_COUNT):
            command = f"#{servo} P{0}\r"
            self.sp.write(command.encode())

    def control_servo_pulse(self, servo, pulse, speed):
        """Sends a command to the particular servo to move to the particular
        pulse value, moving with the specified speed.
        """
        command = f"#{servo} P{pulse} S{speed}\r"
        logger.info(command)
        self.sp.write(command.encode())
        # FIXME: do we need this???
        self.wait_until_complete()
        self.positions_pulse[servo] = pulse

    def wait_until_complete(self, timeout=30):
        """Wait until all the commands are complete"""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            command = "Q\r"
            self.sp.write(command.encode())
            readbytes = self.sp.read()
            # print(f"readbytes = {readbytes}")
            if readbytes == b'+':
                #print("command executed done")
                self.command_finished = True
                return
            self.command_finished = False
        raise TimeoutError(f"Robot command did not finish within {timeout} seconds")

    def control_pulses(self, target_pulses):
        """Sends a command to set all the motors simultaneously"""
        target_pulses = np.asarray(target_pulses, dtype=float)
        if target_pulses.shape != (SERVO_COUNT,) or not np.all(np.isfinite(target_pulses)):
            raise ValueError(f"Expected {SERVO_COUNT} finite pulse values")
        if not np.all(target_pulses == np.floor(target_pulses)):
            raise ValueError("Pulse values must be integers")
        if not np.all((self.exp["CST_PULSE_MIN"] <= target_pulses) &
                      (target_pulses <= self.exp["CST_PULSE_MAX"])):
            raise ValueError("Pulse values are outside configured limits")
        target_pulses = target_pulses.astype(int)
        time=self.exp["TIME_DEFAULT"]
        command = ""
        for i in range(SERVO_COUNT):
            command += f"#{i} P {target_pulses[i]} "
        command += f" T{time}\r"
        logger.info(command)
        self.sp.write(command.encode())
        self.wait_until_complete()
        self.positions_pulse = target_pulses


    def control_servo_relative_pulse(self, servo, relative_pulse):
        """Sends a command a particular servo to change its position with 
        the specified pulse value (positive or negative), and a specific speed
        """
        # TODO: some way to check whether a bad value was passed, probably error should be raised.
        # TODO: option to wait here to execute
        speed=self.exp["CST_SPEED_DEFAULT"]
        pulse = self.positions_pulse[servo] + relative_pulse
        pulse, constrained = RobotHelper.constrain(
            pulse, self.exp["CST_PULSE_MIN"], self.exp["CST_PULSE_MAX"])
        if constrained:
            logger.warning("out of range at control_servo_relative_pulse")
        self.control_servo_pulse(servo, pulse, speed)

    def __str__(self):
        """Print the status of the robot"""
        return f"AL5D_PulseController positions = {self.positions_pulse}"
