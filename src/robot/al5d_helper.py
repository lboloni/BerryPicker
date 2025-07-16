"""
al5d_helper.py

Helper functions for the al5d robot
"""

from exp_run_config import Config, Experiment
Config.PROJECTNAME = "BerryPicker"


import logging
logging.basicConfig(level=logging.WARNING)

class RobotHelper:
    """A collection of helper methods for the robot controller"""

    @staticmethod
    def constrain(value, min, max):
        """Constrain a value between min and max. It is usually used to contrain an angle. Returns the value and a boolean that says whether it had been actually constrained"""
        constrained = False
        if value < min:
            logging.warning(f"value too small {value}")
            constrained = True
            value = min
        if value > max:
            logging.warning(f"value too large {value}")
            constrained = True
            value = max
        return value, constrained

    @staticmethod
    def map_ranges(value, min1, max1, min2 = 0, max2 = 1):
        """Map a 0-1 from the range [min1, max1] to [min2, max2]"""
        value = ((value - min1) * (max2 - min2) / (max1 - min1) + min2)
        return (value)

    @staticmethod
    def angle_to_pulse(exp: Experiment, angle, constrain=True):
        """Returns the pulse that correspond to a certain angle. 
        If constrain is set, the angle will be constrained in the legal range."""
        if constrain:
            angle, constrained = RobotHelper.constrain(
                angle, exp["CST_ANGLE_MIN"], exp["CST_ANGLE_MAX"])
        pulse = RobotHelper.map_ranges(
            angle, exp["CST_ANGLE_MIN"], exp["CST_ANGLE_MAX"], exp["CST_PULSE_MIN"], exp["CST_PULSE_MAX"])
        return int(pulse), constrained

    @staticmethod
    def servo_angle_to_pulse(exp_angle: Experiment, exp_pulse: Experiment, servo, angle, constrain=True):
        """Performs the angle to pulse transformation on a servo-specific basis, while taking into consideration the specific limits and adding pulse corrections"""
        if constrain:
            min_value = exp_angle["ANGLE_LIMITS"][servo][0]
            max_value = exp_angle["ANGLE_LIMITS"][servo][2]
            angle, constrained = RobotHelper.constrain(
                angle, min_value, max_value)
        pulse = RobotHelper.map_ranges(
            angle, exp_angle["CST_ANGLE_MIN"], exp_angle["CST_ANGLE_MAX"], exp_pulse["CST_PULSE_MIN"], exp_pulse["CST_PULSE_MAX"])
        corrected_pulse = int(pulse) + exp_pulse["PULSE_CORRECTION"][servo]
        return corrected_pulse, constrained


    @staticmethod
    def pulse_to_angle(exp_pulse: Experiment, exp_angle: Experiment, pulse):
        """Returns the angle corresponding to a certain pulse"""
        angle = RobotHelper.map_ranges(
            pulse, exp_pulse["CST_PULSE_MIN"], exp_pulse["CST_PULSE_MAX"], exp_angle["CST_ANGLE_MIN"], exp_angle["CST_ANGLE_MAX"])
        return angle
