"""
mobile_camera_controller.py

Controller placeholder for a mobile camera used during demonstration recording.
"""


class MobileCamera:
    """Store the configuration for a mobile camera controller."""

    def __init__(self, exp, robot_controller=None):
        self.exp = exp
        self.robot_controller = robot_controller
        self.name = exp["controller_name"]
        self.mobile_camera_id = exp["mobile_camera_id"]
        self.robot_position = None

    def update(self):
        """Update the mobile camera after the robot has moved."""
        if self.robot_controller is not None:
            self.robot_position = self.robot_controller.get_position()

    def stop(self):
        """Stop the mobile camera controller."""
        pass
