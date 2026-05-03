"""
mobile_camera_controller.py

Controller placeholder for a mobile camera used during demonstration recording.
"""


class MobileCamera:
    """Store the configuration for a mobile camera controller."""

    def __init__(self, exp, robot_controller=None):
        from interbotix_common_modules.common_robot.robot import robot_shutdown
        from interbotix_common_modules.common_robot.robot import robot_startup
        from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
        from mobile_camera.mobile_camera_decision import SimpleHighLowMobileCameraDecision
        from mobile_camera.observation_dome import ObservationDome
        from mobile_camera.widowx_observation_dome import WidowX_ObservationDomeDriver

        self.exp = exp
        self.robot_controller = robot_controller
        self.name = exp["controller_name"]
        self.mobile_camera_id = exp["mobile_camera_id"]
        self.robot_position = None
        self.robot_shutdown = robot_shutdown

        dome_config = exp["observation_dome"]
        self.dome = ObservationDome(
            x=dome_config["x"],
            y=dome_config["y"],
            z=dome_config["z"],
            r=dome_config["r"],
            no_long=dome_config["no_long"],
            no_lat=dome_config["no_lat"],
        )
        self.default_lat_index = exp.get("default_lat_index", 0)
        self.default_long_index = exp.get("default_long_index", 0)
        self.move_kwargs = exp.get("move_kwargs", {})

        self.bot = InterbotixManipulatorXS(
            robot_model=exp.get("robot_model", "wx250s"),
            group_name=exp.get("group_name", "arm"),
            gripper_name=exp.get("gripper_name", "gripper"),
        )
        robot_startup()

        self.dome_driver = WidowX_ObservationDomeDriver(
            self.bot.arm,
            self.dome,
            safety_box=exp.get("safety_box", None),
        )
        decision_config = exp.get("decision", {})
        self.decision = SimpleHighLowMobileCameraDecision(
            self.dome_driver,
            height_threshold=decision_config.get("height_threshold", 3.0),
            middle_lat_index=decision_config.get("middle_lat_index", None),
            middle_long_index=decision_config.get("middle_long_index", 0),
            reachability_kwargs=decision_config.get("reachability_kwargs", {}),
        )

    def update(self):
        """Update the mobile camera after the robot has moved."""
        if self.robot_controller is not None:
            self.robot_position = self.robot_controller.get_position()
        lat_index, long_index = self.get_lat_long_for_robot_position(
            self.robot_position
        )
        return self.move_to_lat_long(lat_index, long_index)

    def get_lat_long_for_robot_position(self, robot_position):
        """Return the dome node to use for a robot position."""
        return self.decision.get_lat_long_for_robot_position(robot_position)

    def move_to_lat_long(self, lat_index, long_index, **kwargs):
        """Move the mobile camera to a latitude/longitude node on the dome."""
        move_kwargs = dict(self.move_kwargs)
        move_kwargs.update(kwargs)
        print(f"Moving mobile camera to lat {lat_index}, long {long_index} with kwargs {move_kwargs}")
        return self.dome_driver.try_move_to_lat_long(
            lat_index,
            long_index,
            **move_kwargs,
        )

    def move_to_long_lat(self, long_index, lat_index, **kwargs):
        """Move the mobile camera to a longitude/latitude node on the dome."""
        return self.move_to_lat_long(lat_index, long_index, **kwargs)

    def stop(self):
        """Stop the mobile camera controller."""
        if hasattr(self, "bot"):
            self.robot_shutdown(self.bot.core.get_node())
