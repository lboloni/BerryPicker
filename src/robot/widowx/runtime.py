"""Process-wide lifecycle management for the Interbotix ROS API."""


class WidowXRuntime:
    """Share one Interbotix node and executor among registered WidowX arms."""

    def __init__(self, api=None):
        self.api = api
        self.node = None
        self.started = False
        self.closed = False
        self._controllers = set()
        self._active_controllers = set()
        self._robot_names = set()

    def _load_api(self):
        if self.api is None:
            from interbotix_common_modules import angle_manipulation
            from interbotix_common_modules.common_robot.robot import robot_shutdown, robot_startup
            from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS

            self.api = type("InterbotixAPI", (), {
                "Manipulator": InterbotixManipulatorXS,
                "angle_manipulation": angle_manipulation,
                "robot_startup": staticmethod(robot_startup),
                "robot_shutdown": staticmethod(robot_shutdown),
            })
        return self.api

    def create_manipulator(self, owner, **kwargs):
        if self.started or self.closed:
            raise RuntimeError("Register all WidowX controllers before starting the ROS runtime")
        robot_name = kwargs.get("robot_name") or kwargs["robot_model"]
        if robot_name in self._robot_names:
            raise ValueError(f"Duplicate Interbotix robot name: {robot_name}")
        api = self._load_api()
        if self.node is not None:
            kwargs["node"] = self.node
        bot = api.Manipulator(**kwargs)
        node = bot.core.get_node()
        if self.node is None:
            self.node = node
        elif node is not self.node:
            raise RuntimeError("Interbotix manipulator did not use the shared WidowX node")
        self.register_manipulator(owner, bot, robot_name)
        return bot

    def register_manipulator(self, owner, bot, robot_name=None):
        """Register an already-created manipulator, primarily for dependency injection."""
        if self.started or self.closed:
            raise RuntimeError("Register all WidowX controllers before starting the ROS runtime")
        if owner in self._controllers:
            raise ValueError("WidowX controller is already registered")
        if robot_name is not None and robot_name in self._robot_names:
            raise ValueError(f"Duplicate Interbotix robot name: {robot_name}")
        node = bot.core.get_node()
        if self.node is None:
            self.node = node
        elif node is not self.node:
            raise RuntimeError("Interbotix manipulator did not use the shared WidowX node")
        self._controllers.add(owner)
        if robot_name is not None:
            self._robot_names.add(robot_name)

    def acquire(self, owner):
        if owner not in self._controllers:
            raise RuntimeError("WidowX controller is not registered with this runtime")
        if owner in self._active_controllers:
            raise RuntimeError("WidowX controller is already started")
        if self.closed:
            raise RuntimeError("The Interbotix ROS runtime has already been shut down")
        if not self.started:
            if self.node is None:
                raise RuntimeError("Cannot start a WidowX runtime without a registered robot")
            self._load_api().robot_startup(self.node)
            self.started = True
        self._active_controllers.add(owner)

    def release(self, owner):
        if owner not in self._active_controllers:
            raise RuntimeError("WidowX controller is not started")
        self._active_controllers.remove(owner)
        if not self._active_controllers:
            self._load_api().robot_shutdown(self.node)
            self.started = False
            self.closed = True

    def rotation_matrix_to_euler_angles(self, matrix):
        return self._load_api().angle_manipulation.rotation_matrix_to_euler_angles(matrix)


_DEFAULT_RUNTIME = None


def get_default_runtime():
    global _DEFAULT_RUNTIME
    if _DEFAULT_RUNTIME is None:
        _DEFAULT_RUNTIME = WidowXRuntime()
    return _DEFAULT_RUNTIME
