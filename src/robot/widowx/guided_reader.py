"""Read stamped feedback while the separately started gravity-compensation node owns the arm."""
from time import monotonic


class JointStateReader:
    def __init__(self, exp, spec):
        self.robot = exp.get("interbotix_robot_name") or exp["robot_model"]
        self.names = spec["joint_names"]
        self.timeout = spec["timeout_s"]
        self.latest = None

    def start(self):
        import rclpy
        from rclpy.context import Context
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import JointState
        self.context = Context()
        rclpy.init(context=self.context)
        self.node = rclpy.create_node("berrypicker_guided_recorder", context=self.context)
        from rclpy.executors import SingleThreadedExecutor
        self.executor = SingleThreadedExecutor(context=self.context)
        self.executor.add_node(self.node)
        self.node.create_subscription(JointState, f"/{self.robot}/joint_states",
                                      self.receive, qos_profile_sensor_data)

    def receive(self, message):
        named = dict(zip(message.name, message.position))
        self.latest = [float(named[name]) for name in self.names]
        self.stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9

    def get_joints(self):
        self.latest = None
        end = monotonic() + self.timeout
        while self.latest is None:
            remaining = end - monotonic()
            if remaining <= 0:
                raise TimeoutError("No joint feedback from the hand-guided WidowX")
            self.executor.spin_once(timeout_sec=remaining)
        return self.latest

    def stop(self):
        self.executor.shutdown()
        self.node.destroy_node()
        self.context.shutdown()
