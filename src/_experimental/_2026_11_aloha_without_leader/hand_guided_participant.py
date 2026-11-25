from time import monotonic
from threading import Event, Thread
from queue import Empty, SimpleQueue

from demonstration.demonstration_participant import DemonstrationParticipant, DemonstrationSample


class HandGuidedParticipant(DemonstrationParticipant):
    """Record measured waypoints from one manually guided task arm."""
    def start(self, context):
        self.remaining = self.spec["max_timesteps"]
        if self.exp["robot_name"] == "wxai":
            from robot.wxai.leader_controller import LeaderController
            self.reader = LeaderController(self.exp)
        else:
            from .guided_reader import JointStateReader
            self.reader = JointStateReader(self.exp, self.spec)
        self.reader.start()
        self.samples = SimpleQueue()
        self.end = Event()
        self.worker = Thread(target=self._poll, daemon=True)
        self.worker.start()

    def _poll(self):
        try:
            while not self.end.is_set():
                joints = self.reader.get_joints()
                sample = {"joints": joints, "read_monotonic": monotonic()}
                if hasattr(self.reader, "stamp"):
                    sample["ros_stamp"] = self.reader.stamp
                self.samples.put(sample)
                self.end.wait(self.spec["state_interval"])
        except Exception as error:
            self.samples.put(error)

    def update(self, context, dt):
        if self.remaining <= 0 or context.values.get("key") is True:
            context.request_stop()
        self.remaining -= 1

    def sample(self, context):
        # Preserve high-rate measurements between camera ticks, including startup.
        samples = [self.samples.get(timeout=self.spec["timeout_s"])]
        while True:
            try:
                samples.append(self.samples.get_nowait())
            except Empty:
                break
        for item in samples:
            if isinstance(item, Exception):
                raise item
        key = context.values.get("camera_key")
        return DemonstrationSample(
            action={"taught-joints": samples[-1]["joints"]},
            telemetry={"state_samples": samples},
            annotations={"gripper_event": {ord("g"): "grasp", ord("r"): "release"}.get(key)}
        )

    def stop(self, context):
        self.end.set()
        self.worker.join()
        self.reader.stop()
