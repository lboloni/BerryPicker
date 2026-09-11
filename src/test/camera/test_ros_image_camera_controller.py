import pathlib
import sys
import threading
import time
import unittest

import numpy as np

sys.path.append(str(pathlib.Path(__file__).parents[2]))

from camera.ros_image_camera_controller import RosImageCameraController


class FakeStamp:
    def __init__(self, seconds):
        self.sec = int(seconds)
        self.nanosec = int((seconds - self.sec) * 1e9)


class FakeMessage:
    def __init__(self, image, timestamp, encoding="bgr8"):
        self.height, self.width = image.shape[:2]
        self.encoding = encoding
        self.step = image.strides[0]
        self.data = image.tobytes()
        self.header = type("Header", (), {"stamp": FakeStamp(timestamp)})()


class FakeNode:
    def __init__(self):
        self.callbacks = {}
        self.destroyed = False

    def create_subscription(self, message_type, topic, callback, qos):
        if topic in self.callbacks:
            raise RuntimeError(f"Duplicate subscription for {topic}")
        self.callbacks[topic] = callback
        return topic

    def emit(self, topic, message):
        self.callbacks[topic](message)

    def destroy_node(self):
        self.destroyed = True


class FakeExecutor:
    def __init__(self, api):
        self.api = api
        self.node = None
        self.stopped = threading.Event()

    def add_node(self, node):
        self.node = node

    def spin(self):
        for topic, message in self.api.initial_messages.items():
            self.node.emit(topic, message)
        self.stopped.wait()

    def shutdown(self, timeout_sec):
        self.stopped.set()

    def remove_node(self, node):
        if node is not self.node:
            raise RuntimeError("Removed an unknown node")


class FakeRosApi:
    Image = object
    qos_profile_sensor_data = object()

    def __init__(self, initial_messages=None):
        self.initial_messages = initial_messages or {}
        self.node = FakeNode()
        self.executor = FakeExecutor(self)
        self.initialized = False
        self.shutdown_called = False

    @staticmethod
    def create_context():
        return object()

    def init(self, context):
        self.initialized = True

    def shutdown(self, context):
        self.shutdown_called = True

    def create_node(self, name, context):
        return self.node

    def create_executor(self, context):
        return self.executor

def camera_exp(**overrides):
    result = {
        "views": {
            "left": {"topic": "/camera/left/image_raw"},
            "right": {"topic": "/camera/right/image_raw"},
        },
        "saved_image_size": [4, 3],
        "encoding": "bgr8",
        "startup_timeout": 0.2,
        "shutdown_timeout": 0.2,
        "max_frame_age": 0.2,
        "max_frame_skew": 0.1,
        "visualize": False,
    }
    result.update(overrides)
    return result


class TestRosImageCameraController(unittest.TestCase):
    def test_captures_one_fresh_synchronized_frame_per_view(self):
        initial = np.zeros((2, 2, 3), dtype=np.uint8)
        api = FakeRosApi({
            "/camera/left/image_raw": FakeMessage(initial, 10.0),
            "/camera/right/image_raw": FakeMessage(initial, 10.0),
        })
        controller = RosImageCameraController(camera_exp(), ros_api=api)
        controller.start()

        api.node.emit(
            "/camera/left/image_raw",
            FakeMessage(np.full((2, 2, 3), 17, dtype=np.uint8), 11.00),
        )
        api.node.emit(
            "/camera/right/image_raw",
            FakeMessage(np.full((2, 2, 3), 23, dtype=np.uint8), 11.04),
        )
        update_result = controller.update()

        self.assertFalse(update_result)
        images = controller.get_images()
        self.assertEqual(images["left"].shape, (3, 4, 3))
        self.assertTrue(np.all(images["left"] == 17))
        self.assertTrue(np.all(images["right"] == 23))
        metadata = controller.get_metadata()["views"]
        self.assertEqual(metadata["left"]["sequence"], 2)
        self.assertAlmostEqual(metadata["right"]["source_timestamp"], 11.04)
        self.assertLessEqual(metadata["left"]["age"], 0.2)

        images["left"][:] = 99
        self.assertTrue(np.all(controller.get_images()["left"] == 17))
        controller.stop()
        self.assertTrue(api.node.destroyed)
        self.assertTrue(api.shutdown_called)

    def test_update_reuses_a_recent_cached_frame_without_waiting(self):
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        api = FakeRosApi({
            "/camera/left/image_raw": FakeMessage(image, 10.0),
            "/camera/right/image_raw": FakeMessage(image, 10.0),
        })
        controller = RosImageCameraController(camera_exp(), ros_api=api)
        controller.start()

        started = time.monotonic()
        controller.update()
        controller.update()
        elapsed = time.monotonic() - started

        self.assertLess(elapsed, 0.05)
        self.assertEqual(
            controller.get_metadata()["views"]["left"]["sequence"], 1
        )
        controller.stop()

    def test_start_raises_when_a_topic_has_no_frames_and_cleans_up(self):
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        api = FakeRosApi({
            "/camera/left/image_raw": FakeMessage(image, 10.0),
        })
        controller = RosImageCameraController(
            camera_exp(startup_timeout=0.02), ros_api=api
        )

        with self.assertRaisesRegex(RuntimeError, "/camera/right/image_raw"):
            controller.start()

        self.assertFalse(controller.started)
        self.assertTrue(api.node.destroyed)
        self.assertTrue(api.shutdown_called)

    def test_rejects_duplicate_topics(self):
        exp = camera_exp(views={
            "first": {"topic": "/camera/image_raw"},
            "second": {"topic": "/camera/image_raw"},
        })
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            RosImageCameraController(exp, ros_api=FakeRosApi())


if __name__ == "__main__":
    unittest.main()
