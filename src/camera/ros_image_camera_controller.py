"""Capture named camera views from ROS ``sensor_msgs/Image`` topics."""

from __future__ import annotations

import logging
import math
import threading
import time

import cv2
import numpy as np


logger = logging.getLogger(__name__)


class _RosApi:
    """Lazy ROS imports and lifecycle operations used by the controller."""

    def __init__(self):
        try:
            import rclpy
            from rclpy.context import Context
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.qos import qos_profile_sensor_data
            from sensor_msgs.msg import Image
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "ROS image cameras require rclpy and sensor_msgs"
            ) from error
        self.rclpy = rclpy
        self.Context = Context
        self.SingleThreadedExecutor = SingleThreadedExecutor
        self.Image = Image
        self.qos_profile_sensor_data = qos_profile_sensor_data

    def create_context(self):
        return self.Context()

    def init(self, context):
        self.rclpy.init(context=context)

    def shutdown(self, context):
        self.rclpy.shutdown(context=context)

    def create_node(self, name, context):
        return self.rclpy.create_node(name, context=context)

    def create_executor(self, context):
        return self.SingleThreadedExecutor(context=context)


class RosImageCameraController:
    """Cache synchronized snapshots from one or more ROS image topics."""

    def __init__(self, exp, ros_api=None, monotonic=time.monotonic):
        self.exp = exp
        self._monotonic = monotonic
        self.views = self._read_views(exp)
        self.image_size = self._read_image_size(exp["saved_image_size"])
        self.encoding = self._nonempty_string(
            exp.get("encoding", "bgr8"), "encoding"
        )
        if self.encoding != "bgr8":
            raise ValueError("ROS image camera output encoding must be bgr8")
        self.startup_timeout = self._positive_finite(
            exp.get("startup_timeout", 5.0), "startup_timeout"
        )
        self.shutdown_timeout = self._positive_finite(
            exp.get("shutdown_timeout", 0.5), "shutdown_timeout"
        )
        self.max_frame_age = self._positive_finite(
            exp.get("max_frame_age", 0.25), "max_frame_age"
        )
        self.max_frame_skew = self._nonnegative_finite(
            exp.get("max_frame_skew", 0.05), "max_frame_skew"
        )
        self.visualize = exp.get("visualize", False)
        if not isinstance(self.visualize, bool):
            raise ValueError("ROS image camera visualize must be boolean")

        self._api = ros_api
        self._context = None
        self._context_initialized = False
        self._node = None
        self._executor = None
        self._executor_thread = None
        self._subscriptions = []
        self._condition = threading.Condition()
        self._frames = {}
        self._frame_metadata = {}
        self._sequences = {name: 0 for name in self.views}
        self._snapshot_images = {}
        self._snapshot_metadata = {}
        self._callback_error = None
        self._caption = "ROS cameras: " + " ".join(self.views)
        self.started = False

    @staticmethod
    def _nonempty_string(value, name):
        if not isinstance(value, str) or not value:
            raise ValueError(f"ROS image camera {name} must be a nonempty string")
        return value

    @staticmethod
    def _positive_finite(value, name):
        if (
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"ROS image camera {name} must be positive and finite")
        return float(value)

    @staticmethod
    def _nonnegative_finite(value, name):
        if (
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"ROS image camera {name} must be nonnegative and finite")
        return float(value)

    @staticmethod
    def _read_image_size(value):
        if (
            not isinstance(value, (list, tuple))
            or len(value) != 2
            or any(not isinstance(item, int) or isinstance(item, bool) or item <= 0
                   for item in value)
        ):
            raise ValueError(
                "ROS image camera saved_image_size must contain two positive integers"
            )
        return tuple(value)

    @classmethod
    def _read_views(cls, exp):
        views = exp.get("views")
        if not isinstance(views, dict) or not views:
            raise ValueError("ROS image camera views must be a nonempty mapping")
        result = {}
        topics = set()
        for view_name, view_config in views.items():
            cls._nonempty_string(view_name, "view name")
            if not isinstance(view_config, dict):
                raise ValueError(
                    f"ROS image camera view {view_name} must be a mapping"
                )
            topic = cls._nonempty_string(
                view_config.get("topic"), f"view {view_name} topic"
            )
            if topic in topics:
                raise ValueError(f"Duplicate ROS image camera topic: {topic}")
            topics.add(topic)
            result[view_name] = topic
        return result

    def start(self):
        if self.started:
            raise RuntimeError("ROS image camera controller is already started")
        self._api = self._api or _RosApi()
        try:
            self._context = self._api.create_context()
            self._api.init(self._context)
            self._context_initialized = True
            self._node = self._api.create_node(
                f"berrypicker_ros_image_cameras_{id(self):x}", self._context
            )
            for view_name, topic in self.views.items():
                callback = self._make_callback(view_name, topic)
                subscription = self._node.create_subscription(
                    self._api.Image,
                    topic,
                    callback,
                    self._api.qos_profile_sensor_data,
                )
                self._subscriptions.append(subscription)
            self._executor = self._api.create_executor(self._context)
            self._executor.add_node(self._node)
            self.started = True
            self._executor_thread = threading.Thread(
                target=self._executor.spin,
                name="berrypicker-ros-image-cameras",
                daemon=True,
            )
            self._executor_thread.start()
            self._wait_for_frames(
                {name: 0 for name in self.views}, self.startup_timeout, "initial"
            )
            logger.info("ROS camera subscriptions ready: %s", self.views)
        except Exception:
            self._close_ros()
            raise

    def _make_callback(self, view_name, topic):
        def callback(message):
            received_at = self._monotonic()
            try:
                image = self._message_to_bgr(message)
                if (image.shape[1], image.shape[0]) != self.image_size:
                    image = cv2.resize(image, self.image_size)
                stamp = message.header.stamp
                source_timestamp = float(stamp.sec) + float(stamp.nanosec) / 1e9
            except Exception as error:
                with self._condition:
                    self._callback_error = RuntimeError(
                        f"Unable to process ROS image view {view_name} from {topic}"
                    )
                    self._callback_error.__cause__ = error
                    self._condition.notify_all()
                return
            with self._condition:
                self._sequences[view_name] += 1
                self._frames[view_name] = image.copy()
                self._frame_metadata[view_name] = {
                    "topic": topic,
                    "sequence": self._sequences[view_name],
                    "source_timestamp": source_timestamp,
                    "received_at": received_at,
                }
                self._condition.notify_all()

        return callback

    @staticmethod
    def _message_to_bgr(message):
        conversions = {
            "bgr8": (3, None),
            "rgb8": (3, cv2.COLOR_RGB2BGR),
            "bgra8": (4, cv2.COLOR_BGRA2BGR),
            "rgba8": (4, cv2.COLOR_RGBA2BGR),
            "mono8": (1, cv2.COLOR_GRAY2BGR),
        }
        try:
            channels, conversion = conversions[message.encoding.lower()]
        except (AttributeError, KeyError) as error:
            raise ValueError(
                f"Unsupported ROS image encoding: {message.encoding!r}"
            ) from error
        row_size = message.width * channels
        if message.height <= 0 or message.width <= 0 or message.step < row_size:
            raise ValueError(
                "ROS image dimensions and row stride must describe a nonempty image"
            )
        raw = np.frombuffer(message.data, dtype=np.uint8)
        required_size = message.height * message.step
        if raw.size < required_size:
            raise ValueError(
                f"ROS image buffer has {raw.size} bytes; expected {required_size}"
            )
        image = raw[:required_size].reshape(message.height, message.step)
        image = image[:, :row_size]
        if channels == 1:
            image = image.reshape(message.height, message.width)
        else:
            image = image.reshape(message.height, message.width, channels)
        if conversion is not None:
            image = cv2.cvtColor(image, conversion)
        return image

    def _wait_for_frames(self, minimum_sequences, timeout, description):
        deadline = self._monotonic() + timeout
        with self._condition:
            while True:
                if self._callback_error is not None:
                    raise self._callback_error
                missing = [
                    name for name, sequence in minimum_sequences.items()
                    if self._sequences[name] <= sequence
                ]
                if not missing:
                    return
                remaining = deadline - self._monotonic()
                if remaining <= 0:
                    topics = [self.views[name] for name in missing]
                    raise RuntimeError(
                        f"Timed out waiting for {description} ROS camera frames: {topics}"
                    )
                self._condition.wait(remaining)

    def update(self):
        if not self.started:
            raise RuntimeError("ROS image camera controller is not started")
        with self._condition:
            if self._callback_error is not None:
                raise self._callback_error
            captured_at = self._monotonic()
            metadata = {
                name: dict(self._frame_metadata[name]) for name in self.views
            }
            images = {name: self._frames[name].copy() for name in self.views}

        ages = {
            name: captured_at - values["received_at"]
            for name, values in metadata.items()
        }
        stale = [name for name, age in ages.items() if age > self.max_frame_age]
        if stale:
            raise RuntimeError(
                f"Stale ROS camera frames: "
                f"{[(name, ages[name]) for name in stale]}"
            )
        source_timestamps = [
            values["source_timestamp"] for values in metadata.values()
        ]
        if max(source_timestamps) - min(source_timestamps) > self.max_frame_skew:
            raise RuntimeError(
                "ROS camera frame timestamp skew exceeds max_frame_skew: "
                f"{metadata}"
            )
        for name, values in metadata.items():
            values["age"] = ages[name]
            del values["received_at"]
        self._snapshot_images = images
        self._snapshot_metadata = {"views": metadata}
        logger.info(
            "ROS camera snapshot sequences=%s ages=%s",
            {name: values["sequence"] for name, values in metadata.items()},
            ages,
        )

        if self.visualize:
            cv2.imshow(
                self._caption,
                cv2.hconcat(list(images.values())),
            )
            return (cv2.waitKey(1) & 0xFF) == ord("q")
        return False

    def get_images(self):
        if not self._snapshot_images:
            raise RuntimeError("ROS image cameras have no sampled images")
        return {
            name: image.copy() for name, image in self._snapshot_images.items()
        }

    def get_metadata(self):
        if not self._snapshot_metadata:
            raise RuntimeError("ROS image cameras have no sampled metadata")
        return {
            "views": {
                name: dict(values)
                for name, values in self._snapshot_metadata["views"].items()
            }
        }

    def stop(self):
        if not self.started:
            raise RuntimeError("ROS image camera controller is not started")
        try:
            self._close_ros()
        finally:
            if self.visualize:
                cv2.destroyWindow(self._caption)

    def _close_ros(self):
        self.started = False
        with self._condition:
            self._condition.notify_all()
        executor_alive = False
        if self._executor is not None:
            self._executor.shutdown(timeout_sec=self.shutdown_timeout)
        if self._executor_thread is not None:
            self._executor_thread.join(timeout=self.shutdown_timeout)
            executor_alive = self._executor_thread.is_alive()
        if self._executor is not None and self._node is not None:
            self._executor.remove_node(self._node)
        if self._node is not None:
            self._node.destroy_node()
        if self._api is not None and self._context_initialized:
            self._api.shutdown(self._context)
            self._context_initialized = False
        self._subscriptions = []
        self._executor_thread = None
        self._executor = None
        self._node = None
        self._context = None
        if executor_alive:
            raise RuntimeError("ROS image camera executor did not stop")
