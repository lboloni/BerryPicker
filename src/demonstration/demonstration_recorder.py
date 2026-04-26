"""
demonstration_recorder.py

Code that helps in recording ongoing demonstrations

"""
import sys
sys.path.append("..")

from exp_run_config import Config
from robot.al5d_position_controller import PositionController
from camera.camera_controller import CameraController
import pathlib
import cv2
import copy


class DemonstrationRecorder:
    """Record demonstration data collected from various controllers, sensors etc.
    """

    @staticmethod
    def create_for_controller(demonstration, remote_control, robot_controller,
                              camera_controller, save_dir,
                              mobile_camera_controller=None):
        demo_recorder = DemonstrationRecorder(
            demonstration,
            camera_controller=camera_controller,
            mobile_camera_controller=mobile_camera_controller,
            remote_control=remote_control,
            robot_controller=robot_controller,
            save_dir=save_dir,
        )
        remote_control.demonstration_recorder = demo_recorder
        return demo_recorder

    @staticmethod
    def record_with_controller(controller, exp, demonstration, robot_controller,
                               camera_controller, demo_dir,
                               mobile_camera_controller=None):
        if controller == "xbox":
            return DemonstrationRecorder.record_with_xbox(
                exp, demonstration, robot_controller, camera_controller, demo_dir,
                mobile_camera_controller)
        if controller == "keyboard":
            return DemonstrationRecorder.record_with_keyboard(
                exp, demonstration, robot_controller, camera_controller, demo_dir,
                mobile_camera_controller)
        if controller == "widowx":
            return DemonstrationRecorder.record_with_widowx(
                exp, demonstration, robot_controller, camera_controller, demo_dir,
                mobile_camera_controller)
        if controller == "camera":
            return DemonstrationRecorder.record_with_camera(
                exp, demonstration, robot_controller, camera_controller, demo_dir,
                mobile_camera_controller)
        if controller == "automove":
            return DemonstrationRecorder.record_with_automove(
                exp, demonstration, robot_controller, camera_controller, demo_dir,
                mobile_camera_controller)
        if controller == "double":
            return DemonstrationRecorder.record_with_double(
                exp, demonstration, robot_controller, camera_controller, demo_dir,
                mobile_camera_controller)
        raise ValueError(f"Unknown primary controller: {controller}")

    @staticmethod
    def record_with_xbox(exp, demonstration, robot_controller, camera_controller,
                         demo_dir, mobile_camera_controller=None):
        gamepad_controller = DemonstrationRecorder.create_gamepad_controller(
            exp, robot_controller, camera_controller)
        DemonstrationRecorder.create_for_controller(
            demonstration, gamepad_controller, robot_controller, camera_controller,
            demo_dir, mobile_camera_controller)
        gamepad_controller.control()
        DemonstrationRecorder.finish_recording(demonstration)

    @staticmethod
    def record_with_keyboard(exp, demonstration, robot_controller,
                             camera_controller, demo_dir,
                             mobile_camera_controller=None):
        from remote_control.keyboard_controller import KeyboardController

        exp_keyboard_controller = Config().get_experiment(
            exp["exp_keyboard_controller"], exp["run_keyboard_controller"])
        kb_controller = KeyboardController(
            exp_keyboard_controller,
            robot_controller=robot_controller,
            camera_controller=camera_controller,
        )
        DemonstrationRecorder.create_for_controller(
            demonstration, kb_controller, robot_controller, camera_controller,
            demo_dir, mobile_camera_controller)
        kb_controller.control()
        DemonstrationRecorder.finish_recording(demonstration)

    @staticmethod
    def record_with_widowx(exp, demonstration, robot_controller,
                           camera_controller, demo_dir,
                           mobile_camera_controller=None):
        from remote_control.widowx_controller import WidowXController

        exp_widowx_controller = Config().get_experiment(
            exp["exp_widowx_controller"], exp["run_widowx_controller"])
        widowx_controller = WidowXController(
            exp_widowx_controller,
            robot_controller=robot_controller,
            camera_controller=camera_controller,
        )
        DemonstrationRecorder.create_for_controller(
            demonstration, widowx_controller, robot_controller, camera_controller,
            demo_dir, mobile_camera_controller)
        widowx_controller.control()
        DemonstrationRecorder.finish_recording(demonstration)

    @staticmethod
    def record_with_camera(exp, demonstration, robot_controller,
                           camera_controller, demo_dir,
                           mobile_camera_controller=None):
        # FIXME: future controller which is based on a camera and capturing human movement etc.
        pass

    @staticmethod
    def record_with_automove(exp, demonstration, robot_controller,
                             camera_controller, demo_dir,
                             mobile_camera_controller=None):
        from remote_control.automove_controller import AutoMoveController

        exp_automove_controller = Config().get_experiment(
            exp["exp_automove_controller"], exp["run_automove_controller"])
        automove_controller = AutoMoveController(
            exp_automove_controller,
            robot_controller=robot_controller,
            camera_controller=camera_controller,
        )
        automove_controller.generate_waypoints()
        DemonstrationRecorder.create_for_controller(
            demonstration, automove_controller, robot_controller, camera_controller,
            demo_dir, mobile_camera_controller)
        demonstration.save_metadata()
        automove_controller.control()

    @staticmethod
    def record_with_double(exp, demonstration, robot_controller,
                           camera_controller, demo_dir,
                           mobile_camera_controller=None):
        from remote_control.double_demo_controller import DoubleDemoController
        from remote_control.widowx_controller import WidowXController

        gamepad_controller = DemonstrationRecorder.create_gamepad_controller(
            exp, robot_controller, camera_controller)
        exp_widowx_controller = Config().get_experiment(
            exp["exp_widowx_controller"], exp["run_widowx_controller"])
        widowx_controller = WidowXController(
            exp_widowx_controller,
            robot_controller=robot_controller,
            camera_controller=camera_controller,
        )
        DemonstrationRecorder.create_for_controller(
            demonstration, gamepad_controller, robot_controller, camera_controller,
            demo_dir, mobile_camera_controller)

        filename = pathlib.Path(demo_dir, "double_results.yaml")
        double_controller = DoubleDemoController(
            al5d_controller=robot_controller,
            widowx_controller=widowx_controller,
            filename=filename,
        )
        gamepad_controller.double_demo_controller = double_controller

        gamepad_controller.control()
        DemonstrationRecorder.finish_recording(demonstration)

    @staticmethod
    def create_gamepad_controller(exp, robot_controller, camera_controller):
        try:
            from remote_control.gamepad_controller import GamepadController
        except ModuleNotFoundError:
            print("Approxeng module not found, cannot use gamepad")
            raise

        exp_gamepad_controller = Config().get_experiment(
            exp["exp_gamepad_controller"], exp["run_gamepad_controller"])
        return GamepadController(
            exp_gamepad_controller,
            robot_controller=robot_controller,
            camera_controller=camera_controller,
        )

    @staticmethod
    def create_mobile_camera_controller(exp, robot_controller=None):
        if "exp_mobile_camera_controller" not in exp:
            return None
        if "run_mobile_camera_controller" not in exp:
            return None

        from mobile_camera.mobile_camera_controller import MobileCamera

        exp_mobile_camera_controller = Config().get_experiment(
            exp["exp_mobile_camera_controller"],
            exp["run_mobile_camera_controller"],
        )
        return MobileCamera(exp_mobile_camera_controller, robot_controller)

    @staticmethod
    def finish_recording(demonstration):
        demonstration.save_metadata()
        print("====== Demonstration terminated and recorded successfully, bye. ======")

    def __init__(self, demonstration, remote_control,
                 robot_controller: PositionController,
                 camera_controller: CameraController,
                 mobile_camera_controller=None, save_dir=None):
        self.demonstration = demonstration
        self.save_dir = save_dir
        self.remote_control = remote_control
        self.robot_controller = robot_controller
        self.camera_controller = camera_controller
        self.mobile_camera_controller = mobile_camera_controller
        self.counter = 0
        pass

    def save(self):
        """Write the data from the various sources with a common prefix
        """
        save_prefix = f"{self.counter:05d}"
        # writing the robot data if the robot is available
        if self.robot_controller:
            assert len(self.demonstration.actions) == self.counter
            assert self.demonstration.metadata["maxsteps"] == self.counter
            data = {}
            data["rc-position-target"] = copy.copy(self.remote_control.pos_target.values)
            # the simulated controller does not have this
            if isinstance(self.robot_controller, PositionController):
                data["rc-angle-target"] = self.robot_controller.angle_controller.as_dict()
                data["rc-pulse-target"] = self.robot_controller.pulse_controller.as_dict()
            self.demonstration.actions.append(data)
            self.demonstration.annotations.append({})
        # save the captured images
        if self.camera_controller is not None:
            for index in self.camera_controller.images:
                filename = pathlib.Path(self.save_dir, f"{save_prefix}_{index}.jpg")
                cv2.imwrite(str(filename), self.camera_controller.images[index])
        self.counter += 1
        self.demonstration.metadata["maxsteps"] = self.counter

    def update_mobile_camera(self):
        """Update the mobile camera before the fixed cameras capture a frame."""
        if self.mobile_camera_controller is not None:
            self.mobile_camera_controller.update()

    def stop(self):
        if self.mobile_camera_controller is not None:
            self.mobile_camera_controller.stop()
        self.demonstration.save_metadata()
