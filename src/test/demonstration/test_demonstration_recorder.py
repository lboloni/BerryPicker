import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.append(str(pathlib.Path(__file__).parents[2] / "demonstration"))
sys.path.append(str(pathlib.Path(__file__).parents[2]))

from demonstration_participant import (
    DemonstrationParticipant,
    DemonstrationSample,
    _load_participant_experiment,
)
from demonstration_recorder import DemonstrationRecorder


class FakeDemonstration:
    def __init__(self):
        self.actions = []
        self.annotations = []
        self.metadata = {"maxsteps": 0}
        self.saved = False

    def save_metadata(self):
        self.saved = True


class FakeActionParticipant(DemonstrationParticipant):
    def __init__(self):
        super().__init__("robot", {}, None)
        self.events = []
        self.update_count = 0

    def start(self, context):
        self.events.append("start")

    def update(self, context, dt):
        self.update_count += 1
        self.events.append("update")
        if self.update_count == 2:
            context.request_stop()

    def sample(self, context):
        return DemonstrationSample(
            action={"rc-position-target": {"height": 5.0}},
            telemetry={"position": {"height": 5.0}},
        )

    def stop(self, context):
        self.events.append("stop")


class FakeCameraParticipant(DemonstrationParticipant):
    def __init__(self):
        super().__init__("camera", {}, None)
        self.events = []

    def start(self, context):
        self.events.append("start")

    def update(self, context, dt):
        self.events.append("update")

    def sample(self, context):
        return DemonstrationSample(images={"overhead": object()})

    def stop(self, context):
        self.events.append("stop")


class TestDemonstrationRecorder(unittest.TestCase):
    def test_participant_experiment_overrides_machine_binding_experiment(self):
        spec = {"name": "automove", "exp": "automove", "run": "automove_random_ee_box_00"}
        binding = {"exp": "machine_default", "run": "machine_default_run"}
        with patch("demonstration_participant.Config") as config_class:
            config = config_class.return_value
            config.get_experiment.return_value = "loaded-exp"

            self.assertEqual(_load_participant_experiment(spec, binding), "loaded-exp")

        config.get_experiment.assert_called_once_with("automove", "automove_random_ee_box_00")

    def test_participant_experiment_fails_without_spec_or_binding_experiment(self):
        with self.assertRaisesRegex(ValueError, "must define exp/run"):
            _load_participant_experiment({"name": "automove"}, {})

    def test_runs_participants_and_saves_one_synchronized_timestep(self):
        demonstration = FakeDemonstration()
        robot = FakeActionParticipant()
        camera = FakeCameraParticipant()
        with tempfile.TemporaryDirectory() as directory:
            recorder = DemonstrationRecorder(
                demonstration, [robot, camera], directory, tick_interval=0.0
            )
            with patch("demonstration_recorder.cv2.imwrite", return_value=True) as write:
                recorder.run()

        self.assertEqual(robot.events, ["start", "update", "update", "stop"])
        self.assertEqual(camera.events, ["start", "update", "update", "stop"])
        self.assertEqual(demonstration.actions, [{"rc-position-target": {"height": 5.0}}])
        self.assertEqual(demonstration.annotations, [{"robot": {"position": {"height": 5.0}}}])
        self.assertEqual(demonstration.metadata["maxsteps"], 1)
        self.assertEqual(demonstration.metadata["cameras"], ["overhead"])
        self.assertTrue(demonstration.saved)
        write.assert_called_once()

    def test_rejects_more_than_one_action_producer(self):
        demonstration = FakeDemonstration()
        with tempfile.TemporaryDirectory() as directory:
            recorder = DemonstrationRecorder(demonstration, [FakeActionParticipant()], directory, 0.1)
            with self.assertRaisesRegex(RuntimeError, "exactly one action producer"):
                recorder.save(
                    {
                        "first": DemonstrationSample(action={}),
                        "second": DemonstrationSample(action={}),
                    }
                )


if __name__ == "__main__":
    unittest.main()
