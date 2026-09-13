"""Architecture tests use test-only operations, not production components."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parents[2]))

from exp_run_config import Config
from sensorprocessing.composite import (
    OPERATIONS, CompositeModel, create_composite,
    read_configuration, register_operation, restore_composite,
)
from sensorprocessing import sp_factory
from training_harness.checkpoints import CheckpointStore, TrainingState, model_file


class Fields(nn.Module):
    def forward(self, x):
        return {"nested": {"double": 2 * x}}


class Add(nn.Module):
    def forward(self, left, right):
        return left + right


class Views(nn.Module):
    def forward(self, views):
        return views[0] + 2 * views[1]


class Accumulate(nn.Module):
    """Test-only temporal operation with an explicit, immutable context."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.))

    def advance(self, x, *, context, dt):
        previous = torch.zeros_like(x) if context is None else context
        updated = previous + self.scale * x * dt
        return {"value": updated}, updated


def step(name, operation, inputs, **kwargs):
    return dict(name=name, operation=operation, inputs=inputs, **kwargs)


def experiment(directory):
    return {
        "class": "CompositeSensorProcessing", "image_size": [4, 4],
        "latent_size": 2, "data_dir": directory, "model_file": "composite.pth",
        "steps": [step("linear", "linear", ["input"], width=2)],
        "output": "linear",
    }


class TestComposite(unittest.TestCase):
    def setUp(self):
        self.registry = patch.dict(OPERATIONS, {}, clear=True)
        self.registry.start()
        self.addCleanup(self.registry.stop)
        register_operation("linear", lambda s: nn.Linear(s["width"], s["width"], bias=False))
        register_operation("fields", lambda s: Fields())
        register_operation("add", lambda s: Add())
        register_operation("views", lambda s: Views())
        register_operation("accumulate", lambda s: Accumulate(), temporal=True)

    def temporal_exp(self, directory=""):
        exp = experiment(directory)
        exp.update(steps=[step("memory", "accumulate", ["input"])],
                   output="memory.value", sample_interval=0.5)
        return exp

    def test_temporal_branches_context_isolation_and_gradients(self):
        exp = self.temporal_exp()
        exp["steps"] += [step("second", "accumulate", ["memory.value"], frozen=True),
                         step("sum", "add", ["memory.value", "second.value"])]
        exp["output"] = "sum"
        model = CompositeModel(exp)
        x = torch.ones(1, 2, requires_grad=True)
        first, context = model.advance(x, None, dt=1.)
        saved = {key: value.clone() for key, value in context.items()}
        second, next_context = model.advance(x, context, dt=2.)
        torch.testing.assert_close(first, 2 * x)
        torch.testing.assert_close(second, 10 * x)
        fresh, _ = model.advance(x, None, dt=1.)
        torch.testing.assert_close(fresh, first)
        for key in context:
            torch.testing.assert_close(context[key], saved[key])
        second.sum().backward()
        torch.testing.assert_close(x.grad, torch.full_like(x, 10.))
        self.assertIsNotNone(model.operations["memory"].scale.grad)
        self.assertIsNone(model.operations["second"].scale.grad)
        self.assertFalse(model.operations["second"].training)
        self.assertEqual(set(next_context), {"memory", "second"})
        with self.assertRaises(KeyError):
            model.advance(x, {}, dt=1.)
        for method in (model, model.encode, model.forward_steps):
            with self.assertRaisesRegex(RuntimeError, "advance"):
                method(x)

    def test_explicit_context_detachment(self):
        model = CompositeModel(self.temporal_exp())
        x = torch.ones(1, 2, requires_grad=True)
        y = torch.ones(1, 2, requires_grad=True)
        _, context = model.advance(x, None, dt=1.)
        detached = {key: value.detach() for key, value in context.items()}
        z, _ = model.advance(y, detached, dt=1.)
        z.sum().backward()
        self.assertIsNone(x.grad)
        torch.testing.assert_close(y.grad, torch.ones_like(y))

    def test_temporal_wrapper_reset_reload_timing_and_helpers(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(Config().runtime, {"device": "cpu"}):
            exp = self.temporal_exp(directory)
            model = create_composite(exp)
            torch.save(model.state_dict(), model_file(exp))
            self.assertTrue(sp_factory.is_temporal_sp(exp))
            self.assertEqual(read_configuration(exp)["sample_interval"], 0.5)
            exp["sample_interval"] = 99.
            sp = sp_factory.create_sp(exp)
            x = torch.ones(1, 2, requires_grad=True)
            np.testing.assert_allclose(sp.process(x), [0.5, 0.5])
            np.testing.assert_allclose(sp.process(x, dt=2.), [2.5, 2.5])
            self.assertFalse(sp.context["memory"].requires_grad)
            self.assertEqual(set(sp.enc.state_dict()), {"operations.memory.scale"})
            sp.reset_context()
            np.testing.assert_allclose(sp.process(x), [0.5, 0.5])
            sp.load_encoder_checkpoint(required=True)
            self.assertIsNone(sp.context)
            with patch.object(sp.preprocessor, "from_file", return_value=x):
                np.testing.assert_allclose(sp.process_file("unused", dt=3.), [3., 3.])

    def test_multiview_temporal_helpers_advance_once_per_camera_set(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(Config().runtime, {"device": "cpu"}):
            exp = self.temporal_exp(directory)
            exp.update({"class": "CompositeMultiViewSensorProcessing", "num_views": 2,
                        "cameras": ["left", "right"]})
            exp["steps"].insert(0, step("views", "views", ["input"]))
            exp["steps"][1]["inputs"] = ["views"]
            torch.save(create_composite(exp).state_dict(), model_file(exp))
            sp = sp_factory.create_sp(exp)
            x = torch.ones(1, 2)
            with patch.object(sp.preprocessor, "from_capture", return_value=x):
                np.testing.assert_allclose(sp.process_captures([None, None], dt=2.), [6., 6.])
            demonstration = Mock()
            demonstration.get_image.return_value = (x, None)
            np.testing.assert_allclose(
                sp.process_demonstration(demonstration, 0, ["left", "right"], dt=1.), [9., 9.]
            )

    def test_variable_rate_without_default_requires_explicit_dt(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(Config().runtime, {"device": "cpu"}):
            exp = self.temporal_exp(directory)
            del exp["sample_interval"]
            torch.save(create_composite(exp).state_dict(), model_file(exp))
            sp = sp_factory.create_sp(exp)
            with self.assertRaises(KeyError):
                sp.process(torch.ones(1, 2))
            self.assertIsNone(sp.context)
            np.testing.assert_allclose(sp.process(torch.ones(1, 2), dt=2.), [2., 2.])

    def test_branches_structured_results_and_gradients(self):
        exp = experiment("")
        exp["steps"] = [
            step("fields", "fields", ["input"]),
            step("sum", "add", ["input", "fields.nested.double"]),
        ]
        exp["output"] = "sum"
        model = CompositeModel(exp)
        x = torch.tensor([[1., 2.]], requires_grad=True)
        torch.testing.assert_close(model(x), 3 * x)
        model(x).sum().backward()
        torch.testing.assert_close(x.grad, torch.full_like(x, 3))
        self.assertEqual(set(model.forward_steps(x)), {"input", "fields", "sum"})
        exp["steps"][1]["inputs"][0] = "missing"
        # The constructed graph is independent of later configuration edits.
        torch.testing.assert_close(model(x), 3 * x)

    def test_frozen_module_stays_eval_but_transmits_gradients(self):
        register_operation("frozen", lambda s: nn.Sequential(nn.Linear(2, 2), nn.Dropout()))
        exp = experiment("")
        exp["steps"].append(step("fixed", "frozen", ["linear"], frozen=True))
        exp["output"] = "fixed"
        model = CompositeModel(exp).eval().train()
        self.assertFalse(model.operations["fixed"].training)
        self.assertTrue(model.operations["linear"].training)
        model(torch.ones(3, 2)).sum().backward()
        self.assertIsNotNone(model.operations["linear"].weight.grad)
        self.assertTrue(all(p.grad is None for p in model.operations["fixed"].parameters()))
        self.assertIn("operations.fixed.0.weight", model.state_dict())

    def test_saved_architecture_and_source_initialization_only_once(self):
        resolver = Mock(side_effect=lambda s: dict(s, width=2))
        def initialize(module, config):
            with torch.no_grad():
                module.weight.fill_(4)
        initializer = Mock(side_effect=initialize)
        register_operation("source", lambda s: nn.Linear(s["width"], s["width"], bias=False),
                           resolve=resolver, initialize=initializer)
        with tempfile.TemporaryDirectory() as directory:
            exp = experiment(directory)
            exp["steps"] = [step("linear", "source", ["input"], frozen=True)]
            model = create_composite(exp)
            torch.save(model.state_dict(), model_file(exp))
            self.assertEqual(read_configuration(exp)["steps"][0]["width"], 2)
            resolver.side_effect = RuntimeError("Source configuration unavailable")
            initializer.side_effect = RuntimeError("Source checkpoint unavailable")
            exp["steps"] = []
            exp["image_size"] = [99, 99]
            restored = restore_composite(exp)
            restored.load_state_dict(torch.load(model_file(exp), weights_only=True))
            with patch.dict(Config().runtime, {"device": "cpu"}):
                processor = sp_factory.create_sp(exp)
            np.testing.assert_allclose(processor.process(torch.ones(1, 2)), [8, 8])
            self.assertEqual(processor.preprocessor.image_size, (4, 4))
            torch.testing.assert_close(restored(torch.ones(1, 2)), model(torch.ones(1, 2)))
            self.assertEqual(resolver.call_count, 1)
            self.assertEqual(initializer.call_count, 1)
            with self.assertRaises(FileExistsError):
                create_composite(exp)

    def test_harness_resume_restores_optimizer_and_model(self):
        with tempfile.TemporaryDirectory() as directory:
            exp = experiment(directory)
            model = create_composite(exp)
            optimizer = torch.optim.Adam(model.parameters())
            model(torch.ones(2, 2)).square().sum().backward()
            optimizer.step()
            store = CheckpointStore(directory, exp["model_file"])
            store.save_epoch(0, TrainingState(best_loss=1.), model, optimizer, 1., 1.)
            restored = restore_composite(exp)
            restored_optimizer = torch.optim.Adam(restored.parameters())
            state = store.resume_latest(restored, restored_optimizer, "cpu")
            self.assertEqual(state.next_epoch, 1)
            self.assertTrue(restored_optimizer.state)
            torch.testing.assert_close(model(torch.ones(1, 2)), restored(torch.ones(1, 2)))

    def test_multiview_wrapper_uses_saved_order(self):
        with tempfile.TemporaryDirectory() as directory:
            exp = experiment(directory)
            exp.update({"class": "CompositeMultiViewSensorProcessing", "num_views": 2,
                        "cameras": ["left", "right"],
                        "steps": [step("views", "views", ["input"])], "output": "views"})
            model = create_composite(exp)
            torch.save({"model_state_dict": model.state_dict()}, model_file(exp))
            exp["cameras"] = ["right", "left"]
            with patch.dict(Config().runtime, {"device": "cpu"}):
                processor = sp_factory.create_sp(exp)
            self.assertTrue(sp_factory.is_multiview_sp(exp))
            self.assertIsNone(processor.process_file)
            np.testing.assert_allclose(processor.process([torch.ones(1, 2), 3 * torch.ones(1, 2)]), [7, 7])
            with self.assertRaisesRegex(ValueError, "Camera order"):
                processor.process_demonstration(None, 0, ["right", "left"])

    def test_unknown_operation_and_reference_errors_propagate(self):
        exp = experiment("")
        exp["steps"][0]["inputs"] = ["missing"]
        with self.assertRaisesRegex(KeyError, "missing"):
            CompositeModel(exp)(torch.ones(1, 2))
        with tempfile.TemporaryDirectory() as directory:
            exp = experiment(directory)
            create_composite(exp)
            OPERATIONS.clear()
            with self.assertRaisesRegex(KeyError, "linear"):
                sp_factory.create_sp(exp)


if __name__ == "__main__":
    unittest.main()
