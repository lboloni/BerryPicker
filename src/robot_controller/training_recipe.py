"""Staged, resumable training for VAE--LSTM--MDN robot controllers."""

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import traceback

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from behavior_cloning.mdn import mdn_loss
from exp_run_config import Config
from robot_controller.abstract_trec import AbstractTrainingRecipe
from robot_controller.graph_robot_controller import load_controller_spec
from robot_controller.training_data import make_controller_dataloaders
from robot_controller.training_model import RobotControllerTrainingModel
from training_harness.checkpoints import model_file


Config.PROJECTNAME = "BerryPicker"
SCHEMA_VERSION = 1


def _now():
    return datetime.now(timezone.utc).isoformat()


def _plain(value):
    if hasattr(value, "values") and isinstance(value.values, dict):
        value = value.values
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _without_volatile(value):
    volatile = {
        "data_dir", "time_started", "time_done", "exp_run_sys_indep_file",
        "exp_run_sys_dep_file", "subrun_name",
    }
    value = _plain(value)
    if isinstance(value, dict):
        return {
            key: _without_volatile(item) for key, item in value.items()
            if key not in volatile
        }
    if isinstance(value, list):
        return [_without_volatile(item) for item in value]
    return value


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.flush()
    temporary.replace(path)


def _atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_positive(value, name):
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return float(value)


class StagedRobotControllerTrainingRecipe(AbstractTrainingRecipe):
    """Train configured controller components in explicit consecutive stages."""

    def __init__(
        self, exp_trec, *, experiment_loader=None, dataloader_factory=None
    ):
        self.exp = exp_trec
        self.load_experiment = experiment_loader or Config().get_experiment
        self.data_dir = Path(exp_trec["data_dir"])
        self.data_dir.mkdir(parents=True, exist_ok=True)
        controller_ref = exp_trec["controller"]
        self.controller_exp = self.load_experiment(
            controller_ref.get("exp", "robot_controller"), controller_ref["run"]
        )
        seed = exp_trec.get("random_seed", 0)
        if type(seed) is not int:
            raise ValueError("random_seed must be an integer")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.spec = load_controller_spec(
            self.controller_exp, experiment_loader=self.load_experiment
        )
        self.model = self._create_training_model(self.spec).to(
            Config().runtime["device"]
        )
        self.stages = self._validate_stages(exp_trec["stages"])
        robot_ref = exp_trec["robot"]
        self.robot_exp = self.load_experiment(robot_ref["exp"], robot_ref["run"])
        self.fingerprint = self._fingerprint()
        self.status_path = self.data_dir / "recipe_status.json"
        self.metrics_path = self.data_dir / "metrics.jsonl"
        self.source_manifest_path = self.data_dir / "source_manifest.json"
        self._dataloader_factory = (
            dataloader_factory or make_controller_dataloaders
        )
        self._optimizer = None
        self._scheduler = None
        self._active_stage_index = None
        self.status = self._read_or_create_status()

    def _create_training_model(self, spec):
        return RobotControllerTrainingModel(spec)

    def _default_monitor(self):
        return "validation_nll"

    def _supported_monitors(self):
        return {"validation_loss", "validation_nll"}

    def _training_metric_name(self):
        return "train_nll"

    def _loss_and_prediction(self, output, targets):
        mu, sigma, pi = output
        loss = mdn_loss(targets, mu, sigma, pi)
        prediction = torch.sum(pi * mu, dim=-1)
        return loss, prediction

    def _output_size(self):
        mdn_label = self.model.labels["MDN"]
        return self.spec["components"][mdn_label]["exp"]["output_dim"]

    def _validate_stages(self, stages):
        if not isinstance(stages, list) or not stages:
            raise ValueError("exp['stages'] must be a nonempty list")
        names = set()
        valid_labels = set(self.model.component_labels)
        validated = []
        for stage in stages:
            if not isinstance(stage, dict):
                raise TypeError("Every training stage must be a mapping")
            name = stage.get("name")
            if not isinstance(name, str) or not name or name in names:
                raise ValueError("Training stage names must be unique nonempty strings")
            names.add(name)
            trainable = stage.get("trainable_components")
            if not isinstance(trainable, list) or not trainable:
                raise ValueError(f"Stage {name!r} has no trainable components")
            if len(set(trainable)) != len(trainable):
                raise ValueError(f"Stage {name!r} repeats a trainable component")
            unknown = set(trainable) - valid_labels
            if unknown:
                raise KeyError(
                    f"Stage {name!r} names unknown components: {sorted(unknown)}"
                )
            epochs = stage.get("epochs")
            if type(epochs) is not int or epochs <= 0:
                raise ValueError(f"Stage {name!r} epochs must be positive")
            rates = stage.get("learning_rates")
            if not isinstance(rates, dict) or set(rates) != set(trainable):
                raise ValueError(
                    f"Stage {name!r} learning_rates must exactly match "
                    "trainable_components"
                )
            for label, rate in rates.items():
                _finite_positive(rate, f"Stage {name!r} learning rate for {label}")
            monitor = stage.get("monitor", self._default_monitor())
            if monitor not in self._supported_monitors():
                raise ValueError(
                    f"Unsupported monitor {monitor!r}; expected one of "
                    f"{sorted(self._supported_monitors())}"
                )
            optimizer = stage.get("optimizer", "Adam").lower()
            if optimizer not in {"adam", "adamw"}:
                raise ValueError(f"Unsupported optimizer {optimizer!r}")
            if "grad_clip_norm" in stage:
                _finite_positive(stage["grad_clip_norm"], "grad_clip_norm")
            validated.append(dict(stage))
        return validated

    def _fingerprint(self):
        architecture = {
            label: self.model.architecture_signature(label)
            for label in self.model.component_labels
        }
        component_configs = {
            label: _without_volatile(item["exp"])
            for label, item in self.spec["components"].items()
        }
        payload = {
            "recipe": _without_volatile(self.exp),
            "controller": self._resolved_controller_document(),
            "component_configs": component_configs,
            "sensor_config": _without_volatile(self.model.sensor_exp),
            "robot_config": _without_volatile(self.robot_exp),
            "architecture": architecture,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def _initial_status(self):
        return {
            "schema_version": SCHEMA_VERSION,
            "fingerprint": self.fingerprint,
            "state": "not_started",
            "current_stage_index": None,
            "current_stage": None,
            "epoch": 0,
            "epochs": None,
            "latest_checkpoint": None,
            "best_metric": None,
            "error": None,
            "updated_at": _now(),
            "stages": [
                {
                    "name": stage["name"], "state": "pending",
                    "epoch": 0, "epochs": stage["epochs"],
                    "monitor": stage.get("monitor", self._default_monitor()),
                    "best_metric": None,
                }
                for stage in self.stages
            ],
        }

    def _read_or_create_status(self):
        if not self.status_path.is_file():
            status = self._initial_status()
            _atomic_json(self.status_path, status)
            return status
        with self.status_path.open(encoding="utf-8") as handle:
            status = json.load(handle)
        if status.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported training recipe status schema")
        if status.get("fingerprint") != self.fingerprint:
            raise ValueError(
                "Training recipe configuration changed; refusing to resume "
                "in the existing data directory"
            )
        return status

    def _write_status(self, **changes):
        self.status.update(changes)
        self.status["updated_at"] = _now()
        _atomic_json(self.status_path, self.status)

    def _resolved_controller_document(self):
        return {
            "name": self.spec["name"],
            "topological_order": self.spec["topological_order"],
            "components": {
                label: {
                    key: value for key, value in item.items()
                    if key in {"experiment", "run", "type", "inputs", "outputs"}
                }
                for label, item in self.spec["components"].items()
            },
            "connections": self.spec["connections"],
        }

    def _source_path(self, label):
        item = self.spec["components"][label]
        if item["type"] in {"SP_VAE", "SP_CNN"}:
            return model_file(self.model.sensor_exp)
        component_exp = item["exp"]
        return Path(component_exp["data_dir"]) / component_exp["model_file"]

    def _materialize_sources(self):
        sources_dir = self.data_dir / "sources"
        sources_dir.mkdir(parents=True, exist_ok=True)
        initial = self.exp["initial_states"]
        if set(initial) != set(self.model.component_labels):
            raise ValueError(
                "initial_states must name exactly the trainable controller "
                f"components: {list(self.model.component_labels)}"
            )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "fingerprint": self.fingerprint,
            "components": {},
        }
        for label in self.model.component_labels:
            mode = initial[label].get("mode")
            destination = sources_dir / f"{label}.pth"
            if mode == "configured":
                source = self._source_path(label)
                if not source.is_file():
                    raise FileNotFoundError(
                        f"Configured source for {label!r} does not exist: {source}"
                    )
                payload = torch.load(
                    source, map_location=Config().runtime["device"],
                    weights_only=True,
                )
                self.model.load_component_state(
                    label, payload,
                    full_vae=self.spec["components"][label]["type"] == "SP_VAE",
                )
                if self.spec["components"][label]["type"] == "SP_VAE":
                    # The controller deliberately gathers only encoder.* and
                    # fc_mu.* from the full VAE source checkpoint.
                    _atomic_torch(
                        destination, self.model.component_state_dict(label)
                    )
                else:
                    temporary = destination.with_suffix(".pth.tmp")
                    shutil.copy2(source, temporary)
                    temporary.replace(destination)
                original = str(source)
                original_sha256 = _checksum(source)
            elif mode == "random":
                _atomic_torch(destination, self.model.component_state_dict(label))
                original = None
                original_sha256 = None
            else:
                raise ValueError(
                    f"initial_states[{label!r}].mode must be configured or random"
                )
            manifest["components"][label] = {
                "mode": mode, "original": original,
                "original_sha256": original_sha256,
                "gathered": str(destination.relative_to(self.data_dir)),
                "sha256": _checksum(destination),
                "architecture": self.model.architecture_signature(label),
            }
        _atomic_json(self.source_manifest_path, manifest)
        resolved_path = self.data_dir / "resolved_controller.yaml"
        temporary = resolved_path.with_suffix(".yaml.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self._resolved_controller_document(), handle, sort_keys=False)
        temporary.replace(resolved_path)

    def _load_materialized_sources(self):
        if not self.source_manifest_path.is_file():
            raise FileNotFoundError("Source manifest is missing")
        with self.source_manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("fingerprint") != self.fingerprint:
            raise ValueError("Source manifest fingerprint does not match recipe")
        for label in self.model.component_labels:
            entry = manifest["components"][label]
            path = self.data_dir / entry["gathered"]
            if not path.is_file() or _checksum(path) != entry["sha256"]:
                raise RuntimeError(f"Gathered source state is missing or changed: {path}")
            payload = torch.load(
                path, map_location=Config().runtime["device"], weights_only=True
            )
            self.model.load_component_state(label, payload)

    def initialize(self):
        if self.source_manifest_path.is_file():
            self._load_materialized_sources()
        else:
            self._write_status(state="initializing", error=None)
            self._materialize_sources()
            self._write_status(state="not_started")
        completed = [
            index for index, stage in enumerate(self.status["stages"])
            if stage["state"] == "completed"
        ]
        if completed:
            index = completed[-1]
            path = self._stage_dir(index) / "best_model.pth"
            if not path.is_file():
                raise FileNotFoundError(
                    f"Completed stage is missing its best model: {path}"
                )
            payload = torch.load(
                path, map_location=Config().runtime["device"], weights_only=False
            )
            if payload.get("fingerprint") != self.fingerprint:
                raise ValueError("Completed stage fingerprint does not match recipe")
            self.model.load_state_dict(payload["model_state_dict"])

    def _stage_dir(self, index):
        safe_name = self.stages[index]["name"].replace("/", "_")
        return self.data_dir / "stages" / f"{index:02d}_{safe_name}"

    def _optimizer_for(self, stage):
        groups = []
        for label in stage["trainable_components"]:
            groups.append({
                "params": list(self.model.component_module(label).parameters()),
                "lr": float(stage["learning_rates"][label]),
                "component": label,
            })
        optimizer_class = (
            torch.optim.Adam if stage.get("optimizer", "Adam").lower() == "adam"
            else torch.optim.AdamW
        )
        weight_decay = stage.get("weight_decay", 0.0)
        if (
            not isinstance(weight_decay, (int, float))
            or not math.isfinite(weight_decay) or weight_decay < 0
        ):
            raise ValueError("weight_decay must be a finite nonnegative number")
        optimizer = optimizer_class(groups, weight_decay=float(weight_decay))
        scheduler = None
        scheduler_config = stage.get("scheduler")
        if scheduler_config is not None:
            if scheduler_config.get("class") != "ReduceLROnPlateau":
                raise ValueError("Only ReduceLROnPlateau scheduler is supported")
            factor = scheduler_config.get("factor", 0.5)
            patience = scheduler_config.get("patience", 5)
            if (
                not isinstance(factor, (int, float))
                or not math.isfinite(factor) or not 0 < factor < 1
            ):
                raise ValueError("Scheduler factor must be between zero and one")
            if type(patience) is not int or patience < 0:
                raise ValueError("Scheduler patience must be a nonnegative integer")
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=float(factor), patience=patience,
            )
        return optimizer, scheduler

    def _checkpoint_payload(self, stage_index, next_epoch, best_metric):
        payload = {
            "schema_version": SCHEMA_VERSION,
            "fingerprint": self.fingerprint,
            "stage_index": stage_index,
            "stage_name": self.stages[stage_index]["name"],
            "next_epoch": next_epoch,
            "best_metric": best_metric,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            payload["cuda_random_state"] = torch.cuda.get_rng_state_all()
        if self._scheduler is not None:
            payload["scheduler_state_dict"] = self._scheduler.state_dict()
        return payload

    def _save_epoch_checkpoint(self, stage_index, next_epoch, best_metric):
        checkpoint_dir = self._stage_dir(stage_index) / "checkpoints"
        path = checkpoint_dir / f"epoch_{next_epoch:06d}.pth"
        _atomic_torch(
            path, self._checkpoint_payload(stage_index, next_epoch, best_metric)
        )
        keep = self.exp.get("keep_checkpoints", 3)
        if type(keep) is not int or keep < 1:
            raise ValueError("keep_checkpoints must be at least one")
        checkpoints = sorted(checkpoint_dir.glob("epoch_*.pth"))
        for old in checkpoints[:-keep]:
            old.unlink()
        return path

    def _save_diagnostic_checkpoint(self):
        if self._active_stage_index is None or self._optimizer is None:
            return None
        best = self.status.get("best_metric")
        if best is None:
            best = float("inf")
        path = self._stage_dir(self._active_stage_index) / "diagnostic_failure.pth"
        _atomic_torch(
            path,
            self._checkpoint_payload(
                self._active_stage_index, self.status.get("epoch", 0), best
            ),
        )
        return path

    def _restore_checkpoint(self, path, stage_index):
        payload = torch.load(
            path, map_location=Config().runtime["device"], weights_only=False
        )
        if payload.get("fingerprint") != self.fingerprint:
            raise ValueError("Checkpoint fingerprint does not match recipe")
        if payload.get("stage_index") != stage_index:
            raise ValueError("Checkpoint belongs to a different training stage")
        self.model.load_state_dict(payload["model_state_dict"])
        self._optimizer.load_state_dict(payload["optimizer_state_dict"])
        if self._scheduler is not None and "scheduler_state_dict" in payload:
            self._scheduler.load_state_dict(payload["scheduler_state_dict"])
        random.setstate(payload["python_random_state"])
        np.random.set_state(payload["numpy_random_state"])
        torch.set_rng_state(payload["torch_random_state"])
        if torch.cuda.is_available() and "cuda_random_state" in payload:
            torch.cuda.set_rng_state_all(payload["cuda_random_state"])
        return payload["next_epoch"], payload["best_metric"]

    def _train_epoch(self, loader, grad_clip):
        self.model.train(True)
        total = 0.0
        samples = 0
        device = Config().runtime["device"]
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            self._optimizer.zero_grad(set_to_none=True)
            loss, _ = self._loss_and_prediction(self.model(images), targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("Training loss is not finite")
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    [
                        parameter for parameter in self.model.parameters()
                        if parameter.requires_grad
                    ],
                    grad_clip,
                )
            self._optimizer.step()
            count = targets.size(0)
            total += loss.item() * count
            samples += count
        if samples == 0:
            raise ValueError("Training loader produced no samples")
        return total / samples

    def _validate_epoch(self, loader):
        self.model.eval()
        totals = {"validation_loss": 0.0, "validation_mse": 0.0,
                  "validation_mae": 0.0}
        samples = 0
        device = Config().runtime["device"]
        with torch.no_grad():
            for images, targets in loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                loss, prediction = self._loss_and_prediction(
                    self.model(images), targets
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Validation loss is not finite")
                count = targets.size(0)
                totals["validation_loss"] += loss.item() * count
                totals["validation_mse"] += torch.mean(
                    (prediction - targets) ** 2
                ).item() * count
                totals["validation_mae"] += torch.mean(
                    torch.abs(prediction - targets)
                ).item() * count
                samples += count
        if samples == 0:
            raise ValueError("Validation loader produced no samples")
        metrics = {key: value / samples for key, value in totals.items()}
        if "validation_nll" in self._supported_monitors():
            metrics["validation_nll"] = metrics["validation_loss"]
        return metrics

    def _append_metrics(self, record):
        records = []
        if self.metrics_path.is_file():
            with self.metrics_path.open(encoding="utf-8") as handle:
                records = [json.loads(line) for line in handle if line.strip()]
        key = (record["stage_index"], record["epoch"])
        records = [
            item for item in records
            if (item["stage_index"], item["epoch"]) != key
        ]
        records.append(record)
        temporary = self.metrics_path.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for item in records:
                handle.write(json.dumps(item, sort_keys=True) + "\n")
            handle.flush()
        temporary.replace(self.metrics_path)

    def _make_dataloaders(self):
        return self._dataloader_factory(
            self.exp, self.model.sensor_exp, self.robot_exp,
            self.model.sequence_length, self._output_size(),
        )

    def _start_state_for_stage(self, index):
        stage = self.stages[index]
        self.model.set_trainable(stage["trainable_components"])
        self._optimizer, self._scheduler = self._optimizer_for(stage)
        self._active_stage_index = index
        latest = self.status.get("latest_checkpoint")
        if (
            self.status.get("current_stage_index") == index
            and latest is not None and Path(latest).is_file()
        ):
            return self._restore_checkpoint(Path(latest), index)
        return 0, float("inf")

    def train(self, cont_train=True, *, progress_callback=None):
        if not isinstance(cont_train, bool):
            raise TypeError("cont_train must be boolean")
        if not cont_train and self.status["state"] != "not_started":
            raise ValueError(
                "cont_train=False cannot overwrite an existing recipe run; "
                "use a new exp/run data directory"
            )
        if self.status["state"] == "completed":
            self.load()
            if progress_callback is not None:
                progress_callback(self.status)
            return self.model
        try:
            self.initialize()
            training_loader, validation_loader = self._make_dataloaders()
            start_stage = next(
                (
                    index for index, item in enumerate(self.status["stages"])
                    if item["state"] != "completed"
                ),
                len(self.stages),
            )
            for index in range(start_stage, len(self.stages)):
                stage = self.stages[index]
                next_epoch, best = self._start_state_for_stage(index)
                self.status["stages"][index]["state"] = "running"
                self._write_status(
                    state="running", current_stage_index=index,
                    current_stage=stage["name"], epoch=next_epoch,
                    epochs=stage["epochs"], best_metric=(None if math.isinf(best) else best),
                    error=None,
                )
                if progress_callback is not None:
                    progress_callback(self.status)
                for epoch in range(next_epoch, stage["epochs"]):
                    train_nll = self._train_epoch(
                        training_loader, stage.get("grad_clip_norm")
                    )
                    validation = self._validate_epoch(validation_loader)
                    monitor = stage.get("monitor", self._default_monitor())
                    validation_metric = validation[monitor]
                    if self._scheduler is not None:
                        self._scheduler.step(validation_metric)
                    if validation_metric < best:
                        best = validation_metric
                        _atomic_torch(
                            self._stage_dir(index) / "best_model.pth",
                            self._checkpoint_payload(index, epoch + 1, best),
                        )
                    path = self._save_epoch_checkpoint(index, epoch + 1, best)
                    record = {
                        "timestamp": _now(), "stage_index": index,
                        "stage": stage["name"], "epoch": epoch + 1,
                        "train_loss": train_nll,
                        self._training_metric_name(): train_nll,
                        **validation,
                        "learning_rates": {
                            group["component"]: group["lr"]
                            for group in self._optimizer.param_groups
                        },
                    }
                    self._append_metrics(record)
                    stage_status = self.status["stages"][index]
                    stage_status.update(epoch=epoch + 1, best_metric=best)
                    self._write_status(
                        epoch=epoch + 1, best_metric=best,
                        latest_checkpoint=str(path),
                    )
                    if progress_callback is not None:
                        progress_callback(self.status)
                best_path = self._stage_dir(index) / "best_model.pth"
                if not best_path.is_file():
                    raise RuntimeError(f"Stage produced no best checkpoint: {best_path}")
                best_payload = torch.load(
                    best_path, map_location=Config().runtime["device"],
                    weights_only=False,
                )
                self.model.load_state_dict(best_payload["model_state_dict"])
                self.status["stages"][index]["state"] = "completed"
                _atomic_json(
                    self._stage_dir(index) / "completed.json",
                    {"stage": stage["name"], "best_metric": best, "completed_at": _now()},
                )
                self._write_status(latest_checkpoint=None)
            self.export()
            self._write_status(
                state="completed", current_stage_index=None,
                current_stage=None, epoch=0, epochs=None,
                latest_checkpoint=None, best_metric=None,
            )
            if hasattr(self.exp, "done"):
                self.exp.done()
            if progress_callback is not None:
                progress_callback(self.status)
            return self.model
        except KeyboardInterrupt:
            index = self.status.get("current_stage_index")
            if index is not None:
                self.status["stages"][index]["state"] = "interrupted"
            self._write_status(state="interrupted", error="KeyboardInterrupt")
            if progress_callback is not None:
                progress_callback(self.status)
            raise
        except Exception as error:
            index = self.status.get("current_stage_index")
            if index is not None:
                self.status["stages"][index]["state"] = "failed"
            diagnostic = None
            diagnostic_error = None
            try:
                diagnostic = self._save_diagnostic_checkpoint()
            except Exception as save_error:
                diagnostic_error = (
                    f"{type(save_error).__name__}: {save_error}"
                )
            self._write_status(
                state="failed",
                error="".join(traceback.format_exception_only(type(error), error)).strip(),
                diagnostic_checkpoint=(
                    None if diagnostic is None else str(diagnostic)
                ),
                diagnostic_error=diagnostic_error,
            )
            if progress_callback is not None:
                progress_callback(self.status)
            raise

    def load(self, snapshot_no=0):
        if type(snapshot_no) is not int or snapshot_no < 0:
            raise ValueError("snapshot_no must be a nonnegative integer")
        if snapshot_no == 0 and (self.data_dir / self.exp["model_file"]).is_file():
            payload = torch.load(
                self.data_dir / self.exp["model_file"],
                map_location=Config().runtime["device"], weights_only=True,
            )
            if payload.get("fingerprint") != self.fingerprint:
                raise ValueError("Controller bundle fingerprint does not match recipe")
            self.model.load_state_dict(payload["model_state_dict"])
            return self.model
        stage_index = self.status.get("current_stage_index")
        if stage_index is None:
            raise FileNotFoundError("No current stage checkpoint is available")
        if snapshot_no == 0:
            path = self.status.get("latest_checkpoint")
            if path is None:
                raise FileNotFoundError("No latest checkpoint is recorded")
            path = Path(path)
        else:
            path = self._stage_dir(stage_index) / "checkpoints" / f"epoch_{snapshot_no:06d}.pth"
        self.model.set_trainable(self.stages[stage_index]["trainable_components"])
        self._optimizer, self._scheduler = self._optimizer_for(self.stages[stage_index])
        self._restore_checkpoint(path, stage_index)
        return self.model

    def save(self, new_snapshot=True):
        if type(new_snapshot) is not bool:
            raise TypeError("new_snapshot must be boolean")
        if self._active_stage_index is None or self._optimizer is None:
            raise RuntimeError("There is no active training stage to save")
        epoch = self.status["epoch"]
        best = self.status.get("best_metric")
        if best is None:
            best = float("inf")
        path = self._save_epoch_checkpoint(
            self._active_stage_index, epoch, best
        )
        self._write_status(latest_checkpoint=str(path))
        return path

    def export(self):
        components_dir = self.data_dir / "components"
        components_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "fingerprint": self.fingerprint,
            "components": {},
        }
        component_states = {}
        for label in self.model.component_labels:
            path = components_dir / f"{label}.pth"
            state = self.model.component_state_dict(label)
            _atomic_torch(path, state)
            component_states[label] = state
            manifest["components"][label] = {
                "file": str(path.relative_to(self.data_dir)),
                "sha256": _checksum(path),
                "architecture": self.model.architecture_signature(label),
            }
        _atomic_json(self.data_dir / "component_manifest.json", manifest)
        if not self.source_manifest_path.is_file():
            raise FileNotFoundError("Cannot export without a source manifest")
        with self.source_manifest_path.open(encoding="utf-8") as handle:
            source_manifest = json.load(handle)
        controller_ref = self.exp["controller"]
        bundle = {
            "schema_version": SCHEMA_VERSION,
            "fingerprint": self.fingerprint,
            "controller": {
                "experiment": controller_ref.get("exp", "robot_controller"),
                "run": controller_ref["run"],
            },
            "resolved_controller": self._resolved_controller_document(),
            "architectures": {
                label: self.model.architecture_signature(label)
                for label in self.model.component_labels
            },
            "component_state_dicts": component_states,
            "model_state_dict": self.model.state_dict(),
            "provenance": {
                "source_manifest": source_manifest,
                "stages": _plain(self.status["stages"]),
            },
            "exported_at": _now(),
        }
        path = self.data_dir / self.exp["model_file"]
        _atomic_torch(path, bundle)
        return path


class StagedCNNMLPTrainingRecipe(StagedRobotControllerTrainingRecipe):
    """Staged deterministic behavior cloning for an SP_CNN--MLP graph."""

    def _create_training_model(self, spec):
        from robot_controller.cnn_mlp_training_model import CNNMLPTrainingModel

        return CNNMLPTrainingModel(spec)

    def _default_monitor(self):
        return "validation_mse"

    def _supported_monitors(self):
        return {"validation_loss", "validation_mse", "validation_mae"}

    def _training_metric_name(self):
        return "train_mse"

    def _loss_and_prediction(self, output, targets):
        if not isinstance(output, torch.Tensor) or output.shape != targets.shape:
            raise ValueError(
                "CNN--MLP output and normalized action target must have "
                "identical shapes"
            )
        return F.mse_loss(output, targets), output

    def _output_size(self):
        return self.model.output_size


def create_training_recipe(exp, **kwargs):
    """Construct the recipe selected by ``exp['class']``."""
    if exp["class"] == "StagedRobotControllerTrainingRecipe":
        return StagedRobotControllerTrainingRecipe(exp, **kwargs)
    if exp["class"] == "StagedCNNMLPTrainingRecipe":
        return StagedCNNMLPTrainingRecipe(exp, **kwargs)
    raise ValueError(f"Unknown robot-controller training recipe {exp['class']!r}")
