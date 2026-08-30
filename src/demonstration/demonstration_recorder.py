"""Collect synchronized demonstrations from configured participants."""

from __future__ import annotations

import pathlib
import time

import cv2

from demonstration.demonstration_participant import DemonstrationContext, create_participants


class DemonstrationRecorder:
    """Coordinate participant lifecycle and write one synchronized tick at a time."""

    def __init__(self, demonstration, participants, save_dir, tick_interval):
        if not participants:
            raise ValueError("A demonstration collection requires at least one participant")
        self.demonstration = demonstration
        self.participants = participants
        self.save_dir = pathlib.Path(save_dir)
        self.tick_interval = tick_interval
        self.counter = 0

    @classmethod
    def create(cls, collection_exp, machine_exp, demonstration, save_dir):
        participants = create_participants(collection_exp, machine_exp)
        return cls(
            demonstration=demonstration,
            participants=participants,
            save_dir=save_dir,
            tick_interval=collection_exp.get("tick_interval", 0.1),
        )

    def run(self):
        """Run until a participant requests termination, then save final metadata."""
        context = DemonstrationContext()
        started_participants = []
        try:
            for participant in self.participants:
                participant.start(context)
                started_participants.append(participant)
            previous_time = time.monotonic()
            while not context.stop_requested:
                started = time.monotonic()
                dt = started - previous_time
                previous_time = started
                for participant in self.participants:
                    participant.update(context, dt)
                if context.stop_requested:
                    continue
                samples = {
                    participant.name: participant.sample(context)
                    for participant in self.participants
                }
                self.save(samples)
                time.sleep(max(0.0, self.tick_interval - (time.monotonic() - started)))
        finally:
            for participant in reversed(started_participants):
                participant.stop(context)
            self.demonstration.save_metadata()

    def save(self, samples):
        """Persist one sample from every participant using a common timestep prefix."""
        if len(self.demonstration.actions) != self.counter:
            raise RuntimeError("Demonstration action count does not match recorder timestep")
        if self.demonstration.metadata["maxsteps"] != self.counter:
            raise RuntimeError("Demonstration maxsteps does not match recorder timestep")

        actions = [
            (name, sample.action)
            for name, sample in samples.items()
            if sample.action is not None
        ]
        if len(actions) != 1:
            raise RuntimeError(
                "Each timestep requires exactly one action producer, found "
                f"{[name for name, _ in actions]}"
            )
        self.demonstration.actions.append(actions[0][1])
        annotations = {}
        image_names = set()
        save_prefix = f"{self.counter:05d}"
        for name, sample in samples.items():
            if sample.telemetry:
                annotations[name] = sample.telemetry
            if sample.annotations:
                annotations.update(sample.annotations)
            for image_name, image in sample.images.items():
                if image_name in image_names:
                    raise RuntimeError(f"Duplicate image name in timestep: {image_name}")
                image_names.add(image_name)
                filename = self.save_dir / f"{save_prefix}_{image_name}.jpg"
                if not cv2.imwrite(str(filename), image):
                    raise RuntimeError(f"Unable to write demonstration image {filename}")
        self.demonstration.annotations.append(annotations)
        if image_names:
            self.demonstration.metadata["cameras"] = sorted(image_names)
        self.counter += 1
        self.demonstration.metadata["maxsteps"] = self.counter
