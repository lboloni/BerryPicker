import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import yaml

sys.path.extend([
    str(pathlib.Path(__file__).parents[2]),
    str(pathlib.Path(__file__).parents[2] / "demonstration"),
])

from import_aloha import AlohaImportError, import_aloha


class TestAlohaImport(unittest.TestCase):
    @staticmethod
    def _write_robot_positions(source, rows=13, synchronized=True):
        leader = np.arange(rows * 7, dtype=np.float32).reshape(rows, 7)
        follower = leader + 100.0
        if isinstance(synchronized, bool):
            synchronized = np.full(rows, synchronized, dtype=bool)
        np.savez(
            source / "robot_positions.npz",
            pair_1_leader=leader,
            pair_1_follower=follower,
            synced=synchronized,
            fps=np.asarray(60),
        )
        return leader, follower

    @staticmethod
    def _write_video(path, rows=13):
        size = (16, 12)
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), 60, size
        )
        if not writer.isOpened():
            raise unittest.SkipTest("OpenCV mp4v video writer is unavailable")
        try:
            for index in range(rows):
                frame = np.full((size[1], size[0], 3), index * 10, np.uint8)
                writer.write(frame)
        finally:
            writer.release()

    def test_imports_every_sixth_row_into_separate_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            source = root / "source"
            source.mkdir()
            leader, follower = self._write_robot_positions(source)
            self._write_video(source / "camera_222_rgb.mp4")
            self._write_video(source / "camera_111_rgb.mp4")
            source_files = sorted(path.name for path in source.iterdir())
            target = root / "target"

            result = import_aloha(source, target)

            self.assertEqual(result, target.resolve())
            self.assertEqual(
                sorted(path.name for path in source.iterdir()), source_files
            )
            with (target / "_metadata.yaml").open() as stream:
                metadata = yaml.safe_load(stream)
            with (target / "_action.yaml").open() as stream:
                actions = yaml.safe_load(stream)
            with (target / "_annotation.yaml").open() as stream:
                annotations = yaml.safe_load(stream)

            self.assertEqual(metadata["maxsteps"], 3)
            self.assertEqual(metadata["cameras"], [
                "aloha_111_rgb", "aloha_222_rgb",
            ])
            self.assertTrue(metadata["import"]["partial"])
            self.assertTrue(metadata["import"]["temporary"])
            self.assertEqual(metadata["import"]["source_frame_stride"], 6)
            self.assertEqual(
                actions[1]["aloha-pair-1-leader"],
                [float(value) for value in leader[6]],
            )
            self.assertEqual(
                annotations[2]["aloha"]["pair_1_follower"],
                [float(value) for value in follower[12]],
            )

            for camera in metadata["cameras"]:
                capture = cv2.VideoCapture(
                    str(target / f"video_{camera}.mp4")
                )
                try:
                    self.assertTrue(capture.isOpened())
                    self.assertEqual(
                        int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 3
                    )
                    self.assertAlmostEqual(
                        capture.get(cv2.CAP_PROP_FPS), 10.0, places=1
                    )
                finally:
                    capture.release()

    def test_rejects_target_inside_source(self):
        with tempfile.TemporaryDirectory() as directory:
            source = pathlib.Path(directory) / "source"
            source.mkdir()
            with self.assertRaisesRegex(AlohaImportError, "not inside"):
                import_aloha(source, source / "target")

    def test_rejects_unsynchronized_selected_row(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            source = root / "source"
            source.mkdir()
            synchronized = np.ones(13, dtype=bool)
            synchronized[6] = False
            self._write_robot_positions(source, synchronized=synchronized)
            target = root / "target"

            with self.assertRaisesRegex(
                AlohaImportError, r"not synchronized: \[6\]"
            ):
                import_aloha(source, target)

            self.assertFalse(target.exists())

    def test_removes_staging_directory_after_conversion_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            source = root / "source"
            source.mkdir()
            self._write_robot_positions(source)
            self._write_video(source / "camera_111_rgb.mp4")
            target = root / "target"

            with patch(
                "import_aloha._write_video",
                side_effect=AlohaImportError("write failed"),
            ):
                with self.assertRaisesRegex(AlohaImportError, "write failed"):
                    import_aloha(source, target)

            self.assertFalse(target.exists())
            self.assertEqual(
                [path for path in root.iterdir() if path.name != "source"], []
            )


if __name__ == "__main__":
    unittest.main()
