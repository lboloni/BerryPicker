import json
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from _experimental._2026_11_aloha_without_leader.hand_removal.clip import Clip
from _experimental._2026_11_aloha_without_leader.hand_removal.fixture import generate
from _experimental._2026_11_aloha_without_leader.hand_removal.reconstruct import masks, reconstruct, TEMPORAL, UNKNOWN


class OfflineTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = generate(Path(tmp.name)/"fork", "fork", "test", 3)
        self.clip = Clip(self.path)
        self.mask = masks(self.clip)
        self.mask.flags.writeable = False

    def test_no_edit_and_no_motion_evidence(self):
        rgb, source, _ = reconstruct(self.clip, np.zeros_like(self.mask), "reference")
        self.assertTrue(np.array_equal(rgb, self.clip.rgb))
        self.assertFalse(np.any(source))
        _, source, _ = reconstruct(self.clip, self.mask, "bidirectional")
        self.assertTrue(np.all(source[10][self.mask[10]]==UNKNOWN))

    def test_future_donors_and_original_pixels(self):
        rgb, source, donor = reconstruct(self.clip,self.mask,"bidirectional")
        index = np.arange(len(rgb))[:,None,None]
        self.assertTrue(np.any(np.logical_and(source==TEMPORAL, donor>index)))
        changed = np.any(rgb!=self.clip.rgb,axis=3)
        self.assertFalse(np.any(np.logical_and(changed,np.logical_not(self.mask))))
        _, past_source, past = reconstruct(self.clip,self.mask,"past")
        self.assertFalse(np.any(np.logical_and(past_source==TEMPORAL, past>=index)))

    def test_two_views_recover_hidden_scene(self):
        rgb, _, _ = reconstruct(self.clip,self.mask,"multiview")
        self.assertTrue(np.array_equal(rgb,self.clip.series("clean")))

    def test_sparse_keyframes_and_video_reader(self):
        from _experimental._2026_11_aloha_without_leader.hand_removal.clip import video
        s = json.loads(self.path.read_text())
        video(self.path.parent/"source.mp4",self.clip.rgb,12)
        s["rgb"]="source.mp4"
        tracks = s.pop("tracks")
        s["keyframes"]=[dict(frame=i, points=tracks[i]) for i in (0,9,15,23)]
        self.path.write_text(json.dumps(s))
        clip = Clip(self.path)
        _,source,_ = reconstruct(clip,masks(clip),"reference")
        self.assertTrue(np.any(source==2))

    def test_reader_export_keeps_action_rows_across_gaps(self):
        from types import SimpleNamespace
        from _experimental._2026_11_aloha_without_leader.hand_removal.clip import pngs, write_json
        from _experimental._2026_11_aloha_without_leader.hand_removal.demonstrations import export_view, import_view
        from demonstration.demonstration import Demonstration
        root = self.path.parent
        out_exp = SimpleNamespace(data_dir=lambda: root/"demos")
        src = SimpleNamespace(demo="teaching", metadata={}, actions=[{"taught-joints":[i]} for i in range(24)],
                              annotations=[{} for i in range(24)])
        pngs(root/"method"/"frames", self.clip.rgb)
        write_json(root/"method"/"frames.json",
                   [{"frame":i, "time":i/12, "unresolved":int(i==10)} for i in range(24)])
        names = export_view(src, "front", root/"method", out_exp, "clean", rejected=[2])
        self.assertEqual(names,["clean_000000","clean_000003","clean_000011"])
        result = Demonstration(out_exp,names[2])
        self.assertEqual(result.get_action(0,"taught-joints"),[11])
        path = import_view(result,"front",root/"roundtrip",dict(fps=12))
        self.assertEqual(json.loads(path.read_text())["frame_ids"], list(range(13)))

    def test_calibration_clock_offset(self):
        from _experimental._2026_11_aloha_without_leader.hand_removal.calibration import time_offset
        offset, residual = time_offset([1.02, 2.02, 3.02], [1.,2.,3.])
        self.assertAlmostEqual(offset,.02)
        self.assertLess(residual,1e-10)

    def test_articulated_input_and_temporal_abstention(self):
        # Independently tracked parts are composited by the common renderer.
        c = self.clip
        tracks = c.spec.pop("tracks")
        references = c.spec.pop("references")
        c.spec["parts"] = [dict(tracks=tracks, references=references)]
        rgb, source, _ = reconstruct(c, self.mask, "bidirectional")
        hidden = np.asarray(self.mask[10])
        self.assertTrue(np.all(source[10][hidden] == UNKNOWN))
        self.assertFalse(np.array_equal(rgb[10][hidden],c.plate[hidden]))

    def test_one_arm_teaching_without_motor_hardware(self):
        from unittest.mock import patch
        from demonstration.demonstration_participant import DemonstrationContext
        from _experimental._2026_11_aloha_without_leader.hand_guided_participant import HandGuidedParticipant
        context = DemonstrationContext()
        context.values["camera_key"] = ord("g")
        spec = dict(timeout_s=1, state_interval=.005, max_timesteps=4)
        p = HandGuidedParticipant("arm",spec,dict(robot_name="widowx"))
        with patch("_experimental._2026_11_aloha_without_leader.guided_reader.JointStateReader") as reader:
            reader.return_value.get_joints.return_value = [0.1]*7
            reader.return_value.stamp = 1.
            p.start(context)
            try:
                p.update(context,.1)
                sample = p.sample(context)
                self.assertEqual(sample.action["taught-joints"],[.1]*7)
                self.assertEqual(sample.annotations["gripper_event"],"grasp")
                self.assertEqual(sample.telemetry["state_samples"][0]["ros_stamp"],1.)
            finally:
                p.stop(context)
            reader.return_value.stop.assert_called_once()

    def test_sparse_calibrated_auxiliary_texture(self):
        from _experimental._2026_11_aloha_without_leader.hand_removal.reconstruct import project_auxiliary
        c = self.clip
        aux = c.spec["auxiliary"]
        # Three visible points on the same physical patch in the planar fixture.
        pts = np.float32([[30,30],[45,30],[40,45]])
        aux["landmarks"] = dict(target=[pts.tolist()]*len(c.rgb),
                                source=[(pts-[4,0]).tolist()]*len(c.rgb),
                                triangles=[[0,1,2]])
        aux["max_reprojection_px"] = .5
        rgb, valid, _ = next(project_auxiliary(c))
        self.assertTrue(valid[34,38])
        self.assertFalse(valid[3,3])
        self.assertTrue(np.array_equal(rgb[34,38], c.series("clean")[0,34,38]))

    def test_exposure_interpolation_preserves_event_indices(self):
        from types import SimpleNamespace
        from _experimental._2026_11_aloha_without_leader.hand_removal.demonstrations import joints_at_exposure
        rows = [dict(teaching_arm={"state_samples":[{"read_monotonic":float(i),"joints":[float(i)]*7}]},
                     cameras={"read_monotonic":{"front":i+.5}}) for i in range(3)]
        pose = joints_at_exposure(SimpleNamespace(annotations=rows),"front",0,1.1)
        self.assertEqual(pose,[[.5]*7,[1.5]*7,None])
