"""Fit a rectified auxiliary/target RGB camera pair from paired chessboard images."""
import cv2
import numpy as np


def calibrate_pair(source, target, corners, square_m):
    rows, cols = corners[1], corners[0]
    pattern = np.zeros((rows*cols,3), np.float32)
    pattern[:,:2] = np.indices((cols,rows)).T.reshape(-1,2)*square_m
    objects, points_a, points_b = [], [], []
    size = source[0].shape[1::-1]
    for a,b in zip(source,target,strict=True):
        ok_a, pa = cv2.findChessboardCornersSB(a, tuple(corners))
        ok_b, pb = cv2.findChessboardCornersSB(b, tuple(corners))
        if ok_a and ok_b:
            points_a.append(pa)
            points_b.append(pb)
            objects.append(pattern)
    _,ka,da,_,_ = cv2.calibrateCamera(objects,points_a,size,None,None)
    _,kb,db,_,_ = cv2.calibrateCamera(objects,points_b,size,None,None)
    rms,ka,da,kb,db,rot,trans,_,_ = cv2.stereoCalibrate(
        objects,points_a,points_b,ka,da,kb,db,size,flags=cv2.CALIB_FIX_INTRINSIC)
    pose = np.eye(4)
    pose[:3,:3], pose[:3,3] = rot, trans.ravel()
    return dict(K=ka.tolist(), target_K=kb.tolist(), target_from_source=pose.tolist(),
                source_distortion=da.ravel().tolist(), target_distortion=db.ravel().tolist(),
                calibration_rms_px=float(rms), paired_boards=len(objects))


def time_offset(target_events, source_events):
    """Corresponding observed LED transitions, seconds on each camera clock."""
    differences = np.asarray(target_events)-np.asarray(source_events)
    offset = np.median(differences)
    return float(offset), float(np.max(np.abs(differences-offset)))


def triangulate_landmarks(points_source, points_target, calibration):
    """Triangulate matched rectified-image landmarks and report both reprojection residuals."""
    a, b = np.asarray(points_source, float), np.asarray(points_target, float)
    pose = np.asarray(calibration["target_from_source"])
    source = np.c_[calibration["K"], np.zeros(3)]
    target = np.asarray(calibration["target_K"]) @ pose[:3]
    homogeneous = cv2.triangulatePoints(source, target, a.T, b.T)
    xyz = homogeneous/homogeneous[3]
    pa, pb = source@xyz, target@xyz
    return (xyz[:3].T, np.linalg.norm((pa[:2]/pa[2]).T-a, axis=1),
            np.linalg.norm((pb[:2]/pb[2]).T-b, axis=1))
