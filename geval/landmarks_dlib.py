#!/usr/bin/env python
import argparse
import glob
import os
import shutil

import dlib
import numpy as np
from tqdm import tqdm

from evaluations.utils import is_image_file

DATA_PATH = "evaluations/ffhq_lm_data.npz"

# NOTE: download detector from
# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_dir', type=str, help='Directory with images.')
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--real', action="store_true")
    parser.add_argument('--std-mul', type=float, default=3.0)
    parser.add_argument('--max-num', type=int, default=None)
    parser.add_argument('--save-leak', action="store_true")
    args = parser.parse_args()

    detector = dlib.get_frontal_face_detector()
    # detector = dlib.cnn_face_detection_model_v1("evaluations/mmod_human_face_detector.dat")
    predictor = dlib.shape_predictor("evaluations/shape_predictor_68_face_landmarks.dat")

    if args.real:
        lm_data = compute_ffhq_stats("evaluations/ffhq-dataset-v2.json")
        np.savez(DATA_PATH, **lm_data)
        return

    with open(DATA_PATH, "rb") as f:
        lm_data = np.load(f)
        c0 = lm_data["c0"] * (args.image_size / 1024)
        e2e_angle = lm_data["e2e"]
        e2m_angle = lm_data["e2m"]
        qsize = lm_data["s"] * (args.image_size / 1024)

        c0_mean = c0.mean(axis=0)
        c0_std = c0.std(axis=0)
        e2e_angle_mean = e2e_angle.mean()
        e2e_angle_std = e2e_angle.std()
        e2m_angle_mean = e2m_angle.mean()
        e2m_angle_std = e2m_angle.std()
        qsize_mean = qsize.mean()
        qsize_std = qsize.std()
    print(f" c0: {c0_mean} {chr(177)} {c0_std}")
    print(f"e2e: {e2e_angle_mean} {chr(177)} {e2e_angle_std}")
    print(f"e2m: {e2m_angle_mean} {chr(177)} {e2m_angle_std}")
    print(f"  s: {qsize_mean} {chr(177)} {qsize_std}")

    no_face, leak, align = 0, 0, 0

    paths = glob.glob(os.path.join(args.image_dir, "**"))
    paths = tuple(sorted(filter(is_image_file, paths)))
    if args.max_num is not None:
        paths = paths[:args.max_num]
    pbar = tqdm(paths, desc=f"# leak: {leak}")
    for path in pbar:
        ret, points, num_faces = detect_landmarks(path, detector, predictor)
        if not ret:
            no_face += 1
            continue
        elif num_faces > 1:
            print(f"Warning: {path} has more than one face.")

        c0, e2e_angle, e2m_angle, qsize = compute(points)

        c_diff = np.abs(c0-c0_mean)
        e2e_diff = np.abs(e2e_angle-e2e_angle_mean)
        e2m_diff = np.abs(e2m_angle-e2e_angle_mean)
        s_diff = np.abs(qsize-qsize_mean)
        if (
            c_diff[0] > args.std_mul * c0_std[0]
            or c_diff[1] > args.std_mul * c0_std[1]
            or e2e_diff > args.std_mul * e2e_angle_std
            or e2m_diff > args.std_mul * e2m_angle_std
            or s_diff > args.std_mul * qsize_std
        ):
            leak += 1
            if args.save_leak:
                os.makedirs("leak", exist_ok=True)  # TODO: use args
                shutil.copy(path, "leak")
        else:
            align += 1
        pbar.set_description(f"# leak: {leak}")

    print("")
    print("===== Summary =====")
    print("#detection failed:", no_face)
    print("#leak:", leak)
    print("#align:", align)


def angle_between(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'"""
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    return np.arccos(np.dot(v1, v2) / (v1_norm * v2_norm))


def detect_landmarks(path, detector, predictor, verbose=False):
    image = dlib.load_rgb_image(path)

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = detector(image, 1)
    if verbose:
        print(f"Number of faces detected: {len(dets)}")
    if not dets:
        return False, None, 0

    # Get the landmarks/parts for the face in box d.
    if isinstance(dets, dlib.mmod_rectangles):
        det = dets[0].rect
    else:
        det = dets[0]
    shape = predictor(image, det)
    points = np.asarray([[p.x, p.y] for p in shape.parts()])
    return True, points, len(dets)


def compute(face_landmarks):
    # Align function from FFHQ dataset pre-processing step
    # https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py

    lm = np.asarray(face_landmarks)
    lm_chin          = lm[0  : 17]  # left-right
    lm_eyebrow_left  = lm[17 : 22]  # left-right
    lm_eyebrow_right = lm[22 : 27]  # left-right
    lm_nose          = lm[27 : 31]  # top-down
    lm_nostrils      = lm[31 : 36]  # top-down
    lm_eye_left      = lm[36 : 42]  # left-clockwise
    lm_eye_right     = lm[42 : 48]  # left-clockwise
    lm_mouth_outer   = lm[48 : 60]  # left-clockwise
    lm_mouth_inner   = lm[60 : 68]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left     = np.mean(lm_eye_left, axis=0)
    eye_right    = np.mean(lm_eye_right, axis=0)
    eye_avg      = (eye_left + eye_right) * 0.5
    eye_to_eye   = eye_right - eye_left
    mouth_left   = lm_mouth_outer[0]
    mouth_right  = lm_mouth_outer[6]
    mouth_avg    = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c0 = eye_avg + eye_to_mouth * 0.1

    # quad = np.stack([c0 - x - y, c0 - x + y, c0 + x + y, c0 + x - y])
    qsize = np.hypot(*x) * 2

    e2e_angle = angle_between(eye_to_eye, [1, 0])
    e2m_angle = angle_between(eye_to_mouth, [0, 1])

    return c0, e2e_angle, e2m_angle, qsize


def compute_dir(directory, image_size, detector, predictor):
    c0s, e2e_angles, e2m_angles, sizes = [], [], [], []
    no_face, success = 0, 0

    paths = glob.glob(os.path.join(directory, "**"))
    paths = tuple(filter(is_image_file, paths))
    for path in tqdm(paths):
        ret, points, num_faces = detect_landmarks(path, detector, predictor)
        if not ret:
            no_face += 1
            continue
        elif num_faces > 1:
            print(f"Warning: {path} has more than one face.")


        # NOTE: Landmarks are sorted by the confidence of their bounding boxes.
        c0, e2e_angle, e2m_angle, qsize = compute(points)

        c0s.append(c0)
        e2e_angles.append(e2e_angle)
        e2m_angles.append(e2m_angle)
        sizes.append(qsize)
        success += 1

    c0s = np.stack(c0s)
    e2e_angles = np.stack(e2e_angles)
    e2m_angles = np.stack(e2m_angles)
    sizes = np.stack(sizes)

    print(f"#detection failed: {no_face}")
    print("#success:", success)
    print(f" c0: {c0s.mean(axis=0)} {chr(177)} {c0s.std(axis=0)}")
    print(f"e2e: {e2e_angles.mean()} {chr(177)} {e2e_angles.std()}")
    print(f"e2m: {e2m_angles.mean()} {chr(177)} {e2m_angles.std()}")
    print(f"  s: {sizes.mean()} {chr(177)} {sizes.std()}")
    return dict(
        c0=c0s,
        e2e=e2e_angles,
        e2m=e2m_angles,
        s=sizes,
        success=success,
        no_face=no_face,
    )


def compute_ffhq_stats(ffhq_json):
    import json

    with open(ffhq_json, "r") as f:
        ffhq_data = json.load(f)

    c0s, e2e_angles, e2m_angles, sizes = [], [], [], []

    for i in range(70000):
        face_landmarks = ffhq_data[str(i)]['image']['face_landmarks']

        # NOTE: Landmarks are sorted by the confidence of their bounding boxes.
        c0, e2e_angle, e2m_angle, qsize = compute(face_landmarks)

        c0s.append(c0)
        e2e_angles.append(e2e_angle)
        e2m_angles.append(e2m_angle)
        sizes.append(qsize)

    c0s = np.stack(c0s)
    e2e_angles = np.stack(e2e_angles)
    e2m_angles = np.stack(e2m_angles)
    sizes = np.stack(sizes)

    print(f" c0: {c0s.mean(axis=0)} {chr(177)} {c0s.std(axis=0)}")
    print(f"e2e: {e2e_angles.mean()} {chr(177)} {e2e_angles.std()}")
    print(f"e2m: {e2m_angles.mean()} {chr(177)} {e2m_angles.std()}")
    print(f"  s: {sizes.mean()} {chr(177)} {sizes.std()}")
    return dict(
        c0=c0s,
        e2e=e2e_angles,
        e2m=e2m_angles,
        s=sizes,
    )


if __name__ == "__main__":
    main()
