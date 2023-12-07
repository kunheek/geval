#!/usr/bin/env python
import argparse
import csv
import os

import numpy as np
import torch
from face_alignment import FaceAlignment, LandmarksType
from PIL import Image
from tqdm import tqdm

from evaluations.utils import get_image_files, ImageDataset

DATA_PATH = "evaluations/ffhq{}_{}_data.npz"

# NOTE: download detector from
# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_dir', type=str, help='Directory with images.')
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--lm_type', type=str, default="2d")
    parser.add_argument('--real', action="store_true")
    parser.add_argument('--std_mul', type=float, default=3.0)
    parser.add_argument('--max_num', type=int, default=None)
    parser.add_argument('--save_leak', action="store_true")
    parser.add_argument('--csv', type=str, default="alignment.csv")
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    if args.lm_type == "2d":
        lm_type = LandmarksType.TWO_D
    elif args.lm_type == "2.5d":
        lm_type = LandmarksType.TWO_HALF_D
    elif args.lm_type == "3d":
        lm_type = LandmarksType.THREE_D
    else:
        raise ValueError(f"Unknown landmarks type: {lm_type}")
    fa = FaceAlignment(lm_type, flip_input=False)

    data_path = DATA_PATH.format(args.image_size, args.lm_type)

    if args.real:
        lm_data = compute_dir(args.image_dir, args.image_size, fa)
        np.savez(data_path, **lm_data)
        return

    with open(data_path, "rb") as f:
        lm_data = np.load(f)
        c0_mean = lm_data["c0"].mean(axis=0)
        c0_std = lm_data["c0"].std(axis=0)
        e2e_angle_mean = lm_data["e2e"].mean()
        e2e_angle_std = lm_data["e2e"].std()
        e2m_angle_mean = lm_data["e2m"].mean()
        e2m_angle_std = lm_data["e2m"].std()
        qsize_mean = lm_data["s"].mean()
        qsize_std = lm_data["s"].std()
    print(f" c0: {c0_mean} {chr(177)} {c0_std}")
    print(f"e2e: {e2e_angle_mean} {chr(177)} {e2e_angle_std}")
    print(f"e2m: {e2m_angle_mean} {chr(177)} {e2m_angle_std}")
    print(f"  s: {qsize_mean} {chr(177)} {qsize_std}")


    no_face, leak, align = 0, 0, 0

    dataset = ImageDataset(args.image_dir, None)
    pbar = tqdm(dataset, desc=f"# fail: {no_face}, # leak: {leak}")
    for image in pbar:
        image = image.astype(np.uint8)
        face_landmarks = fa.get_landmarks_from_image(image)
        if not face_landmarks:
            no_face += 1
            continue

        c0, e2e_angle, e2m_angle, qsize = compute(face_landmarks[0])

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
                Image.fromarray(image).save(f"leak/{leak:05d}.png")
        else:
            align += 1
        pbar.set_description(f"# fail: {no_face}, # leak: {leak}")

    print("")
    print("===== Summary =====")
    print("#detection failed:", no_face)
    print("#leak:", leak)
    print("#align:", align)
    print("===================")
    # If file not exists, create it and write header
    if not os.path.isfile(args.csv):
        with open(args.csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                "image_path", "image_size", "failed", "leak", "align",
            ])

    with open(args.csv, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            args.image_dir, args.image_size, no_face, leak, align
        ])

def open_image_as_np(path, image_size):
    image = Image.open(path).convert("RGB")
    image = image.resize((image_size, image_size), Image.BICUBIC)
    image = np.asarray(image)
    return image


def angle_between(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'"""
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    return np.arccos(np.dot(v1, v2) / (v1_norm * v2_norm))


def compute(face_landmarks):
    # Align function from FFHQ dataset pre-processing step
    # https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py

    lm = np.asarray(face_landmarks)
    lm_chin          = lm[0  : 17, :2]  # left-right
    lm_eyebrow_left  = lm[17 : 22, :2]  # left-right
    lm_eyebrow_right = lm[22 : 27, :2]  # left-right
    lm_nose          = lm[27 : 31, :2]  # top-down
    lm_nostrils      = lm[31 : 36, :2]  # top-down
    lm_eye_left      = lm[36 : 42, :2]  # left-clockwise
    lm_eye_right     = lm[42 : 48, :2]  # left-clockwise
    lm_mouth_outer   = lm[48 : 60, :2]  # left-clockwise
    lm_mouth_inner   = lm[60 : 68, :2]  # left-clockwise

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


def compute_dir(directory, image_size, fa):
    c0s, e2e_angles, e2m_angles, sizes = [], [], [], []
    no_face, success = 0, 0

    paths = get_image_files(directory, None)
    for path in tqdm(paths):
        image = open_image_as_np(path, image_size)

        face_landmarks = fa.get_landmarks_from_image(image)
        if not face_landmarks:
            no_face += 1
            continue

        # NOTE: Landmarks are sorted by the confidence of their bounding boxes.
        c0, e2e_angle, e2m_angle, qsize = compute(face_landmarks[0])

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


if __name__ == "__main__":
    main()
