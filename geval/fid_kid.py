#!/usr/bin/env python
import argparse
import csv
import glob
import os

import numpy as np
import torch
import torch.utils.data as data
from cleanfid import fid, resize
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from vdmpp.image_datasets import is_image_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ffhq')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--source_dir', type=str, help='Directory with training images')
    parser.add_argument('--target_dir', type=str, help='Directory with generated samples')
    parser.add_argument('--csv_file', type=str, default="evaluations/metrics.csv")
    # clean-FID options
    parser.add_argument('--mode', type=str, default='clean')
    parser.add_argument('--clip_fid', action='store_true')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()

    # Use deterministic algorithms for reproducibility.
    torch.use_deterministic_algorithms(True)
    # Disable TF32 on Ampere GPUs for better reproducibility.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    fid_kwargs = {
        "mode": args.mode,
        "model_name": "clip_vit_b_32" if args.clip_fid else "inception_v3",
        "num_workers": args.num_workers,
        "batch_size": args.batch_size,
    }
    if (
        not args.clip_fid
        and args.dataset == "ffhq" and args.image_size in (256, 1024)
    ):
        dataset_name = args.dataset
        dataset_split = "trainval70k"
    else:
        dataset_name = compute_ref_stats(
            args.dataset, args.image_size, args.source_dir, fid_kwargs
        )
        dataset_split = "custom"
    fid_kwargs.update(dict(
        dataset_name=dataset_name,
        dataset_res=args.image_size,
        dataset_split=dataset_split,
    ))

    fid_value = compute_fid(args.target_dir, **fid_kwargs)
    fid_kwargs.pop("model_name")
    kid_value = compute_kid(args.target_dir, **fid_kwargs) * 1e3

    with open(args.csv_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([dataset_name, args.target_dir, fid_value, kid_value])
    print("======================")
    print(f"FID: {round(fid_value, 3)}")
    print(f"KID (x 10^3): {round(kid_value, 3)}")
    print("======================")


def resize_and_save_image(image_path, output_path, img_size):
    image = Image.open(image_path).convert("RGB")
    resized_image = image.resize((img_size, img_size), resample=Image.BICUBIC)
    resized_image.save(output_path)


def compute_ref_stats(dataset, image_size, ori_dir, cleanfid_kwargs):
    """
    Create directory with bibubic-resized images of img_size (with PIL)
    """
    ref_dir = f'datasets/{dataset}-{image_size}'  # output directory for resized images
    stats_name = ref_dir.split('/')[-1]

    # if directory doesn't exist, create a new directory with resized images
    if not fid.test_stats_exists(stats_name, mode='clean'):
        assert os.path.exists(ori_dir), f'{ori_dir} does not exist.'
        image_files = glob.glob(os.path.join(ori_dir, '**'), recursive=True)
        image_files = list(filter(is_image_file, image_files))
        assert len(image_files) > 0, f'{ori_dir} does not contain any image files.'

        os.makedirs(ref_dir, exist_ok=True)
        fn_resize = resize.make_resizer("PIL", True, "bicubic", (image_size,)*2)
        dataset = resize.FolderResizer(image_files, ref_dir, fn_resize)
        loader = data.DataLoader(dataset, batch_size=128, num_workers=8)
        for _ in tqdm(loader, desc="Resizing images"):
            pass

        # save custom stats
        fid.make_custom_stats(stats_name, ref_dir, **cleanfid_kwargs)
    else:
        print(f"Custom stats ({stats_name}) already exist. "
              "Skipping computation...")
    return stats_name


def fid_npzfile(npzfile, dataset_name, dataset_res, dataset_split,
                model=None, mode="clean", model_name="inception_v3", num_workers=8,
                batch_size=128, device=torch.device("cuda"), verbose=True,
                custom_image_tranform=None, custom_fn_resize=None):
    # Load reference FID statistics (download if needed)
    ref_mu, ref_sigma = fid.get_reference_statistics(
        dataset_name, dataset_res,
        mode=mode, model_name=model_name, seed=0, split=dataset_split)

    np_feats = get_npz_features(npzfile, model, num_workers=num_workers,
                                batch_size=batch_size, device=device,
                                mode=mode, description=f"FID {npzfile} : ", verbose=verbose,
                                custom_image_tranform=custom_image_tranform,
                                custom_fn_resize=custom_fn_resize)
    mu = np.mean(np_feats, axis=0)
    sigma = np.cov(np_feats, rowvar=False)
    fid_value = fid.frechet_distance(mu, sigma, ref_mu, ref_sigma)
    return fid_value


def get_npz_features(npzfile, model=None, num_workers=4,
                     batch_size=128, device=torch.device("cuda"),
                     mode="clean", custom_fn_resize=None,
                     description="", verbose=True,
                     custom_image_tranform=None):
    assert model is not None
    # wrap the images in a dataloader for parallelizing the resize operation
    dataset = NpzDataset(npzfile, mode)
    if custom_image_tranform is not None:
        dataset.custom_image_tranform=custom_image_tranform
    if custom_fn_resize is not None:
        dataset.fn_resize = custom_fn_resize

    dataloader = data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        drop_last=False, num_workers=num_workers)

    # collect all inception features
    l_feats = []
    if verbose:
        pbar = tqdm(dataloader, desc=description)
    else:
        pbar = dataloader

    for batch in pbar:
        l_feats.append(fid.get_batch_features(batch, model, device))
    np_feats = np.concatenate(l_feats)
    return np_feats


def compute_fid(
        sample_path, mode="clean", model_name="inception_v3", num_workers=12,
        batch_size=32, device=torch.device("cuda"), dataset_name="FFHQ",
        dataset_res=1024, dataset_split="train",
        custom_feat_extractor=None, verbose=True,
        custom_image_tranform=None, custom_fn_resize=None, use_dataparallel=True):
    if os.path.isfile(sample_path) and sample_path.endswith(".npz"):
        fid_fn = fid_npzfile
    elif os.path.isdir(sample_path):
        fid_fn = fid.fid_folder
    else:
        raise ValueError(f"Invalid sample path: {sample_path}")

    # build the feature extractor based on the mode and the model to be used
    if custom_feat_extractor is not None:
        feat_model = custom_feat_extractor
    elif model_name=="inception_v3":
        feat_model = fid.build_feature_extractor(mode, device, use_dataparallel=use_dataparallel)
    elif model_name=="clip_vit_b_32":
        from cleanfid.clip_features import CLIP_fx, img_preprocess_clip
        clip_fx = CLIP_fx("ViT-B/32", device=device)
        feat_model = clip_fx
        custom_fn_resize = img_preprocess_clip

    fid_value = fid_fn(
        sample_path, dataset_name, dataset_res, dataset_split,
        feat_model, mode, model_name, num_workers,
        batch_size, device, verbose,
        custom_image_tranform, custom_fn_resize)
    return fid_value


def compute_kid(
        sample_path, mode="clean", num_workers=8, batch_size=64,
        device=torch.device("cuda"), dataset_name="FFHQ",
        dataset_res=1024, dataset_split="train",
        verbose=True, use_dataparallel=True):
    # build the feature extractor based on the mode and the model to be used
    feat_model = fid.build_feature_extractor(mode, device, use_dataparallel=use_dataparallel)

    # Load reference FID statistics (download if needed)
    ref_feats = fid.get_reference_statistics(
        dataset_name, dataset_res,
        mode=mode, seed=0, split=dataset_split, metric="KID")
    if os.path.isfile(sample_path) and sample_path.endswith(".npz"):
        np_feats = get_npz_features(
                sample_path, feat_model, num_workers=num_workers,
                batch_size=batch_size, device=device,
                mode=mode, description=f"KID {sample_path} : ", verbose=verbose)
    elif os.path.isdir(sample_path):
        np_feats = fid.get_folder_features(
                sample_path, feat_model, num_workers=num_workers,
                batch_size=batch_size, device=device,
                mode=mode, description=f"KID {sample_path} : ", verbose=verbose)
    else:
        raise ValueError(f"Invalid sample path: {sample_path}")

    kid_value = fid.kernel_distance(ref_feats, np_feats)
    return kid_value


class NpzDataset(data.Dataset):
    def __init__(self, npzfile, mode):
        super().__init__()
        with open(npzfile, "rb") as f:
            self.arr = np.load(f)["arr_0"]
        if self.arr.ndim != 4 or self.arr.shape[0] < 1:
            raise ValueError(f"Invalid npzfile: {npzfile}")

        self.fn_resize = resize.build_resizer(mode)
        self.custom_image_tranform = lambda x: x
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return self.arr.shape[0]

    def __getitem__(self, index):
        img_np = self.arr[index]

        # apply a custom image transform before resizing the image to 299x299
        img_np = self.custom_image_tranform(img_np)
        # fn_resize expects a np array and returns a np array
        img_resized = self.fn_resize(img_np)

        # to_tensor() converts to [0,1] only if input in uint8
        if img_resized.dtype == "uint8":
            img_t = self.to_tensor(np.array(img_resized))*255
        elif img_resized.dtype == "float32":
            img_t = self.to_tensor(img_resized)

        return img_t


if __name__ == "__main__":
    main()
