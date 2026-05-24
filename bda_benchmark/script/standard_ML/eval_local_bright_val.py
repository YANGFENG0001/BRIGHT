import argparse
import csv
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DATASET_ROOT = PROJECT_ROOT / "dataset"
if str(DATASET_ROOT) not in sys.path:
    sys.path.insert(0, str(DATASET_ROOT))
MODEL_ROOT = PROJECT_ROOT / "model"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))
UTIL_ROOT = PROJECT_ROOT / "util_func"
if str(UTIL_ROOT) not in sys.path:
    sys.path.insert(0, str(UTIL_ROOT))

import imutils
from DamageFormer import DamageFormer
from DeepLabV3Plus import DeepLabV3Plus
from SiamAttnUNet import SiamAttnUNet
from SiamCRNN import SiamCRNN
from UNet import UNet
from metrics import Evaluator


DISASTER_EVENTS = [
    "turkey-earthquake",
    "hawaii-wildfire",
    "morocco-earthquake",
    "haiti-earthquake",
    "la_palma-volcano",
    "congo-volcano",
    "beirut-explosion",
    "bata-explosion",
    "libya-flood",
    "noto-earthquake",
    "marshall-wildfire",
    "ukraine-conflict",
    "myanmar-hurricane",
    "mexico-hurricane",
]

DISASTER_TYPES = [
    "earthquake",
    "wildfire",
    "volcano",
    "explosion",
    "flood",
    "conflict",
    "hurricane",
]

COLOR_MAP = {
    0: (255, 255, 255),
    1: (70, 181, 121),
    2: (228, 189, 139),
    3: (182, 70, 69),
}


class LocalBrightDamageDataset(Dataset):
    """Read the locally prepared BRIGHT split layout.

    Expected split layout:
      images_pre/<id>.tif
      images_post_sar/<id>.tif
      targets_cvt_damage/<id>.tif
      test_ids.txt or another id list
    """

    def __init__(self, split_dir, id_list_path=None, limit=None):
        self.split_dir = Path(split_dir)
        id_list_path = Path(id_list_path) if id_list_path else self.split_dir / "test_ids.txt"
        self.ids = [line.strip() for line in id_list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if limit is not None:
            self.ids = self.ids[:limit]
        self.pre_dir = self.split_dir / "images_pre"
        self.post_dir = self.split_dir / "images_post_sar"
        self.target_dir = self.split_dir / "targets_cvt_damage"

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        data_id = self.ids[index]
        pre = imageio.imread(self.pre_dir / f"{data_id}.tif")[:, :, :3]
        post = imageio.imread(self.post_dir / f"{data_id}.tif")
        target = imageio.imread(self.target_dir / f"{data_id}.tif")

        if post.ndim == 2:
            post = np.stack((post,) * 3, axis=-1)
        else:
            post = post[:, :, :3]

        pre = np.transpose(imutils.normalize_img(pre), (2, 0, 1))
        post = np.transpose(imutils.normalize_img(post), (2, 0, 1))
        target = np.asarray(target, dtype=np.int64)

        loc_target = target.copy()
        loc_target[loc_target == 2] = 1
        loc_target[loc_target == 3] = 1
        return pre, post, loc_target, target, data_id


def instantiate_model(model_name):
    model_name = model_name.lower()
    if model_name == "unet":
        return UNet(in_channels=6, num_classes=4), "single_concat"
    if model_name == "deeplabv3plus":
        return DeepLabV3Plus(in_channels=6, num_classes=4, pretrained=False), "single_concat"
    if model_name == "siamattnunet":
        return SiamAttnUNet(in_channels=3, num_classes=4), "single_siamese"
    if model_name == "siamcrnn":
        return SiamCRNN(pretrained=False), "decoupled"
    if model_name == "damageformer":
        return DamageFormer(pretrained=False), "decoupled"
    raise ValueError(f"Unsupported model: {model_name}")


def strip_module_prefix(state_dict):
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return OrderedDict((key.removeprefix("module."), value) for key, value in state_dict.items())


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "model_state_dict", "net"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                state_dict = checkpoint[key]
                break
    model.load_state_dict(strip_module_prefix(state_dict), strict=True)


def harmonic_mean(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[values > 0]
    if values.size == 0:
        return 0.0
    return float(values.size / np.sum(1.0 / values))


def evaluator_summary(evaluator):
    return {
        "pixel_accuracy": float(evaluator.Pixel_Accuracy()),
        "mean_iou": float(evaluator.Mean_Intersection_over_Union()),
        "iou_per_class": [float(x) for x in evaluator.Intersection_over_Union()],
    }


def add_group_metrics(group_dict, group_name, label, pred):
    for name, evaluator in group_dict.items():
        if name in group_name:
            evaluator.add_batch(label, pred)
            return


def colorize(prediction):
    color_map_img = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)
    for cls, color in COLOR_MAP.items():
        color_map_img[prediction == cls] = color
    return color_map_img


def evaluate(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset = LocalBrightDamageDataset(args.split_dir, args.id_list_path, args.limit)
    loader = DataLoader(dataset, batch_size=1, num_workers=args.num_workers, drop_last=False)

    model, model_kind = instantiate_model(args.model)
    load_checkpoint(model, args.model_path, device)
    model.to(device)
    model.eval()

    final_evaluator = Evaluator(num_class=4)
    loc_evaluator = Evaluator(num_class=2)
    clf_evaluator = Evaluator(num_class=4)
    event_evaluators = {event: Evaluator(num_class=4) for event in DISASTER_EVENTS}
    type_evaluators = {event_type: Evaluator(num_class=4) for event_type in DISASTER_TYPES}

    output_dir = Path(args.output_dir)
    pred_original_dir = output_dir / args.model / "original"
    pred_colored_dir = output_dir / args.model / "colored"
    if args.save_predictions:
        pred_original_dir.mkdir(parents=True, exist_ok=True)
        pred_colored_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for pre, post, loc_label, clf_label, data_id in tqdm(loader, desc=f"eval {args.model}"):
            pre = pre.to(device=device, dtype=torch.float32)
            post = post.to(device=device, dtype=torch.float32)
            data_id = data_id[0]

            if model_kind == "single_concat":
                logits = model(torch.cat([pre, post], dim=1))
                loc_logits = None
            elif model_kind == "single_siamese":
                logits = model(pre, post)
                loc_logits = None
            else:
                loc_logits, logits = model(pre, post)

            clf_pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            if loc_logits is not None:
                loc_pred = torch.argmax(loc_logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            else:
                loc_pred = (clf_pred > 0).astype(np.uint8)

            if args.final_mode == "mask" and model_kind == "decoupled":
                final_pred = (clf_pred * loc_pred).astype(np.uint8)
            else:
                final_pred = clf_pred

            loc_label = loc_label.squeeze(0).numpy()
            clf_label = clf_label.squeeze(0).numpy()

            final_evaluator.add_batch(clf_label, final_pred)
            loc_evaluator.add_batch(loc_label, loc_pred)
            clf_evaluator.add_batch(clf_label, clf_pred)
            add_group_metrics(event_evaluators, data_id, clf_label, final_pred)
            add_group_metrics(type_evaluators, data_id, clf_label, final_pred)

            if args.save_predictions:
                Image.fromarray(final_pred).save(pred_original_dir / f"{data_id}_building_damage.png")
                Image.fromarray(colorize(final_pred)).save(pred_colored_dir / f"{data_id}_building_damage.png")

    clf_f1_per_class = clf_evaluator.Damage_F1_score()
    result = {
        "model": args.model,
        "model_path": str(Path(args.model_path).resolve()),
        "split_dir": str(Path(args.split_dir).resolve()),
        "id_list_path": str(Path(args.id_list_path).resolve()) if args.id_list_path else str((Path(args.split_dir) / "test_ids.txt").resolve()),
        "num_samples": len(dataset),
        "device": str(device),
        "final_mode": args.final_mode,
        "overall": evaluator_summary(final_evaluator),
        "loc_f1": float(loc_evaluator.Pixel_F1_score()),
        "clf_f1_hmean": harmonic_mean(clf_f1_per_class),
        "clf_f1_per_damage_class": [float(x) for x in clf_f1_per_class],
        "per_event": {name: evaluator_summary(ev) for name, ev in event_evaluators.items() if ev.confusion_matrix.sum() > 0},
        "per_type": {name: evaluator_summary(ev) for name, ev in type_evaluators.items() if ev.confusion_matrix.sum() > 0},
        "confusion_matrix": final_evaluator.confusion_matrix.tolist(),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{args.model}_{args.final_mode}_metrics.json"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = output_dir / f"{args.model}_{args.final_mode}_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "samples", "final_oa_pct", "final_miou_pct", "f1loc_pct", "f1clf_hmean_pct", "iou_background_pct", "iou_intact_pct", "iou_damaged_pct", "iou_destroyed_pct"])
        ious = result["overall"]["iou_per_class"]
        writer.writerow([
            args.model,
            len(dataset),
            result["overall"]["pixel_accuracy"] * 100,
            result["overall"]["mean_iou"] * 100,
            result["loc_f1"] * 100,
            result["clf_f1_hmean"] * 100,
            ious[0] * 100,
            ious[1] * 100,
            ious[2] * 100,
            ious[3] * 100,
        ])

    print(json.dumps(result["overall"], indent=2, ensure_ascii=False))
    print(f"F1loc: {result['loc_f1'] * 100:.2f}%")
    print(f"F1clf(hmean): {result['clf_f1_hmean'] * 100:.2f}%")
    print(f"Saved metrics: {json_path}")
    print(f"Saved summary: {csv_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate official BRIGHT baselines on a local prepared BRIGHT split.")
    parser.add_argument("--model", required=True, choices=["UNet", "DeepLabV3Plus", "SiamAttnUNet", "SiamCRNN", "DamageFormer"])
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--split_dir", default=r"D:\PycharmProjects\SegEarth-OV-3\data\bright_damage\val")
    parser.add_argument("--id_list_path", default=None)
    parser.add_argument("--output_dir", default="outputs/bright_val_eval")
    parser.add_argument("--device", default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--final_mode", choices=["clf", "mask"], default="clf")
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N ids. Intended for smoke tests.")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
