#!/usr/bin/env python3
# coding: utf-8
"""
检查文本条件是否对视频解码产生实质影响（correct vs shuffled）
"""

import argparse
import os
import sys
from typing import Dict, Any, List

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import TrainingConfig  # noqa: E402
from data_loader import MultimodalDataLoader  # noqa: E402
from train import create_model  # noqa: E402
from utils import load_manifest  # noqa: E402


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((pred - target) ** 2)
    if mse.item() == 0:
        return float("inf")
    return float(10 * torch.log10(1.0 / (mse + 1e-8)))


def main() -> None:
    parser = argparse.ArgumentParser(description="验证文本条件对视频解码的影响")
    parser.add_argument("--model-path", type=str, required=True, help="模型权重路径")
    parser.add_argument("--data-dir", type=str, required=True, help="数据目录")
    parser.add_argument("--manifest", type=str, default=None, help="manifest 路径（默认优先 v2）")
    parser.add_argument("--snr-db", type=float, default=-5.0, help="测试 SNR（低SNR更能体现差异）")
    parser.add_argument("--batch-size", type=int, default=2, help="batch 大小（>=2以便打乱）")
    parser.add_argument("--num-batches", type=int, default=5, help="测试多少个 batch")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = TrainingConfig()
    config.data_dir = args.data_dir
    config.batch_size = args.batch_size
    # 路线1默认开关
    config.use_text_guidance_image = False
    config.use_text_guidance_video = True
    config.enforce_text_condition = True
    config.condition_margin_weight = 0.0  # 在此脚本中仅做对比，不反向传播

    # 选择 manifest
    if args.manifest:
        manifest_path = args.manifest
    else:
        candidate = os.path.join(args.data_dir, "train_manifest_v2.json")
        manifest_path = candidate if os.path.exists(candidate) else os.path.join(args.data_dir, "train_manifest.json")
    data_list = load_manifest(manifest_path)
    if not data_list:
        raise FileNotFoundError(f"无法加载 manifest: {manifest_path}")

    loader_mgr = MultimodalDataLoader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=False,
        image_size=config.img_size,
        max_text_length=config.max_text_length,
        max_video_frames=config.max_video_frames,
        allow_missing_modalities=False,
        strict_mode=True,
        required_modalities=("video", "text"),
        seed=config.seed,
    )
    dataset = loader_mgr.create_dataset(data_list, is_train=False)
    dataloader = loader_mgr.create_dataloader(dataset, shuffle=False)

    model = create_model(config).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    metrics: List[Dict[str, float]] = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= args.num_batches:
                break
            inputs = batch["inputs"]
            targets = batch["targets"]
            text = inputs.get("text_input")
            video = inputs.get("video_input")
            attention_mask = inputs.get("text_attention_mask")
            if text is None or video is None or text.shape[0] < 2:
                continue  # 需要至少两个样本才能打乱
            text = text.to(device)
            video = video.to(device)
            attention_mask = attention_mask.to(device) if attention_mask is not None else None
            target_video = targets["video"].to(device)

            results = model(
                text_input=text,
                image_input=inputs.get("image_input", None).to(device) if inputs.get("image_input") is not None else None,
                video_input=video,
                text_attention_mask=attention_mask,
                snr_db=args.snr_db,
            )
            if "text_encoded" not in results or "video_transmitted" not in results:
                continue
            shuffle_idx = torch.randperm(results["text_encoded"].shape[0], device=device)
            shuffled_context = results["text_encoded"][shuffle_idx]
            shuffled_video = model.video_decoder(
                results["video_transmitted"],
                results["video_guide"],
                semantic_context=shuffled_context,
                reset_state=True,
            )
            correct = results["video_decoded"]
            l1_correct = torch.mean(torch.abs(correct - target_video)).item()
            l1_shuffled = torch.mean(torch.abs(shuffled_video - target_video)).item()
            delta = l1_shuffled - l1_correct
            psnr_correct = psnr(correct, target_video)
            psnr_shuffled = psnr(shuffled_video, target_video)
            gate_mean = results.get("video_semantic_gate_mean")
            metrics.append(
                {
                    "l1_correct": l1_correct,
                    "l1_shuffled": l1_shuffled,
                    "delta_l1": delta,
                    "psnr_correct": psnr_correct,
                    "psnr_shuffled": psnr_shuffled,
                    "gate_mean": gate_mean if gate_mean is not None else 0.0,
                }
            )

    if not metrics:
        print("未能收集到有效样本（可能batch过小或缺少文本/视频）。")
        return
    mean_delta = np.mean([m["delta_l1"] for m in metrics])
    mean_gate = np.mean([m["gate_mean"] for m in metrics])
    print("=" * 60)
    print(f"样本数: {len(metrics)}, SNR={args.snr_db}")
    print(f"L1(correct) 平均: {np.mean([m['l1_correct'] for m in metrics]):.6f}")
    print(f"L1(shuffled) 平均: {np.mean([m['l1_shuffled'] for m in metrics]):.6f}")
    print(f"ΔL1 (shuffled - correct): {mean_delta:.6f} (正值越大表明文本作用显著)")
    print(f"PSNR(correct) 平均: {np.mean([m['psnr_correct'] for m in metrics]):.3f} dB")
    print(f"PSNR(shuffled) 平均: {np.mean([m['psnr_shuffled'] for m in metrics]):.3f} dB")
    print(f"语义门控均值: {mean_gate:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
