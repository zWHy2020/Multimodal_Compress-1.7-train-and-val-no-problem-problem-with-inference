"""
梯度诊断脚本：定位反向传播阶段的 NaN/Inf 与异常梯度。

功能：
- 执行单步前向 + loss + backward
- 检查每一层参数梯度是否包含 NaN/Inf
- 输出异常梯度的参数名、统计信息（min/max/mean/norm）

用法示例：
  python diagnose_gradients.py --data-dir /path/to/data \
    --manifest /path/to/train_manifest.json --max-batches 5 --batch-size 1

可选开关：
  --skip-video / --skip-image / --skip-text
  --disable-quant-noise / --disable-msssim / --disable-perceptual
  --disable-text-contrastive / --disable-video-text-contrastive
"""
import argparse
import json
from typing import Dict, Any, List

import torch

from config import TrainingConfig
from data_loader import MultimodalDataLoader
from losses import MultimodalLoss
from multimodal_jscc import MultimodalJSCC


MODALITIES = ("text", "image", "video")


def load_manifest(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _tensor_has_nan_or_inf(tensor: torch.Tensor) -> bool:
    return torch.isnan(tensor).any().item() or torch.isinf(tensor).any().item()


def _summarize_tensor(tensor: torch.Tensor) -> str:
    return (
        f"shape={tuple(tensor.shape)} dtype={tensor.dtype} "
        f"min={tensor.min().item():.4e} max={tensor.max().item():.4e} "
        f"mean={tensor.mean().item():.4e} norm={tensor.norm().item():.4e}"
    )


def _apply_skip_modalities(
    inputs: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    skip_text: bool,
    skip_image: bool,
    skip_video: bool,
) -> None:
    if skip_text:
        inputs["text_input"] = None
        inputs["text_attention_mask"] = None
        targets.pop("text", None)
    if skip_image:
        inputs["image_input"] = None
        targets.pop("image", None)
    if skip_video:
        inputs["video_input"] = None
        targets.pop("video", None)


def _build_model(config: TrainingConfig, device: torch.device) -> MultimodalJSCC:
    model = MultimodalJSCC(
        vocab_size=config.vocab_size,
        text_embed_dim=config.text_embed_dim,
        text_num_heads=config.text_num_heads,
        text_num_layers=config.text_num_layers,
        text_output_dim=config.text_output_dim,
        img_size=config.img_size,
        patch_size=config.patch_size,
        img_embed_dims=config.img_embed_dims,
        img_depths=config.img_depths,
        img_num_heads=config.img_num_heads,
        img_output_dim=config.img_output_dim,
        img_window_size=getattr(config, "img_window_size", 7),
        pretrained=getattr(config, "pretrained", False),
        freeze_encoder=getattr(config, "freeze_encoder", False),
        pretrained_model_name=getattr(config, "pretrained_model_name", "swin_tiny_patch4_window7_224"),
        video_hidden_dim=config.video_hidden_dim,
        video_num_frames=config.video_num_frames,
        video_use_optical_flow=config.video_use_optical_flow,
        video_use_convlstm=config.video_use_convlstm,
        video_output_dim=config.video_output_dim,
        channel_type=config.channel_type,
        snr_db=config.snr_db,
        use_quantization_noise=getattr(config, "use_quantization_noise", False),
        quantization_noise_range=getattr(config, "quantization_noise_range", 0.5),
    )
    return model.to(device)


def _build_loss_fn(config: TrainingConfig, device: torch.device) -> MultimodalLoss:
    loss_fn = MultimodalLoss(
        text_weight=config.text_weight,
        image_weight=config.image_weight,
        video_weight=config.video_weight,
        reconstruction_weight=config.reconstruction_weight,
        perceptual_weight=config.perceptual_weight,
        temporal_weight=config.temporal_weight,
        text_contrastive_weight=getattr(config, "text_contrastive_weight", 0.1),
        video_text_contrastive_weight=getattr(config, "video_text_contrastive_weight", 0.05),
        rate_weight=getattr(config, "rate_weight", 1e-4),
        temporal_consistency_weight=getattr(config, "temporal_consistency_weight", 0.02),
        discriminator_weight=getattr(config, "discriminator_weight", 0.01),
        use_adversarial=getattr(config, "use_adversarial", False),
        data_range=1.0,
    )
    return loss_fn.to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description="梯度诊断脚本")
    parser.add_argument("--data-dir", required=True, help="数据目录")
    parser.add_argument("--manifest", required=True, help="manifest JSON 路径")
    parser.add_argument("--batch-size", type=int, default=1, help="批次大小")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--max-batches", type=int, default=5, help="最多检查多少个 batch")
    parser.add_argument("--device", type=str, default=None, help="cpu/cuda")
    parser.add_argument("--skip-text", action="store_true", help="跳过文本模态")
    parser.add_argument("--skip-image", action="store_true", help="跳过图像模态")
    parser.add_argument("--skip-video", action="store_true", help="跳过视频模态")
    parser.add_argument("--disable-quant-noise", action="store_true", help="禁用量化噪声")
    parser.add_argument("--disable-msssim", action="store_true", help="禁用MS-SSIM")
    parser.add_argument("--disable-perceptual", action="store_true", help="禁用感知损失")
    parser.add_argument("--disable-text-contrastive", action="store_true", help="禁用文本对比损失")
    parser.add_argument("--disable-video-text-contrastive", action="store_true", help="禁用视频-文本对比损失")
    args = parser.parse_args()

    config = TrainingConfig()
    if args.disable_quant_noise:
        config.use_quantization_noise = False
    if args.disable_perceptual:
        config.perceptual_weight = 0.0
    if args.disable_text_contrastive:
        config.text_contrastive_weight = 0.0
    if args.disable_video_text_contrastive:
        config.video_text_contrastive_weight = 0.0

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    data_list = load_manifest(args.manifest)
    loader_mgr = MultimodalDataLoader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        image_size=config.img_size,
        max_text_length=config.max_text_length,
        max_video_frames=config.max_video_frames,
    )
    dataset = loader_mgr.create_dataset(data_list)
    loader = loader_mgr.create_dataloader(dataset, shuffle=False)

    model = _build_model(config, device)
    loss_fn = _build_loss_fn(config, device)
    if args.disable_msssim and hasattr(loss_fn, "image_loss_fn"):
        loss_fn.image_loss_fn.msssim_weight = 0.0

    model.train()

    print("=== 梯度诊断开始 ===")
    print(f"device: {device}")
    print(f"batch_size: {args.batch_size} | max_batches: {args.max_batches}")
    print(f"skip_text={args.skip_text} skip_image={args.skip_image} skip_video={args.skip_video}")

    for batch_idx, batch in enumerate(loader, start=1):
        inputs = batch.get("inputs", {})
        targets = batch.get("targets", {}).copy()
        attention_mask = batch.get("attention_mask", None)

        inputs = {
            "text_input": inputs.get("text_input"),
            "text_attention_mask": inputs.get("text_attention_mask"),
            "image_input": inputs.get("image_input"),
            "video_input": inputs.get("video_input"),
        }

        _apply_skip_modalities(
            inputs,
            targets,
            skip_text=args.skip_text,
            skip_image=args.skip_image,
            skip_video=args.skip_video,
        )

        if inputs.get("text_input") is not None:
            inputs["text_input"] = inputs["text_input"].to(device)
        if inputs.get("text_attention_mask") is not None:
            inputs["text_attention_mask"] = inputs["text_attention_mask"].to(device)
        if inputs.get("image_input") is not None:
            inputs["image_input"] = inputs["image_input"].to(device)
        if inputs.get("video_input") is not None:
            inputs["video_input"] = inputs["video_input"].to(device)

        device_targets = {
            key: value.to(device) for key, value in targets.items() if value is not None
        }

        model.zero_grad(set_to_none=True)
        results = model(
            text_input=inputs.get("text_input"),
            image_input=inputs.get("image_input"),
            video_input=inputs.get("video_input"),
            text_attention_mask=inputs.get("text_attention_mask"),
            snr_db=config.snr_db,
        )
        loss_dict = loss_fn(
            predictions=results,
            targets=device_targets,
            attention_mask=inputs.get("text_attention_mask"),
        )
        total_loss = loss_dict["total_loss"]

        if _tensor_has_nan_or_inf(total_loss):
            print(f"[NaN/Inf] loss 在 batch {batch_idx} 出现异常: {_summarize_tensor(total_loss)}")
            break

        total_loss.backward()

        nan_params = []
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            if _tensor_has_nan_or_inf(param.grad):
                nan_params.append(name)
                print(f"[NaN/Inf] grad: {name} -> {_summarize_tensor(param.grad)}")

        if nan_params:
            print(f"[停止] batch {batch_idx} 出现异常梯度，参数数={len(nan_params)}")
            break

        print(f"batch {batch_idx}: OK")

        if batch_idx >= args.max_batches:
            break

    print("=== 梯度诊断结束 ===")


if __name__ == "__main__":
    main()
