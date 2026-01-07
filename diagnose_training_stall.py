"""
诊断训练损失几乎不变的潜在原因。

执行内容：
1) 检查梯度累计步数是否大于每个 epoch 的迭代次数。
2) 统计 batch 内各模态是否为全零（会被模型视为无效输入）。
3) 计算一次前向 + 损失，检查 NaN/Inf 或损失恒为 0 的情况。
4) （可选）检查一次反向传播后的梯度是否为 0。
"""

import argparse
import os
import torch

from config import TrainingConfig
from data_loader import MultimodalDataLoader, collate_multimodal_batch
from multimodal_jscc import MultimodalJSCC
from losses import MultimodalLoss
from utils import load_manifest, seed_torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="诊断训练损失停滞问题")
    parser.add_argument("--data-dir", type=str, required=True, help="数据目录")
    parser.add_argument("--train-manifest", type=str, default=None, help="训练清单路径（相对 data-dir）")
    parser.add_argument("--batch-size", type=int, default=None, help="批次大小")
    parser.add_argument("--num-batches", type=int, default=3, help="检查的 batch 数")
    parser.add_argument("--device", type=str, default=None, help="设备，例如 cuda 或 cpu")
    parser.add_argument("--check-grad", action="store_true", help="是否检查梯度是否为 0")
    return parser.parse_args()


def summarize_zero_input(batch_tensor: torch.Tensor) -> float:
    if batch_tensor is None:
        return 1.0
    if batch_tensor.numel() == 0:
        return 1.0
    return (batch_tensor.view(batch_tensor.size(0), -1).abs().sum(dim=1) == 0).float().mean().item()


def main() -> None:
    args = parse_args()
    seed_torch(42)

    config = TrainingConfig()
    config.data_dir = args.data_dir
    if args.batch_size is not None:
        config.batch_size = args.batch_size

    if args.device is not None:
        config.device = torch.device(args.device)
    else:
        config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.train_manifest:
        config.train_manifest = os.path.join(config.data_dir, args.train_manifest)
    else:
        config.train_manifest = os.path.join(config.data_dir, "train_manifest.json")

    train_data_list = load_manifest(config.train_manifest)
    if not train_data_list:
        raise RuntimeError(f"训练数据清单为空或文件不存在: {config.train_manifest}")

    iterations_per_epoch = (len(train_data_list) + config.batch_size - 1) // config.batch_size
    print("=== 配置检查 ===")
    print(f"训练样本数: {len(train_data_list)}")
    print(f"batch_size: {config.batch_size}")
    print(f"gradient_accumulation_steps: {config.gradient_accumulation_steps}")
    print(f"iterations_per_epoch: {iterations_per_epoch}")
    if config.gradient_accumulation_steps > iterations_per_epoch:
        print("警告: gradient_accumulation_steps 大于每个 epoch 的迭代数，可能整轮不会更新参数。")

    data_loader_manager = MultimodalDataLoader(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        num_workers=0,
        shuffle=True,
        image_size=config.img_size,
        max_text_length=config.max_text_length,
        max_video_frames=config.max_video_frames,
        prefetch_factor=2,
    )
    train_dataset = data_loader_manager.create_dataset(train_data_list)
    train_loader = data_loader_manager.create_dataloader(train_dataset, shuffle=True)

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
    ).to(config.device)

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
    ).to(config.device)

    print("\n=== Batch 检查 ===")
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= args.num_batches:
            break

        inputs = batch["inputs"]
        targets = batch["targets"]
        attention_mask = batch.get("attention_mask", None)

        text_input = inputs.get("text_input", None)
        image_input = inputs.get("image_input", None)
        video_input = inputs.get("video_input", None)

        if text_input is not None:
            text_input = text_input.to(config.device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(config.device)
        if image_input is not None:
            image_input = image_input.to(config.device)
        if video_input is not None:
            video_input = video_input.to(config.device)

        device_targets = {}
        for key, value in targets.items():
            if value is not None:
                device_targets[key] = value.to(config.device)

        zero_text = summarize_zero_input(text_input)
        zero_image = summarize_zero_input(image_input)
        zero_video = summarize_zero_input(video_input)

        print(
            f"[Batch {batch_idx + 1}] "
            f"zero_text_ratio={zero_text:.2f}, "
            f"zero_image_ratio={zero_image:.2f}, "
            f"zero_video_ratio={zero_video:.2f}"
        )

        results = model(
            text_input=text_input,
            image_input=image_input,
            video_input=video_input,
            text_attention_mask=attention_mask,
            snr_db=config.snr_db,
        )
        loss_dict = loss_fn(
            predictions=results,
            targets=device_targets,
            attention_mask=attention_mask,
        )
        total_loss = loss_dict["total_loss"]

        status = "OK"
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            status = "NaN/Inf"
        elif total_loss.item() == 0.0:
            status = "Zero"

        print(
            f"[Batch {batch_idx + 1}] total_loss={total_loss.item():.6f} ({status})"
        )

        if args.check_grad:
            model.zero_grad(set_to_none=True)
            total_loss.backward()
            total_grad = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_grad += p.grad.detach().abs().sum().item()
            if total_grad == 0.0:
                print(f"[Batch {batch_idx + 1}] 警告: 梯度总和为 0")
            else:
                print(f"[Batch {batch_idx + 1}] 梯度总和={total_grad:.4f}")

        del results, loss_dict, total_loss


if __name__ == "__main__":
    main()
