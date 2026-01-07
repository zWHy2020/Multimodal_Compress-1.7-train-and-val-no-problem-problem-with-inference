"""
检查 DataLoader 中是否存在全零输入的 batch 或样本。

用法:
  python check_zero_batches.py --data-dir /path/to/data --manifest /path/to/train_manifest.json
"""
import argparse
import json
from typing import Dict, Any, List

import torch

from data_loader import MultimodalDataLoader


MODALITIES = ["text", "image", "video"]


def load_manifest(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _is_all_zero(tensor: torch.Tensor) -> bool:
    return tensor.numel() > 0 and torch.all(tensor == 0).item()


def _count_zero_samples(batch_tensor: torch.Tensor) -> int:
    # 判断每个样本是否全零（按 batch 维度拆分）
    if batch_tensor.dim() == 1:
        # [B]
        return int(torch.sum(batch_tensor == 0).item())
    # 将非 batch 维度展平
    flat = batch_tensor.view(batch_tensor.size(0), -1)
    return int(torch.sum(torch.all(flat == 0, dim=1)).item())


def main() -> None:
    parser = argparse.ArgumentParser(description="检查全零 batch/样本")
    parser.add_argument("--data-dir", required=True, help="数据目录")
    parser.add_argument("--manifest", required=True, help="manifest JSON 路径")
    parser.add_argument("--batch-size", type=int, default=8, help="批次大小")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--max-batches", type=int, default=None, help="最多检查多少个 batch")
    args = parser.parse_args()

    data_list = load_manifest(args.manifest)
    loader_mgr = MultimodalDataLoader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    dataset = loader_mgr.create_dataset(data_list)
    loader = loader_mgr.create_dataloader(dataset, shuffle=False)

    total_batches = 0
    zero_batch_counts = {m: 0 for m in MODALITIES}
    zero_sample_counts = {m: 0 for m in MODALITIES}
    total_samples = 0

    for batch_idx, batch in enumerate(loader, start=1):
        inputs = batch.get("inputs", {})
        batch_size = None

        for m in MODALITIES:
            key = f"{m}_input"
            if key not in inputs:
                continue
            tensor = inputs[key]
            if batch_size is None:
                batch_size = tensor.size(0)
            if _is_all_zero(tensor):
                zero_batch_counts[m] += 1
            zero_sample_counts[m] += _count_zero_samples(tensor)

        if batch_size is None:
            batch_size = 0
        total_samples += batch_size
        total_batches += 1

        if args.max_batches is not None and total_batches >= args.max_batches:
            break

    print("=== 全零输入统计 ===")
    print(f"总 batch 数: {total_batches}")
    print(f"总样本数: {total_samples}")
    print("\n--- 全零 batch 数 ---")
    for m in MODALITIES:
        print(f"{m}: {zero_batch_counts[m]}")
    print("\n--- 全零样本数 ---")
    for m in MODALITIES:
        ratio = (zero_sample_counts[m] / total_samples * 100) if total_samples else 0.0
        print(f"{m}: {zero_sample_counts[m]} ({ratio:.2f}%)")


if __name__ == "__main__":
    main()
