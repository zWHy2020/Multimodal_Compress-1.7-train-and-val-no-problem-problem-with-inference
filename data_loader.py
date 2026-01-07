"""
多模态数据加载器（支持 Manifest v1/v2，严格模式）

关键特性：
- Manifest v2：每视频一条记录，text/texts 与 image/files 列表
- 训练随机采样 caption/keyframe，验证固定 captions[0]/keyframes[0]
- 视频按需抽帧（不读取全量帧），短视频 repeat-last + mask
- 严格模式默认开启：关键模态缺失会丢弃样本并统计；可选 allow_missing_modalities 走零填充调试
"""

from __future__ import annotations

import json
import os
import random
import warnings
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def _default_image_transform(image_size: Tuple[int, int]) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ]
    )


def _default_video_transform(image_size: Tuple[int, int]) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ]
    )


def _load_manifest(manifest: str) -> List[Dict[str, Any]]:
    with open(manifest, "r", encoding="utf-8") as f:
        return json.load(f)


class MultimodalDataset(Dataset):
    """支持 v1/v2 manifest 的数据集。"""

    def __init__(
        self,
        data_dir: str,
        data_list: Sequence[Dict[str, Any]],
        text_tokenizer: Optional[Any] = None,
        image_transform: Optional[transforms.Compose] = None,
        video_transform: Optional[transforms.Compose] = None,
        max_text_length: int = 512,
        max_video_frames: int = 10,
        image_size: Tuple[int, int] = (224, 224),
        is_train: bool = True,
        allow_missing_modalities: bool = False,
        strict_mode: bool = True,
        required_modalities: Tuple[str, ...] = ("video", "text"),
        seed: Optional[int] = None,
    ):
        self.data_dir = data_dir
        self.data_list = list(data_list)
        self.text_tokenizer = text_tokenizer
        self.image_transform = image_transform or _default_image_transform(image_size)
        self.video_transform = video_transform or _default_video_transform(image_size)
        self.max_text_length = max_text_length
        self.max_video_frames = max_video_frames
        self.image_size = image_size
        self.is_train = is_train
        self.allow_missing_modalities = allow_missing_modalities
        self.strict_mode = strict_mode
        self.required_modalities = required_modalities
        self.text_pad_token_id = (
            getattr(self.text_tokenizer, "pad_token_id", 0) if self.text_tokenizer is not None else 0
        )
        self.drop_count = 0
        self.missing_counts: Dict[str, int] = {"text": 0, "image": 0, "video": 0}
        self.random_state = random.Random(seed)
        self.version = "v2" if any("texts" in item.get("text", {}) for item in self.data_list) else "v1"
        if self.version == "v1":
            warnings.warn("检测到 manifest v1：会导致重复解码与语义错配，建议迁移到 v2。", UserWarning)

    def __len__(self) -> int:
        return len(self.data_list)

    def _select_caption(self, text_info: Dict[str, Any]) -> Tuple[str, int]:
        if "texts" in text_info:
            texts = text_info["texts"]
            if not texts:
                raise ValueError("text.texts 为空")
            if self.is_train:
                idx = self.random_state.randint(0, len(texts) - 1)
            else:
                idx = 0
            return texts[idx], idx
        text = text_info.get("text", "")
        return text, 0

    def _select_keyframe(self, image_info: Dict[str, Any]) -> Tuple[str, int]:
        if "files" in image_info:
            files = image_info["files"]
            if not files:
                raise ValueError("image.files 为空")
            if self.is_train:
                idx = self.random_state.randint(0, len(files) - 1)
            else:
                idx = 0
            return files[idx], idx
        return image_info["file"], 0

    def _tokenize(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.text_tokenizer:
            tokens = self.text_tokenizer(
                text, max_length=self.max_text_length, truncation=True, padding="max_length", return_tensors="pt"
            )
            input_ids = tokens["input_ids"].squeeze(0)
            attention_mask = tokens["attention_mask"].squeeze(0)
        else:
            encoded = [ord(c) for c in text[: self.max_text_length]]
            input_ids = torch.tensor(encoded, dtype=torch.long)
            pad_len = self.max_text_length - input_ids.shape[0]
            if pad_len > 0:
                input_ids = torch.cat(
                    [input_ids, torch.full((pad_len,), self.text_pad_token_id, dtype=torch.long)], dim=0
                )
            attention_mask = torch.zeros_like(input_ids)
            attention_mask[: len(encoded)] = 1
        return input_ids, attention_mask

    def _load_image(self, image_path: str) -> torch.Tensor:
        full_path = os.path.join(self.data_dir, image_path)
        image = Image.open(full_path).convert("RGB")
        return self.image_transform(image)

    def _load_video_frames(self, video_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        full_path = os.path.join(self.data_dir, video_path)
        cap = cv2.VideoCapture(full_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"无法打开视频: {full_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            # CAP_PROP_FRAME_COUNT 可能不可用，顺序计数
            total_frames = 0
            while True:
                ret, _ = cap.read()
                if not ret:
                    break
                total_frames += 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        sample_count = min(total_frames, self.max_video_frames)
        if sample_count <= 0:
            cap.release()
            raise RuntimeError(f"视频为空: {full_path}")
        target_indices = (
            np.linspace(0, total_frames - 1, num=sample_count, dtype=int).tolist()
            if total_frames > 1
            else [0] * sample_count
        )
        frames: List[Image.Image] = []
        current_idx = 0
        target_ptr = 0
        while target_ptr < len(target_indices):
            ret, frame = cap.read()
            if not ret:
                break
            if current_idx == target_indices[target_ptr]:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
                target_ptr += 1
            current_idx += 1
        cap.release()
        if not frames:
            raise RuntimeError(f"未能读取任何帧: {full_path}")
        true_frames = len(frames)
        # repeat-last padding
        while len(frames) < self.max_video_frames:
            frames.append(frames[-1].copy())
        video_tensor = torch.stack([self.video_transform(frame) for frame in frames[: self.max_video_frames]])
        mask = torch.zeros(self.max_video_frames, dtype=torch.float32)
        mask[: min(true_frames, self.max_video_frames)] = 1.0
        return video_tensor, mask

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        item = self.data_list[idx]
        sample: Dict[str, Any] = {"meta": {}, "valid": {}}
        meta = sample["meta"]
        meta["video_id"] = item.get("meta", {}).get("video_id") or os.path.splitext(os.path.basename(item.get("video", {}).get("file", "")))[0]
        try:
            caption, caption_idx = self._select_caption(item.get("text", {}))
            input_ids, attention_mask = self._tokenize(caption)
            sample["text"] = input_ids
            sample["text_attention_mask"] = attention_mask
            sample["pad_token_id"] = self.text_pad_token_id
            sample["valid"]["text"] = True
            meta["caption_idx"] = caption_idx
        except Exception as exc:
            self.missing_counts["text"] += 1
            sample["valid"]["text"] = False
            if self.strict_mode and "text" in self.required_modalities:
                self.drop_count += 1
                sample["_dropped"] = f"text_error: {exc}"
                return sample
        try:
            image_path, keyframe_idx = self._select_keyframe(item.get("image", {}))
            sample["image"] = self._load_image(image_path)
            sample["valid"]["image"] = True
            meta["keyframe_idx"] = keyframe_idx
        except Exception:
            self.missing_counts["image"] += 1
            sample["valid"]["image"] = False
            if self.strict_mode and "image" in self.required_modalities:
                self.drop_count += 1
                sample["_dropped"] = "image_error"
                return sample
        try:
            video_path = item.get("video", {}).get("file")
            if not video_path:
                raise FileNotFoundError("video.file 为空")
            video_tensor, frame_mask = self._load_video_frames(video_path)
            sample["video"] = video_tensor
            sample["video_frame_mask"] = frame_mask
            sample["valid"]["video"] = True
        except Exception as exc:
            self.missing_counts["video"] += 1
            sample["valid"]["video"] = False
            if self.strict_mode and "video" in self.required_modalities:
                self.drop_count += 1
                sample["_dropped"] = f"video_error: {exc}"
                return sample
        # 允许缺失模态时的填充
        if self.allow_missing_modalities and not self.strict_mode:
            if not sample["valid"].get("image", False):
                sample["image"] = torch.zeros((3, *self.image_size), dtype=torch.float32)
            if not sample["valid"].get("video", False):
                sample["video"] = torch.zeros((self.max_video_frames, 3, *self.image_size), dtype=torch.float32)
                sample["video_frame_mask"] = torch.zeros(self.max_video_frames, dtype=torch.float32)
            if not sample["valid"].get("text", False):
                sample["text"] = torch.full((self.max_text_length,), self.text_pad_token_id, dtype=torch.long)
                sample["text_attention_mask"] = torch.zeros(self.max_text_length, dtype=torch.long)
        return sample


def _pad_sequence(sequences: List[torch.Tensor], pad_value: int) -> torch.Tensor:
    max_len = max(seq.shape[0] for seq in sequences)
    padded = []
    for seq in sequences:
        pad_len = max_len - seq.shape[0]
        if pad_len > 0:
            seq = torch.cat([seq, torch.full((pad_len,), pad_value, dtype=seq.dtype)], dim=0)
        padded.append(seq)
    return torch.stack(padded, dim=0)


def collate_multimodal_batch(
    batch: List[Optional[Dict[str, Any]]],
    allow_missing_modalities: bool = False,
    required_modalities: Tuple[str, ...] = ("video", "text"),
) -> Dict[str, Any]:
    valid_samples = [b for b in batch if b is not None and not b.get("_dropped")]
    if not valid_samples:
        raise RuntimeError("批次内所有样本均被丢弃，请检查数据质量或关闭严格模式。")
    # 过滤仍缺失关键模态的样本
    filtered_samples: List[Dict[str, Any]] = []
    for s in valid_samples:
        missing_required = [m for m in required_modalities if not s.get("valid", {}).get(m, False)]
        if missing_required and not allow_missing_modalities:
            continue
        filtered_samples.append(s)
    if not filtered_samples:
        raise RuntimeError("批次内样本缺失关键模态，且 allow_missing_modalities=False，批次为空。")

    inputs: Dict[str, Any] = {}
    targets: Dict[str, Any] = {}
    metas: List[Dict[str, Any]] = [s.get("meta", {}) for s in filtered_samples]
    valid_flags: Dict[str, List[bool]] = {"text": [], "image": [], "video": []}

    # 文本
    if all(s.get("text") is not None for s in filtered_samples):
        text_tensors = [s["text"] for s in filtered_samples]
        attn_masks = [s["text_attention_mask"] for s in filtered_samples]
        pad_token = filtered_samples[0].get("pad_token_id", 0)
        inputs["text_input"] = _pad_sequence(text_tensors, pad_token)
        inputs["text_attention_mask"] = _pad_sequence(attn_masks, 0)
        targets["text"] = inputs["text_input"]
        valid_flags["text"] = [s.get("valid", {}).get("text", False) for s in filtered_samples]

    # 图像
    if all("image" in s for s in filtered_samples):
        images = torch.stack([s["image"] for s in filtered_samples])
        inputs["image_input"] = images
        targets["image"] = images
        valid_flags["image"] = [s.get("valid", {}).get("image", False) for s in filtered_samples]

    # 视频
    if all("video" in s for s in filtered_samples):
        videos = torch.stack([s["video"] for s in filtered_samples])
        inputs["video_input"] = videos
        targets["video"] = videos
        masks = [s.get("video_frame_mask", torch.ones(videos.shape[1])) for s in filtered_samples]
        inputs["video_frame_mask"] = torch.stack(masks)
        valid_flags["video"] = [s.get("valid", {}).get("video", False) for s in filtered_samples]

    batch_data = {
        "inputs": inputs,
        "targets": targets,
        "meta": metas,
        "valid": valid_flags,
    }
    if "text_attention_mask" in inputs:
        batch_data["attention_mask"] = inputs["text_attention_mask"]
    return batch_data


class MultimodalDataLoader:
    """统一的数据加载器封装。"""

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        shuffle: bool = True,
        text_tokenizer: Optional[Any] = None,
        image_size: Tuple[int, int] = (224, 224),
        max_text_length: int = 512,
        max_video_frames: int = 10,
        prefetch_factor: int = 2,
        allow_missing_modalities: bool = False,
        strict_mode: bool = True,
        required_modalities: Tuple[str, ...] = ("video", "text"),
        seed: Optional[int] = None,
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.text_tokenizer = text_tokenizer
        self.image_size = image_size
        self.max_text_length = max_text_length
        self.max_video_frames = max_video_frames
        self.prefetch_factor = prefetch_factor
        self.allow_missing_modalities = allow_missing_modalities
        self.strict_mode = strict_mode
        self.required_modalities = required_modalities
        self.seed = seed

    def create_dataset(
        self,
        data_list_or_manifest: Sequence[Dict[str, Any]] | str,
        image_transform: Optional[transforms.Compose] = None,
        video_transform: Optional[transforms.Compose] = None,
        is_train: bool = True,
    ) -> MultimodalDataset:
        data_list = (
            _load_manifest(data_list_or_manifest)
            if isinstance(data_list_or_manifest, str)
            else list(data_list_or_manifest)
        )
        return MultimodalDataset(
            data_dir=self.data_dir,
            data_list=data_list,
            text_tokenizer=self.text_tokenizer,
            image_transform=image_transform,
            video_transform=video_transform,
            max_text_length=self.max_text_length,
            max_video_frames=self.max_video_frames,
            image_size=self.image_size,
            is_train=is_train,
            allow_missing_modalities=self.allow_missing_modalities,
            strict_mode=self.strict_mode,
            required_modalities=self.required_modalities,
            seed=self.seed,
        )

    def create_dataloader(
        self,
        dataset: MultimodalDataset,
        shuffle: Optional[bool] = None,
    ) -> DataLoader:
        if shuffle is None:
            shuffle = self.shuffle
        actual_prefetch_factor = None if self.num_workers == 0 else self.prefetch_factor
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=partial(
                collate_multimodal_batch,
                allow_missing_modalities=self.allow_missing_modalities,
                required_modalities=self.required_modalities,
            ),
            pin_memory=True,
            prefetch_factor=actual_prefetch_factor,
            persistent_workers=self.num_workers > 0,
        )
