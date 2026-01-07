#!/usr/bin/env python3
# coding: utf-8
"""
快速检查 manifest（支持 v1/v2）

打印条目数、unique video 数、caption/keyframe 分布，以及缺失文件计数。
"""

import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List


def load_manifest(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="检查 manifest 完整性")
    parser.add_argument("--data-dir", type=str, required=True, help="数据根目录")
    parser.add_argument("--manifest", type=str, required=True, help="manifest 文件路径")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    version = "v2" if any("texts" in item.get("text", {}) for item in manifest) else "v1"
    caption_dist = Counter()
    keyframe_dist = Counter()
    missing_video = 0
    missing_image = 0
    unique_videos = set()

    for item in manifest:
        video_info = item.get("video", {})
        video_path = video_info.get("file")
        if video_path:
            full_video = os.path.join(args.data_dir, video_path)
            if not os.path.exists(full_video):
                missing_video += 1
        text_info = item.get("text", {})
        if version == "v2":
            caption_count = len(text_info.get("texts", []))
            image_files = item.get("image", {}).get("files", [])
        else:
            caption_count = 1 if "text" in text_info else 0
            image_file = item.get("image", {}).get("file")
            image_files = [image_file] if image_file else []
        caption_dist[caption_count] += 1
        keyframe_dist[len(image_files)] += 1
        for img in image_files:
            if img:
                if not os.path.exists(os.path.join(args.data_dir, img)):
                    missing_image += 1
        vid = item.get("meta", {}).get("video_id") or os.path.basename(video_path or "")
        if vid:
            unique_videos.add(vid)

    print("=" * 60)
    print(f"Manifest: {args.manifest}")
    print(f"版本: {version}")
    print(f"条目数: {len(manifest)}")
    print(f"unique video 数: {len(unique_videos)}")
    print(f"缺失视频: {missing_video}, 缺失关键帧: {missing_image}")
    print(f"caption 数分布: {dict(caption_dist)}")
    print(f"keyframe 数分布: {dict(keyframe_dist)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
