from typing import List, Dict
import os
import random
from collections import defaultdict
from time import sleep
import symusic as sm

from tqdm import tqdm
from pathlib import Path
import argparse
import json

def build_dataset(
        files: List[str],
        root: str,
        bin_size: int,
        max_files_per_bin: int,
        seed: int = 42
) -> List[str]:
    """
    根据文件大小对MIDI文件进行分bin，并返回一个经过筛选的文件路径列表。

    :param files: MIDI文件路径列表。
    :param root: MIDI文件根目录。
    :param bin_size: 每个bin的大小（单位：KB）。
    :param max_files_per_bin: 每个bin中允许的最大文件数量。
    :param seed: 随机种子，保证复现性。
    :return: 筛选后的MIDI文件路径列表。
    """
    # 设置随机种子，保证选择的可复现性
    random.seed(seed)

    # 创建一个字典来存储每个bin中的文件路径
    bins = defaultdict(list)

    # 遍历所有MIDI文件并根据文件大小分bin
        # 对文件路径进行排序，保证每次的顺序一致
    sorted_files = sorted(files)
    for file_path in sorted_files:
        file_size = os.path.getsize(file_path) / 1024  # 文件大小，单位为KB
        bin_index = int(file_size // bin_size)
        # Convert path to relative path
        file_path = os.path.relpath(file_path, root)
        bins[bin_index].append(file_path)

    # 根据每个bin的上限进行筛选
    selected_files = []
    for bin_files in bins.values():
        # 如果某个bin中的文件数量大于上限，则随机选择其中的文件
        if len(bin_files) > max_files_per_bin:
            selected_files.extend(random.sample(bin_files, max_files_per_bin))
        else:
            selected_files.extend(bin_files)

    return selected_files

def is_valid(f: str):
    try:
        sm.Score(f)
        return True
    except:
        return False

if __name__ == '__main__':
    # Define the parser
    parser = argparse.ArgumentParser(description="Sample the dataset according to file size")
    parser.add_argument("--input", type=str, help="Input directory")
    parser.add_argument("--output", type=str, default="./sampled_dataset.json", help="Output json file path")
    # bin size and max files per bin
    parser.add_argument("--bin_size", type=int, default=25, help="Bin size in KB")
    parser.add_argument("--max_files_per_bin", type=int, default=4, help="Max files per bin")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    root = Path(args.input)
    # check if the input directory exists
    if not root.exists():
        raise FileNotFoundError(f"Directory '{args.input}' not found.")


    print("Scanning datasets to filter out invalid files...")
    sleep(0.1)
    MIDI_DATASET = list(filter(is_valid, tqdm(list(map(str, root.glob('**/*.mid*'))))))

    all_midi_files = build_dataset(
        MIDI_DATASET, str(root),
        bin_size=args.bin_size, max_files_per_bin=args.max_files_per_bin, seed=args.seed
    )
    print(f"Number of MIDI files: {len(all_midi_files)} / {len(MIDI_DATASET)}")
    # Save the sampled dataset to a JSON file
    with open(args.output, 'w') as f:
        json.dump(all_midi_files, f, indent=4)