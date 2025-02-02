#!/usr/bin/env python3
"""
Script to sample a MIDI dataset by binning files based on their file sizes.

This script scans a directory for MIDI files, filters out invalid files, bins the valid files based on file size,
and randomly selects a maximum number of files per bin. The resulting list of sampled file paths is saved to a JSON file.
"""

import os
import random
import argparse
import json
from collections import defaultdict
from pathlib import Path
from time import sleep
from typing import List

import symusic as sm
from tqdm import tqdm


def build_dataset(
    files: List[str],
    root: str,
    bin_size: int,
    max_files_per_bin: int,
    seed: int = 42
) -> List[str]:
    """
    Bin MIDI files by their file size and return a filtered list of file paths.

    Args:
        files (List[str]): List of MIDI file paths.
        root (str): Root directory of the MIDI files.
        bin_size (int): Size of each bin in KB.
        max_files_per_bin (int): Maximum allowed number of files per bin.
        seed (int): Random seed for reproducibility (default: 42).

    Returns:
        List[str]: Filtered list of MIDI file paths.
    """
    # Set the random seed for reproducibility
    random.seed(seed)

    # Dictionary to store file paths for each bin (bin index -> list of file paths)
    bins = defaultdict(list)

    # Sort files to ensure consistent order each time
    sorted_files = sorted(files)
    for file_path in sorted_files:
        # Calculate file size in KB
        file_size = os.path.getsize(file_path) / 1024
        # Determine bin index based on file size
        bin_index = int(file_size // bin_size)
        # Convert file path to a relative path with respect to the root directory
        relative_path = os.path.relpath(file_path, root)
        bins[bin_index].append(relative_path)

    # Select files from each bin based on the maximum allowed per bin
    selected_files = []
    for bin_files in bins.values():
        if len(bin_files) > max_files_per_bin:
            # Randomly sample files if the bin exceeds the maximum allowed
            selected_files.extend(random.sample(bin_files, max_files_per_bin))
        else:
            selected_files.extend(bin_files)

    return selected_files


def is_valid(file_path: str) -> bool:
    """
    Check if a MIDI file is valid by attempting to parse it using symusic.

    Args:
        file_path (str): Path to the MIDI file.

    Returns:
        bool: True if the file is valid, False otherwise.
    """
    try:
        sm.Score(file_path)
        return True
    except Exception:
        return False


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Sample the dataset according to file size by binning MIDI files."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory containing MIDI files."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./sampled_dataset.json",
        help="Output JSON file path."
    )
    parser.add_argument(
        "--bin_size",
        type=int,
        default=25,
        help="Size of each bin in KB."
    )
    parser.add_argument(
        "--max_files_per_bin",
        type=int,
        default=4,
        help="Maximum number of files to keep per bin."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )

    args = parser.parse_args()

    # Define the root directory for the MIDI dataset
    root = Path(args.input)
    if not root.exists():
        raise FileNotFoundError(f"Directory '{args.input}' not found.")

    print("Scanning dataset to filter out invalid files...")
    sleep(0.1)

    # Recursively find all MIDI files in the directory (supporting .mid and similar extensions)
    midi_files = list(map(str, root.glob('**/*.mid*')))
    # Filter out invalid MIDI files
    valid_midi_files = list(filter(is_valid, tqdm(midi_files)))

    # Build the dataset by binning files based on their size
    sampled_files = build_dataset(
        valid_midi_files,
        str(root),
        bin_size=args.bin_size,
        max_files_per_bin=args.max_files_per_bin,
        seed=args.seed
    )

    print(f"Number of sampled MIDI files: {len(sampled_files)} / {len(valid_midi_files)}")

    # Save the sampled dataset to a JSON file
    with open(args.output, 'w') as f:
        json.dump(sampled_files, f, indent=4)
    print(f"Sampled dataset saved to '{args.output}'.")


if __name__ == '__main__':
    main()