"""
Script to benchmark MIDI file read and write performance using various libraries,
and output for each library the topk slowest files (lowest throughput in MB/s)
in CSV format, with columns "MB/s" and "Path".
"""

import os
import json
import argparse
import timeit
import multiprocessing
from pathlib import Path
from typing import List

import pandas as pd

# MIDI parsing libraries
import symusic as sm
import miditoolkit as mtk
import pretty_midi as pm
import partitura as pa
import music21 as m21
import midifile_cpp as mf


def benchmark_read_write(library_funcs, file_paths, repeat=1, use_multiprocessing=False, use_tqdm=False):
    """
    Benchmark the read and write performance for a given library.
    For each file, returns a tuple: (file_path, file_size in KB, avg_read_time, avg_write_time).

    Args:
        library_funcs (dict): Dictionary with 'read' and 'write' functions.
        file_paths (list): List of MIDI file paths.
        repeat (int): Number of times to repeat the read/write operations.
        use_multiprocessing (bool): Whether to use multiprocessing.
        use_tqdm (bool): Whether to use tqdm for progress bar.

    Returns:
        list: A list of tuples, each tuple is (file_path, size_in_KB, avg_read_time, avg_write_time)
    """
    results_list = []
    read_func = library_funcs['read']
    write_func = library_funcs['write']

    if use_tqdm:
        from tqdm import tqdm
    else:
        def tqdm(iterable, *args, **kwargs):
            return iterable

    if use_multiprocessing:
        cpu_count = multiprocessing.cpu_count() // 2
        with multiprocessing.Pool(cpu_count) as pool:
            results = list(tqdm(
                pool.imap(measure_read_write_wrapper,
                          [(path, read_func, write_func, repeat) for path in file_paths]),
                desc="Read and Write Benchmark",
                total=len(file_paths)
            ))
        results_list.extend(results)
    else:
        for path in tqdm(file_paths, desc="Read and Write Benchmark"):
            result = measure_read_write(path, read_func, write_func, repeat)
            results_list.append(result)

    return results_list


def measure_read_write_wrapper(args):
    """
    Wrapper for measure_read_write to facilitate multiprocessing.
    """
    return measure_read_write(*args)


def measure_read_write(path, read_func, write_func, repeat):
    """
    Measure the average read and write times for a single MIDI file.
    Returns a tuple: (file_path, file_size in KB, average read time, average write time)
    """
    # Calculate file size in KB
    size = os.path.getsize(path) / 1024

    # Measure read time
    start_time = timeit.default_timer()
    midi_object = None
    for _ in range(repeat):
        midi_object = read_func(path)
    read_total_time = timeit.default_timer() - start_time
    avg_read_time = read_total_time / repeat

    # Measure write time
    start_time = timeit.default_timer()
    for i in range(repeat):
        temp_path = f"temp_{os.path.basename(path)}_{i}.mid"
        write_func(midi_object, temp_path)
    write_total_time = timeit.default_timer() - start_time
    avg_write_time = write_total_time / repeat

    # Remove temporary files
    for i in range(repeat):
        temp_path = f"temp_{os.path.basename(path)}_{i}.mid"
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return path, size, avg_read_time, avg_write_time


def music21_read(path):
    """Read a MIDI file using music21."""
    return m21.converter.parse(path)


def music21_write(midi_obj, path):
    """Write a MIDI file using music21."""
    midi_obj.write('midi', path)


def mtk_write(midi_obj, path):
    """Write a MIDI file using miditoolkit."""
    midi_obj.dump(path)


def pm_write(midi_obj, path):
    """Write a MIDI file using pretty_midi."""
    midi_obj.write(path)


def id_read(path):
    """Read a file in binary mode (identity function)."""
    with open(path, 'rb') as f:
        return f.read()


def id_write(data, path):
    """Write binary data to a file (identity function)."""
    with open(path, 'wb') as f:
        f.write(data)


def sm_read(path):
    """Read a MIDI file using symusic."""
    return sm.Score(path)


def sm_write(midi_obj, path):
    """Write a MIDI file using symusic."""
    midi_obj.dump_midi(path)


# Dictionary mapping library names to their corresponding read and write functions
LIBRARIES = {
    'miditoolkit': {
        'read': mtk.MidiFile,
        'write': mtk_write,
    },
    'symusic': {
        'read': sm_read,
        'write': sm_write,
    },
    'pretty_midi': {
        'read': pm.PrettyMIDI,
        'write': pm_write,
    },
    'music21': {
        'read': music21_read,
        'write': music21_write,
    },
    'partitura': {
        'read': pa.load_performance_midi,
        'write': pa.save_performance_midi,
    },
    'midifile_cpp': {
        'read': mf.load,
        'write': mf.dump,
    },
    'identity': {
        'read': id_read,
        'write': id_write,
    }
}

# Default libraries for benchmarking
DEFAULT_LIBS = ['symusic', 'midifile_cpp', 'miditoolkit', 'pretty_midi', 'partitura']


def main():
    """
    Parse command-line arguments, benchmark the specified libraries,
    and for each library output CSV files listing the topk slowest files (lowest throughput)
    for read and write operations.
    CSV 文件包含两列："MB/s"（吞吐率） 和 "Path"（文件路径）。
    """
    parser = argparse.ArgumentParser(
        description="Benchmark MIDI file read and write performance and output slow files (in CSV)."
    )
    # Libraries to benchmark (list of library names)
    parser.add_argument("--libraries", nargs='+', type=str, default=DEFAULT_LIBS,
                        help="Libraries to benchmark")
    # Number of times to repeat the test (scalar or list matching number of libraries)
    parser.add_argument("--repeat", nargs='+', type=int, default=4,
                        help="Number of times to repeat the test")
    # Root directory of the dataset
    parser.add_argument("--dataset_root", type=str, help="Root directory of the dataset", required=True)
    # JSON file containing the list of relative paths to the MIDI files
    parser.add_argument("--dataset_config", type=str,
                        help="JSON file containing the list of relative paths to the MIDI files", required=True)
    # Output directory for results
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")
    # Flag to use tqdm for progress bar
    parser.add_argument("--tqdm", action='store_true', help="Use tqdm for progress bar")
    # Top k slowest files to output (per library per operation)
    parser.add_argument("--topk", type=int, default=5, help="Number of slowest files to output for each library")

    args = parser.parse_args()

    # Construct full paths for MIDI files using dataset_root and JSON config (relative paths)
    with open(args.dataset_config, 'r') as f:
        rel_paths = json.load(f)
    midi_files = [os.path.join(args.dataset_root, p) for p in rel_paths]

    # Validate provided libraries
    for lib in args.libraries:
        if lib not in LIBRARIES:
            raise ValueError(f"Library '{lib}' is not supported.")

    # Validate the repeat parameter: either a single value or one per library
    if len(args.repeat) == 1:
        repeats = [args.repeat[0]] * len(args.libraries)
    elif len(args.repeat) == len(args.libraries):
        repeats = args.repeat
    else:
        raise ValueError(
            f"The length of 'repeat' should be 1 or equal to the number of libraries, not {args.repeat}."
        )

    # Create output directory using the stem of the dataset_config file
    output_dir = os.path.join(args.output_dir, Path(args.dataset_config).stem)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Libraries: {args.libraries}")
    print(f"Repeat: {repeats}")
    print(f"TopK slow files to output: {args.topk}")

    # 对每个库进行基准测试
    for lib, repeat in zip(args.libraries, repeats):
        print(f"\nBenchmarking {lib} ...")
        measurements = benchmark_read_write(
            LIBRARIES[lib], midi_files, repeat=repeat, use_multiprocessing=False, use_tqdm=args.tqdm
        )
        # measurements 是一个列表，每个元素为 (file_path, size_KB, avg_read_time, avg_write_time)

        # 计算读操作的吞吐率（MB/s = (size in MB) / time），注意 size 单位转换：KB -> MB
        # 如果耗时为 0 则设为 0
        read_sorted = sorted(
            measurements,
            key=lambda x: ((x[1] / 1024) / x[2]) if x[2] > 0 else float('inf')
        )
        # 写操作吞吐率
        write_sorted = sorted(
            measurements,
            key=lambda x: ((x[1] / 1024) / x[3]) if x[3] > 0 else float('inf')
        )

        # 取出 topk 最慢（吞吐率最低）的文件
        # 构造 CSV 数据，包含 "MB/s" 和 "Path"
        read_csv_data = []
        for entry in read_sorted[:args.topk]:
            throughput = (entry[1] / 1024) / entry[2] if entry[2] > 0 else 0
            read_csv_data.append({"MB/s": throughput, "Path": entry[0]})

        write_csv_data = []
        for entry in write_sorted[:args.topk]:
            throughput = (entry[1] / 1024) / entry[3] if entry[3] > 0 else 0
            write_csv_data.append({"MB/s": throughput, "Path": entry[0]})

        # 保存 CSV 文件
        read_out_path = os.path.join(output_dir, f"{lib}_read_slowest_top{args.topk}.csv")
        write_out_path = os.path.join(output_dir, f"{lib}_write_slowest_top{args.topk}.csv")
        pd.DataFrame(read_csv_data).to_csv(read_out_path, index=False)
        pd.DataFrame(write_csv_data).to_csv(write_out_path, index=False)

        print(f"Library '{lib}':")
        print(f"  Slowest READ files saved in: {read_out_path}")
        print(f"  Slowest WRITE files saved in: {write_out_path}")

    print(f"\nAll results saved in '{output_dir}'.")


if __name__ == '__main__':
    main()