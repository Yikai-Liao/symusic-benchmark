"""
Script to benchmark MIDI file read and write performance using various libraries.
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

    Args:
        library_funcs (dict): Dictionary with 'read' and 'write' functions.
        file_paths (list): List of MIDI file paths.
        repeat (int): Number of times to repeat the read/write operations.
        use_multiprocessing (bool): Whether to use multiprocessing.
        use_tqdm (bool): Whether to use tqdm for progress bar.

    Returns:
        tuple: (list of file sizes in KB, list of average read times, list of average write times)
    """
    read_times = []
    write_times = []
    sizes = []
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
        for size, avg_read, avg_write in results:
            if size is None:
                continue
            sizes.append(size)
            read_times.append(avg_read)
            write_times.append(avg_write)
    else:
        for path in tqdm(file_paths, desc="Read and Write Benchmark"):
            size, avg_read, avg_write = measure_read_write(path, read_func, write_func, repeat)
            if size is None:
                continue
            sizes.append(size)
            read_times.append(avg_read)
            write_times.append(avg_write)

    return sizes, read_times, write_times


def measure_read_write_wrapper(args):
    """
    Wrapper for measure_read_write to facilitate multiprocessing.

    Args:
        args (tuple): Arguments for the measure_read_write function.

    Returns:
        tuple: (file size in KB, average read time, average write time)
    """
    return measure_read_write(*args)


def measure_read_write(path, read_func, write_func, repeat):
    """
    Measure the average read and write times for a single MIDI file.

    Args:
        path (str): Path to the MIDI file.
        read_func (function): Function to read the MIDI file.
        write_func (function): Function to write the MIDI file.
        repeat (int): Number of repetitions for the test.

    Returns:
        tuple: (file size in KB, average read time in seconds, average write time in seconds)
    """
    # Calculate file size in KB
    try:
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
        return size, avg_read_time, avg_write_time
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return None, None, None


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
    Parse command-line arguments, benchmark the specified libraries, and save the results.
    """
    parser = argparse.ArgumentParser(
        description="Benchmark MIDI file read and write performance."
    )
    # Libraries to benchmark (list of library names)
    parser.add_argument("--libraries", nargs='+', type=str, default=DEFAULT_LIBS,
                        help="Libraries to benchmark")
    # Number of times to repeat the test (scalar or list matching number of libraries)
    parser.add_argument("--repeat", nargs='+', type=int, default=4,
                        help="Number of times to repeat the test")
    # Root directory of the dataset
    parser.add_argument("--dataset_root", type=str, help="Root directory of the dataset")
    # JSON file containing the list of relative paths to the MIDI files
    parser.add_argument("--dataset_config", type=str,
                        help="JSON file containing the list of relative paths to the MIDI files")
    # Output directory for results
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")
    # Flag to use tqdm for progress bar
    parser.add_argument("--tqdm", action='store_true', help="Use tqdm for progress bar")

    args = parser.parse_args()

    # Import tqdm if enabled; otherwise, define a dummy tqdm

    # Construct full paths for MIDI files using dataset_root and JSON config (relative paths)
    with open(args.dataset_config, 'r') as f:
        rel_paths = json.load(f)
    midi_files = [os.path.join(args.dataset_root, p) for p in rel_paths]

    # Dictionary to store benchmark results
    benchmark_results = {
        'read': {},
        'write': {}
    }

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

    # Output selected libraries and their repeat counts
    print(f"Libraries: {args.libraries}")
    print(f"Repeat: {repeats}")

    # Execute benchmark for each library
    for lib, repeat in zip(args.libraries, repeats):
        print(f"Benchmarking {lib} READ and WRITE ...")
        sizes, read_times, write_times = benchmark_read_write(
            LIBRARIES[lib], midi_files, repeat=repeat, use_multiprocessing=False, use_tqdm=args.tqdm

        )
        benchmark_results['read'][lib] = (sizes, read_times)
        benchmark_results['write'][lib] = (sizes, write_times)

    # Create output directory using the stem of the dataset_config file
    output_dir = os.path.join(args.output_dir, Path(args.dataset_config).stem)
    os.makedirs(output_dir, exist_ok=True)
    prefix = Path(args.dataset_config).stem

    # Save benchmark results as CSV files for each library
    for lib in args.libraries:
        sizes, read_times = benchmark_results['read'][lib]
        read_df = pd.DataFrame({'File Size (KB)': sizes, 'Read Time (s)': read_times})
        read_df.to_csv(f'{output_dir}/{lib}_read.csv', index=False)

        sizes, write_times = benchmark_results['write'][lib]
        write_df = pd.DataFrame({'File Size (KB)': sizes, 'Write Time (s)': write_times})
        write_df.to_csv(f'{output_dir}/{lib}_write.csv', index=False)

    print(f"Results saved in '{output_dir}'.")


if __name__ == '__main__':
    main()