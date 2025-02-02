# MIDI parsing libraries import
import symusic as sm
import miditoolkit as mtk
import pretty_midi as pm
import partitura as pa
import music21 as m21
import midifile_cpp as mf
# Other imports
import timeit
import os
import pandas as pd
import multiprocessing
from typing import List
from pathlib import Path
import argparse
import json



# 测试读取和写入时间的函数，使用传入的读取和写入函数以及重复次数
def benchmark_read_write(library_funcs, file_paths, repeat=1, use_multiprocessing=False):
    _read_times = []
    _write_times = []
    _sizes = []
    read_func = library_funcs['read']
    write_func = library_funcs['write']

    if use_multiprocessing:
        cpu_count = multiprocessing.cpu_count() // 2
        with multiprocessing.Pool(cpu_count) as pool:
            results = list(tqdm(
                pool.imap(measure_read_write_wrapper, [(path, read_func, write_func, repeat) for path in file_paths]),
                desc="Read and Write Benchmark", total=len(file_paths)))

        for size, read_time, write_time in results:
            _sizes.append(size)
            _read_times.append(read_time)
            _write_times.append(write_time)
    else:
        for idx, path in enumerate(tqdm(file_paths, desc="Read and Write Benchmark")):
            size, avg_read_time, avg_write_time = measure_read_write(path, read_func, write_func, repeat)
            _sizes.append(size)
            _read_times.append(avg_read_time)
            _write_times.append(avg_write_time)

    return _sizes, _read_times, _write_times


# 测量单个文件的读取和写入时间
def measure_read_write_wrapper(args):
    return measure_read_write(*args)


def measure_read_write(path, read_func, write_func, repeat):
    size = os.path.getsize(path) / 1024  # 文件大小，单位为KB

    # 测试读取时间
    start_time = timeit.default_timer()
    midi_object = None
    for _ in range(repeat):
        midi_object = read_func(path)
    read_total_time = timeit.default_timer() - start_time
    avg_read_time = read_total_time / repeat

    # 测试写入时间
    start_time = timeit.default_timer()
    for i in range(repeat):
        temp_path = f"temp_{os.path.basename(path)}_{i}.mid"
        write_func(midi_object, temp_path)  # 写入操作
    write_total_time = timeit.default_timer() - start_time
    avg_write_time = write_total_time / repeat

    # 删除临时文件
    for i in range(repeat):
        temp_path = f"temp_{os.path.basename(path)}_{i}.mid"
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return size, avg_read_time, avg_write_time


def music21_read(path):
    return m21.converter.parse(path)


def music21_write(midi_obj, path):
    midi_obj.write('midi', path)


def mtk_write(midi_obj, path):
    midi_obj.dump(path)


def pm_write(midi_obj, path):
    midi_obj.write(path)

def id_read(path):
    with open(path, 'rb') as f:
        return f.read()


def id_write(data, path):
    with open(path, 'wb') as f:
        f.write(data)

def sm_read(path):
    return sm.Score(path)

def sm_write(midi_obj, path):
    midi_obj.dump_midi(path)

# 定义不同库的读取和写入操作
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

# 定义命令行参数
DEFAULT_LIBS = ['symusic', 'midifile_cpp', 'miditoolkit', 'pretty_midi', 'partitura']

parser = argparse.ArgumentParser(description="Benchmark midi file read and write performance.")
# 给定一个序列，用于指定要测试的库
parser.add_argument("--libraries", nargs='+', type=str, default=DEFAULT_LIBS, help="Libraries to benchmark")
# 重复次数, 标量，或者一个与库数量相同的序列
parser.add_argument("--repeat", nargs='+', type=int, default=4, help="Number of times to repeat the test")
# 数据集跟目录
parser.add_argument("--dataset_root", type=str, help="Root directory of the dataset")
# json 文件列表
parser.add_argument("--dataset_config", type=str, help="Json file containing the list of relative paths to the midi files")
# 输出目录
parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")
# tqdm 是否启用
parser.add_argument("--tqdm", action='store_true', help="Use tqdm for progress bar")

args = parser.parse_args()



if args.tqdm:
    from tqdm import tqdm
else:
    def tqdm(iterable, *args, **kwargs):
        return iterable


# json 为相对路径列表，与数据集根目录拼接
with open(args.dataset_config, 'r') as f:
    rel = json.load(f)
    midi_files = [os.path.join(args.dataset_root, p) for p in rel]

# 存储benchmark结果的字典
benchmark_results = {
    'read': {},
    'write': {}
}

# 检查libraries参数是否合法
for lib in args.libraries:
    if lib not in LIBRARIES:
        raise ValueError(f"Library '{lib}' is not supported.")


# 检查repeat参数是否合法
if len(args.repeat) == 1:
    repeats = [args.repeat[0]] * len(args.libraries)
elif len(args.repeat) == len(args.libraries):
    repeats = args.repeat
else:
    raise ValueError(f"The length of 'repeat' should be 1 or equal to the number of libraries, not {args.repeat}.")
# 输出 libraries 和 repeat
print(f"Libraries: {args.libraries}")
print(f"Repeat: {repeats}")

# 执行benchmark
for lib, repeat in zip(args.libraries, repeats):
    print(f"Benchmarking {lib} READ and WRITE ...")
    sizes, read_times, write_times = benchmark_read_write(
        LIBRARIES[lib], midi_files,
        repeat=repeat,
        use_multiprocessing=False
    )

    benchmark_results['read'][lib] = (sizes, read_times)
    benchmark_results['write'][lib] = (sizes, write_times)

# 创建结果文件夹
output_dir = os.path.join(args.output_dir, Path(args.dataset_config).stem)
os.makedirs(output_dir, exist_ok=True)
# 每个 lib 保存两个文件：读和写，文件前缀为 config 文件名 _ lib
prefix = Path(args.dataset_config).stem

# 保存结果为 CSV 文件
for lib in args.libraries:

    sizes, read_times = benchmark_results['read'][lib]
    read_df = pd.DataFrame({'File Size (KB)': sizes, 'Read Time (s)': read_times})
    read_df.to_csv(f'{output_dir}/{lib}_read.csv', index=False)

    sizes, write_times = benchmark_results['write'][lib]
    write_df = pd.DataFrame({'File Size (KB)': sizes, 'Write Time (s)': write_times})
    write_df.to_csv(f'{output_dir}/{lib}_write.csv', index=False)

print(f"Results saved in '{output_dir}'.")