from typing import List
from pathlib import Path
import symusic as sm
import miditoolkit as mtk
import pretty_midi as pm
import partitura as pa
import music21 as m21
import midifile_cpp as mf

import timeit
import os
from time import sleep

ENABLE_TQDM = True

if ENABLE_TQDM:
    from tqdm import tqdm
else:
    def tqdm(iterable, *args, **kwargs):
        return iterable

MIDI_DATASET_NAMES = ('maestro', 'musicnet', 'POP909')
DATASET_ROOT = "./symusic_benchmark_datasets"

def is_valid(f: str):
    try:
        sm.Score(f)
        return True
    except:
        return False

print("Scanning datasets to filter out invalid files...")
sleep(0.1)
MIDI_DATASET = {
    name: sorted(list(filter(
        is_valid,
        tqdm(list(map(str, Path(DATASET_ROOT).joinpath(name).rglob('*.mid*'))), desc=name)
    )))  for name in MIDI_DATASET_NAMES
}

import os
import random
from typing import List, Dict
from collections import defaultdict




import matplotlib.pyplot as plt
import numpy as np
import timeit
import os
from tqdm import tqdm
import pandas as pd
import multiprocessing


# 测试读取和写入时间的函数，使用传入的读取和写入函数以及重复次数
def benchmark_read_write(library_funcs, file_paths, repeat=1, use_multiprocessing=False):
    read_times = []
    write_times = []
    sizes = []
    read_func = library_funcs['read']
    write_func = library_funcs['write']

    if use_multiprocessing:
        cpu_count = multiprocessing.cpu_count() // 2
        with multiprocessing.Pool(cpu_count) as pool:
            results = list(tqdm(
                pool.imap(measure_read_write_wrapper, [(path, read_func, write_func, repeat) for path in file_paths]),
                desc="Read and Write Benchmark", total=len(file_paths)))

        for size, read_time, write_time in results:
            sizes.append(size)
            read_times.append(read_time)
            write_times.append(write_time)
    else:
        for idx, path in enumerate(tqdm(file_paths, desc="Read and Write Benchmark")):
            size, avg_read_time, avg_write_time = measure_read_write(path, read_func, write_func, repeat)
            sizes.append(size)
            read_times.append(avg_read_time)
            write_times.append(avg_write_time)

    return sizes, read_times, write_times


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


def mf_write(midi_obj, path):
    midi_obj.dump_midi(path)


def id_read(path):
    with open(path, 'rb') as f:
        return f.read()


def id_write(data, path):
    with open(path, 'wb') as f:
        f.write(data)


# 定义不同库的读取和写入操作
LIBRARIES = {
    'miditoolkit': {
        'read': mtk.MidiFile,
        'write': mtk_write,
        'repeat': 1,
        'color': 'blue',
        'line_color': 'darkblue',
        'mp': False
    },
    'symusic': {
        'read': lambda path: sm.Score(path),
        'write': lambda midi_obj, path: midi_obj.dump_midi(path),
        'repeat': 32,
        'color': 'green',
        'line_color': 'darkgreen',
        'mp': False
    },
    'pretty_midi': {
        'read': pm.PrettyMIDI,
        'write': pm_write,
        'repeat': 1,
        'color': 'orange',
        'line_color': 'darkorange',
        'mp': False
    },
    'music21': {
        'read': music21_read,
        'write': music21_write,
        'repeat': 1,
        'color': 'purple',
        'line_color': 'darkviolet',
        'mp': False
    },
    'partitura': {
        'read': pa.load_performance_midi,
        'write': pa.save_performance_midi,
        'repeat': 1,
        'color': 'red',
        'line_color': 'darkred',
        'mp': False
    },
    'midifile_cpp': {
        'read': mf.load,
        'write': mf_write,
        'repeat': 8,
        'color': 'black',
        'line_color': 'black',
        'mp': False
    },
    'identity': {
        'read': id_read,
        'write': id_write,
        'repeat': 32,
        'color': 'black',
        'line_color': 'black',
        'mp': False
    }
}

# 要测试的库列表
# LIBRARY_LIST = ['symusic', 'music21']
# LIBRARY_LIST = ['symusic', 'miditoolkit', 'pretty_midi', 'partitura', 'midifile_cpp']
LIBRARY_LIST = ['identity']

# 存储benchmark结果的字典
benchmark_results = {
    'read': {},
    'write': {}
}

# 执行benchmark
for lib in LIBRARY_LIST:
    print(f"Benchmarking {lib} READ and WRITE ...")
    sizes, read_times, write_times = benchmark_read_write(LIBRARIES[lib], all_midi_files,
                                                          repeat=LIBRARIES[lib]['repeat'],
                                                          use_multiprocessing=LIBRARIES[lib]['mp'])

    benchmark_results['read'][lib] = (sizes, read_times)
    benchmark_results['write'][lib] = (sizes, write_times)
    benchmark_results['write'][lib] = (sizes, write_times)

# 创建结果文件夹
os.makedirs('results', exist_ok=True)

# 保存结果为 CSV 文件（每个库保存两个文件：读和写）
for lib in LIBRARY_LIST:
    sizes, read_times = benchmark_results['read'][lib]
    read_df = pd.DataFrame({'File Size (KB)': sizes, 'Read Time (s)': read_times})
    read_df.to_csv(f'results/{lib}_read.csv', index=False)

    sizes, write_times = benchmark_results['write'][lib]
    write_df = pd.DataFrame({'File Size (KB)': sizes, 'Write Time (s)': write_times})
    write_df.to_csv(f'results/{lib}_write.csv', index=False)