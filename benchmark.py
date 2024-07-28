from typing import List
from pathlib import Path
import midifile_binding as mf 
import symusic as sm 
import miditoolkit as mtk
import pretty_midi as pm
import music21 as m21
import timeit
import os
from tqdm import tqdm
from time import sleep


MIDI_DATASET_NAMES = ('maestro', 'musicnet', 'POP909')
ABC_DATASET_NAMES = ('nottingham',)
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

ABC_DATASET = {
    name: sorted(list(filter(
        is_valid, 
        tqdm(list(map(str, Path(DATASET_ROOT).joinpath(name).rglob('*.abc'))), desc=name)
    ))) for name in ABC_DATASET_NAMES
}

# show file numbers and average file size(in KB) for each dataset
# show in pandas dataframe
import pandas as pd
dataset_stat = pd.DataFrame()
for name, files in MIDI_DATASET.items():
    dataset_stat.loc[name, 'format'] = 'midi'
    dataset_stat.loc[name, 'file_num'] = len(files)
    dataset_stat.loc[name, 'avg_size(KB)'] = sum(Path(f).stat().st_size for f in files) / len(files) / 1024

for name, files in ABC_DATASET.items():
    dataset_stat.loc[name, 'format'] = 'abc'
    dataset_stat.loc[name, 'file_num'] = len(files)
    dataset_stat.loc[name, 'avg_size(KB)'] = sum(Path(f).stat().st_size for f in files) / len(files) / 1024

# set file number to integer
dataset_stat['file_num'] = dataset_stat['file_num'].astype(int)
dataset_stat

# MAX_FILES_PER_DATASET = int(dataset_stat['file_num'].max())  
# REPEAT_TIMES = 5

MAX_FILES_PER_DATASET = 20 # for testing
REPEAT_TIMES = 2 # for testing

print(f"MAX_FILES_PER_DATASET: {MAX_FILES_PER_DATASET}")
print(f"REPEAT_TIMES: {REPEAT_TIMES}")

def bench_midi(files: List[str], repeat=10):
    def bench_read(lib, load, _files):
        for f in tqdm(_files, desc=f'r {lib}'):
            load(f)

    def bench_rw(lib, load, dump, _files):
        for f in tqdm(_files, desc=f'w {lib}'):
            score = load(f)
            dump(score, './tmp')
    
    env = globals()
    env.update(locals())
    
    read_time = {
        'symusic': timeit.timeit('bench_read("symusic", sm.Score, files)', number=repeat, globals=env),
        'midifile': timeit.timeit('bench_read("midifile", mf.load, files)', number=repeat, globals=env),
        'miditoolkit': timeit.timeit('bench_read("miditoolkit", mtk.MidiFile, files)', number=repeat, globals=env),
        'prettymidi': timeit.timeit('bench_read("prettymidi", pm.PrettyMIDI, files)', number=repeat, globals=env),
        # 'music21': timeit.timeit('bench_read("music21", m21.converter.parse, files)', number=repeat, globals=env),
    }
    write_time = {
        'symusic': timeit.timeit('bench_rw("symusic", sm.Score, lambda x,y: x.dump_midi(y), files)', number=repeat, globals=env),
        'midifile': timeit.timeit('bench_rw("midifile", mf.load, lambda x,y: x.dump_midi(y), files)', number=repeat, globals=env),
        'miditoolkit': timeit.timeit('bench_rw("miditoolkit", mtk.MidiFile, lambda x,y: x.dump(y), files)', number=repeat, globals=env),
        'prettymidi': timeit.timeit('bench_rw("prettymidi", pm.PrettyMIDI, lambda x,y: x.write(y), files)', number=repeat, globals=env),
        # 'music21': timeit.timeit('bench_rw("music21", m21.converter.parse, lambda x,y: x.write("midi", y), files)', number=repeat, globals=env),
    }
    os.remove('./tmp')
    read_time = {
        k: v / repeat
        for k, v in read_time.items()
    }
    write_time = {
        k: v / repeat - read_time[k]
        for k, v in write_time.items()
    }
    return read_time, write_time
from collections import defaultdict
midi_read_benchmark = defaultdict(list)
midi_write_benchmark = defaultdict(list)

for name, files in MIDI_DATASET.items():
    print(f"benchmarking {name}...")
    read_time, write_time = bench_midi(files[:MAX_FILES_PER_DATASET], repeat=REPEAT_TIMES)
    for k, v in read_time.items():
        midi_read_benchmark[k].append(v)
    for k, v in write_time.items():
        midi_write_benchmark[k].append(v)

midi_read_pd = pd.DataFrame(dict(midi_read_benchmark), index=MIDI_DATASET.keys())
midi_write_pd = pd.DataFrame(dict(midi_write_benchmark), index=MIDI_DATASET.keys())
# dump to csv
midi_read_pd.to_csv('midi_read_benchmark.csv')
midi_write_pd.to_csv('midi_write_benchmark.csv')

print("read midi files:")
print(midi_read_pd)
print("\nwrite midi files:")
print(midi_write_pd)

def bench_abc(files: List[str], repeat=10):
    def bench_read(lib, load, _files):
        for f in tqdm(_files, desc=f'r {lib}'):
            load(f)

    def bench_rw(lib, load, dump, _files):
        for f in tqdm(_files, desc=f'w {lib}'):
            score = load(f)
            dump(score, './tmp')
    
    env = globals()
    env.update(locals())
    
    read_time = {
        'symusic': timeit.timeit('bench_read("symusic", sm.Score, files)', number=repeat, globals=env),
        'music21': timeit.timeit('bench_read("music21", m21.converter.parse, files)', number=repeat, globals=env),
    }
    write_time = {
        'symusic': timeit.timeit('bench_rw("symusic", sm.Score, lambda x,y: x.dump_abc(y), files)', number=repeat, globals=env),
        'music21': float('nan'),
    }
    os.remove('./tmp')

    read_time = {
        k: v / repeat
        for k, v in read_time.items()
    }
    write_time = {
        k: v / repeat - read_time[k]
        for k, v in write_time.items()
    }
    return read_time, write_time

abc_read_benchmark = defaultdict(list)
abc_write_benchmark = defaultdict(list)

for name, files in ABC_DATASET.items():
    print(f"benchmarking {name}...")
    read_time, write_time = bench_abc(files[:MAX_FILES_PER_DATASET], repeat=REPEAT_TIMES)
    for k, v in read_time.items():
        abc_read_benchmark[k].append(v)
    for k, v in write_time.items():
        abc_write_benchmark[k].append(v)


abc_read_pd = pd.DataFrame(dict(abc_read_benchmark), index=ABC_DATASET.keys())
abc_write_pd = pd.DataFrame(dict(abc_write_benchmark), index=ABC_DATASET.keys())
# dump to csv
abc_read_pd.to_csv('abc_read_benchmark.csv')
abc_write_pd.to_csv('abc_write_benchmark.csv')

print("read abc files:")
print(abc_read_pd)
print("\nwrite abc files:")
print(abc_write_pd)