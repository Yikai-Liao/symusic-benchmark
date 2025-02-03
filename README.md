# MIDI Parsing & Dumping Benchmark

## Results

![Image](https://github.com/user-attachments/assets/6e5c98dd-51ea-4cca-a98f-288f35c752cb)

Tested libs are list here:

| Library         | Version  | Language    |
|-----------------|----------|-------------|
| symusic         | 0.5.6    | Cpp/Python     |
| midifile_cpp    | N/A    | Cpp         |
| mido            | 1.3.3    | Python      |
| miditoolkit     | 1.0.1    | Python      |
| pretty_midi     | 0.2.10   | Python      |
| partitura       | 1.6.0    | Python      |
| music21         | 9.3.0    | Python      |
| MIDI.jl         | 2.7.0    | Julia       |
| @tonejs/midi    | 2.0.0    | JavaScript  |

* `symusic` is a c++ library with an easy to use python binding.
* `music21` is too slow to run the full benchmark

| Language | Version  |
|----------|----------|
| Python   | 3.13.1   |
| Node.js  | 23.5.0  |
| Julia    | 1.11.3   |

The test platform is the free macos-lastest runner (M1) on github actions.

## Reproduce

Create a virtual env if you need. And make sure you have c++ toolchain locally.

```bash
git clone --recursive https://github.com/Yikai-Liao/symusic-benchmark.git
cd symusic-benchmark
pip install -r ./requirements.txt
pip install ./binding/midifile_cpp
bash ./script/install_partitura_dev.sh
```

And then, install `julia` if you want to test `MIDI.jl`. 
You could use `prepare_denpendency.jl` script to install required `julia` libs.

```bash
julia ./script/prepare_dependency.jl
```

Download the dataset from Google Drive:
```bash
python ./script/prepare_dataset.py --output ./
```

Now you could run the benchmark

```bash
DATASET_ROOT="./symusic_benchmark_datasets"
DATASET_CONFIG="./config/uniform_10_8.json"
OUTPUT_DIR="./results"

python ./script/benchmark.py \
  --libraries symusic midifile_cpp miditoolkit pretty_midi partitura \
  --repeat 100 100 10 10 10 \
  --dataset_root $DATASET_ROOT \
  --dataset_config $DATASET_CONFIG \
  --output $OUTPUT_DIR \
  --tqdm

julia ./script/benchmark.jl \
  --repeat 100 \
  --dataset_root $DATASET_ROOT \
  --dataset_config $DATASET_CONFIG \
  --output $OUTPUT_DIR \
  --verbose
```

For `uniform_10_8.json`, the results will be stored in `./results/uniform_10_8`.
Then, parse the result dir to `visualize.py`

```bash
python visualize.py --input ./results/uniform_10_8/ --output ./fig
```

You will find the final results there.


## Dataset

Download from [Google Drive](https://drive.google.com/file/d/1uynx2E4aj7iMa8_p4WDhNbQuCFLpl5Hx/view?usp=sharing)

