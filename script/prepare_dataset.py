#!/usr/bin/env python3
import gdown
import zipfile
import argparse
import os

# https://drive.google.com/file/d/1uynx2E4aj7iMa8_p4WDhNbQuCFLpl5Hx/view?usp=sharing
# MIDI_DATASET_NAMES = ('maestro', 'musicnet', 'POP909')
# ABC_DATASET_NAMES = ('nottingham',)

# Define the parser
parser = argparse.ArgumentParser(description="Download and extract the dataset")
# With only one argument, the output directory
parser.add_argument("--output", type=str, default="./", help="Output directory")

if __name__ == '__main__':
    file_id = '1uynx2E4aj7iMa8_p4WDhNbQuCFLpl5Hx'
    url = f'https://drive.google.com/uc?id={file_id}'

    output = 'dataset.zip'
    print("Downloading file...")
    gdown.download(url, output, quiet=False)

    print("Extracting file...")
    # Try making the directory
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall('')

    # Remove the zip file
    os.remove(output)

    print("Done.")