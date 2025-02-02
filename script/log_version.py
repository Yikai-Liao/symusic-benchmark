import json
import os
from typing import Dict
import argparse
import pprint

LIBRARIES = ("symusic", "midifile_cpp", "mido", "miditoolkit", "pretty_midi", "partitura", "music21")


def get_version()->Dict[str, str]:
    version = {}
    # Get the version of the libraries
    for lib in LIBRARIES:
        try:
            version[lib] = __import__(lib).__version__
        except ImportError:
            pass
    return version

# Define the parser
parser = argparse.ArgumentParser(description="Log the version of the libraries")
# With only one argument, the output path
parser.add_argument("--output", type=str, default="./version.json", help="Output path")

if __name__ == '__main__':
    args = parser.parse_args()
    version = get_version()
    info = {
        "libraries": version,
        "python": f"{os.popen('python --version').read().strip().replace('Python ', '')}",
        "time": f"{os.popen('date').read().strip()}"
    }
    print(f"Version of the libraries saved in {args.output}")

    # Try making the directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(info, f, indent=4)
    pprint.pprint(info)