"""
This script logs the version information of specified libraries, along with the Python version
and the current time, then saves the information to a JSON file.
"""

import json
import os
import argparse
import pprint
from typing import Dict
from importlib.metadata import version, PackageNotFoundError
import platform
from datetime import datetime
import sys
import pkg_resources

# List of libraries for which to retrieve the version information.
LIBRARIES = (
    "symusic",
    "midifile_cpp",
    "mido",
    "miditoolkit",
    "pretty_midi",
    "partitura",
    "music21",
    "numba_midi",
    "midi_jl",
    "tone_js",
)

# Mapping for libraries with different registry names or special handling
LIBRARY_MAP = {
    "midifile_cpp": "midifile",
    "midi_jl": "MIDI",        # Julia package name
    "tone_js": "@tonejs/midi" # npm package name
}

def get_versions() -> Dict[str, str]:
    """
    Retrieve the installed version for each library in LIBRARIES.

    Returns:
        A dictionary mapping library names to their installed versions.
    """
    versions = {}
    for lib in LIBRARIES:
        package_name_to_check = LIBRARY_MAP.get(lib, lib)

        if lib == "midi_jl":
            versions[lib] = "Checked via julia-actions/setup-julia"
        elif lib == "tone_js":
            versions[lib] = "Checked via actions/setup-node"
        else:
            try:
                # Use importlib.metadata for Python packages
                versions[lib] = version(package_name_to_check)
            except PackageNotFoundError:
                versions[lib] = None # Library not found
            except Exception as e:
                versions[lib] = f"Error: {e}"
    return versions

def main() -> None:
    """
    Parse command-line arguments, collect version information, and save it as a JSON file.
    """
    parser = argparse.ArgumentParser(
        description="Log the version of the libraries."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSON file path.",
    )
    args = parser.parse_args()

    library_versions = get_versions()
    python_version_detailed = sys.version
    platform_info = platform.platform()
    current_time = datetime.now().isoformat()

    info = {
        "python_version": python_version_detailed,
        "platform_info": platform_info,
        "libraries": library_versions,
        "time": current_time,
    }

    print(f"Version information saved to {args.output}")

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output, 'w') as f:
        json.dump(info, f, indent=4)

    pprint.pprint(info)

if __name__ == "__main__":
    main()