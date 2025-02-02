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

# List of libraries for which to retrieve the version information.
LIBRARIES = (
    "symusic",
    "midifile_cpp",
    "mido",
    "miditoolkit",
    "pretty_midi",
    "partitura",
    "music21",
)

def get_versions() -> Dict[str, str]:
    """
    Retrieve the installed version for each library in LIBRARIES.

    Returns:
        A dictionary mapping library names to their installed versions.
    """
    versions = {}
    for lib in LIBRARIES:
        try:
            versions[lib] = version(lib)
        except PackageNotFoundError:
            # If the library is not installed, skip it.
            pass
    return versions

def main() -> None:
    """
    Parse command-line arguments, collect version information, and save it as a JSON file.
    """
    # Define and parse the command-line argument for the output file path.
    parser = argparse.ArgumentParser(
        description="Log the version of the libraries."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./version.json",
        help="Output file path",
    )
    args = parser.parse_args()

    # Retrieve versions for the specified libraries.
    library_versions = get_versions()

    # Obtain the Python version using the platform module.
    python_version = platform.python_version()

    # Get the current time in ISO 8601 format.
    current_time = datetime.now().isoformat()

    # Prepare the dictionary containing all the version information.
    info = {
        "libraries": library_versions,
        "python": python_version,
        "time": current_time,
    }

    print(f"Library version information will be saved in '{args.output}'.")

    # Ensure the output directory exists (if specified).
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Write the version information to the specified JSON file.
    with open(args.output, 'w') as f:
        json.dump(info, f, indent=4)

    # Pretty print the information to the console.
    pprint.pprint(info)

if __name__ == "__main__":
    main()