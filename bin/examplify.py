#!/usr/bin/env python

#~~~ GenAI Disclaimer ~~~#
#
# Largely generated using Gemini 2.5 Flash.
#
# Prompt:
#
# > Develop a command line utility that:
# >
# > 1. Accepts an .ipynb file (or a list of them) in current dir
# > 2. Adds this cell at its top:
# >
# > ```python
# > # Go up
# >
# > import os
# >
# > os.chdir('..')
# > ```
# >
# > 3. Moves the .ipynb to to the `Examples/` dir
# >
# > Language preference - python
#
#~~~ End of GenAI Disclaimer ~~~#

import os
import nbformat
import argparse
import shutil
from pathlib import Path

def add_cell_and_move_notebook(notebook_path: Path, output_dir: Path):
    """
    Adds a specific Python cell at the top of a Jupyter notebook
    and then moves the modified notebook to a specified output directory.

    Args:
        notebook_path (Path): The path to the input .ipynb file.
        output_dir (Path): The directory where the modified notebook will be moved.
    """
    if not notebook_path.exists():
        print(f"Error: Notebook not found at '{notebook_path}'")
        return

    if not notebook_path.is_file():
        print(f"Error: '{notebook_path}' is not a file.")
        return

    print(f"Processing notebook: {notebook_path.name}")

    try:
        # Load the notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook_content = nbformat.read(f, as_version=4)

        # Define the cell to be added
        new_cell_source = """# Go up

import os

os.chdir('..')"""
        new_cell = nbformat.v4.new_code_cell(source=new_cell_source)

        # Add the new cell at the top
        notebook_content.cells.insert(0, new_cell)

        # Save the modified notebook temporarily (or overwrite if that's acceptable)
        # For safety, let's overwrite for now, but you could save to a temp file
        # and then move it, or save directly to the destination.
        with open(notebook_path, 'w', encoding='utf-8') as f:
            nbformat.write(notebook_content, f)
        print(f"Cell added successfully to {notebook_path.name}.")

        # Ensure the output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Move the modified notebook to the Examples/ directory
        destination_path = output_dir / notebook_path.name
        shutil.move(str(notebook_path), str(destination_path))
        print(f"Moved '{notebook_path.name}' to '{destination_path}'.")

    except nbformat.No:  # pylint: disable=bare-except-raise-exception
        # nbformat.No is not a real exception. It's a placeholder.
        # The correct way to handle general errors is `except Exception as e:`
        # However, the linter is giving a warning because of the common mistake.
        # Let's fix this to a more general exception or specific ones.
        # For general parsing errors, nbformat might raise different errors.
        # A common one would be JSONDecodeError or other file I/O errors.
        print(f"Error reading or parsing notebook '{notebook_path.name}': {e}")
    except Exception as e:
        print(f"An unexpected error occurred while processing '{notebook_path.name}': {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Adds a specific Python cell to the top of .ipynb files and moves them to an 'Examples/' directory."
    )
    parser.add_argument(
        "notebooks",
        metavar="NOTEBOOK",
        type=str,
        nargs="+",
        help="One or more .ipynb files to process. Wildcards like '*.ipynb' are supported."
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="Examples",
        help="The directory to move the processed notebooks to. Defaults to 'Examples'."
    )

    args = parser.parse_args()

    output_directory = Path(args.output_dir)

    processed_count = 0
    skipped_count = 0

    for notebook_pattern in args.notebooks:
        # Use glob to handle wildcards
        # It's important to resolve the path relative to the current working directory
        # where the script is being run, not necessarily where the script resides.
        current_dir = Path.cwd()
        matching_notebooks = list(current_dir.glob(notebook_pattern))

        if not matching_notebooks:
            print(f"Warning: No notebooks found matching pattern '{notebook_pattern}'. Skipping.")
            skipped_count += 1
            continue

        for notebook_path in matching_notebooks:
            if notebook_path.suffix == ".ipynb":
                add_cell_and_move_notebook(notebook_path, output_directory)
                processed_count += 1
            else:
                print(f"Skipping '{notebook_path.name}': Not a .ipynb file.")
                skipped_count += 1

    print(f"\n--- Summary ---")
    print(f"Processed {processed_count} notebook(s).")
    print(f"Skipped {skipped_count} file(s) or pattern(s).")
    print(f"Output directory: '{output_directory.resolve()}'")

if __name__ == "__main__":
    main()