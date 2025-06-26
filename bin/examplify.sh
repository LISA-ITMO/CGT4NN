#!/bin/bash

# Configuration
TARGET_DIR="Examples"

# The JSON representation of the cell to be added.
# Note the escaped newlines (\\n) within the source array strings.
NEW_CELL_JSON='
{
  "cell_type": "code",
  "execution_count": null,
  "metadata": {},
  "outputs": [],
  "source": [
    "# Go up\\n",
    "\\n",
    "import os\\n",
    "\\n",
    "os.chdir('\''..\'')\\n"
  ]
}'

# --- Helper Functions ---

# Function to check if jq is installed
check_jq_installed() {
    if ! command -v jq &> /dev/null; then
        echo "Error: 'jq' is not installed."
        echo "Please install it (e.g., 'sudo apt-get install jq' or 'brew install jq')."
        exit 1
    fi
}

# Function to display usage information
display_usage() {
    echo "Usage: $0 <notebook1.ipynb> [notebook2.ipynb ...]"
    echo ""
    echo "  Makes a top-level notebook an example by adding a special"
    echo "  Python cell to the top of the given Jupyter notebooks"
    echo "  and moving them to the '$TARGET_DIR/' directory."
    echo ""
    echo "Examples:"
    echo "  $0 my_notebook.ipynb"
    echo "  $0 notebook1.ipynb notebook2.ipynb"
    echo "  $0 *.ipynb"
    exit 1
}

# --- Main Script Logic ---

# 1. Check dependencies
check_jq_installed

# 2. Check for arguments
if [ "$#" -eq 0 ]; then
    display_usage
fi

# 3. Create the target directory if it doesn't exist
mkdir -p "$TARGET_DIR" || { echo "Error: Could not create directory '$TARGET_DIR'."; exit 1; }

# 4. Process each notebook file
for notebook_file in "$@"; do
    echo "--- Processing: '$notebook_file' ---"

    # Validate file existence
    if [ ! -f "$notebook_file" ]; then
        echo "Warning: File '$notebook_file' not found. Skipping."
        continue
    fi

    # Validate .ipynb extension
    if [[ "$notebook_file" != *.ipynb ]]; then
        echo "Warning: '$notebook_file' does not have a .ipynb extension. Skipping."
        continue
    fi

    # Create a temporary file for the modified content
    TEMP_NOTEBOOK=$(mktemp)

    # Add the cell using jq
    # Explanation:
    #   --argjson new_cell "$NEW_CELL_JSON": Passes the $NEW_CELL_JSON string as a JSON object to jq.
    #   .cells = ([$new_cell] + .cells):
    #     - [$new_cell]: Creates a new array containing just our new cell.
    #     - + .cells: Concatenates this new array with the existing 'cells' array from the notebook.
    #     - .cells = (...): Assigns the result back to the 'cells' key in the notebook's JSON structure.
    if ! jq --argjson new_cell "$NEW_CELL_JSON" '.cells = ([$new_cell] + .cells)' "$notebook_file" > "$TEMP_NOTEBOOK"; then
        echo "Error: Failed to process '$notebook_file' with jq. Is it a valid .ipynb file?"
        rm -f "$TEMP_NOTEBOOK" # Clean up temp file
        continue
    fi

    # Move the modified notebook to the Examples directory
    # basename "$notebook_file" ensures we just get the filename without its path.
    TARGET_PATH="$TARGET_DIR/$(basename "$notebook_file")"
    if ! mv "$TEMP_NOTEBOOK" "$TARGET_PATH"; then
        echo "Error: Failed to move '$notebook_file' to '$TARGET_DIR/'. Permission issue?"
        rm -f "$TEMP_NOTEBOOK" # Clean up temp file if move failed
        continue
    fi

    echo "Success: '$notebook_file' updated and moved to '$TARGET_PATH'."
    echo ""
done