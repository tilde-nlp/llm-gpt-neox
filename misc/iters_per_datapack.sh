#!/bin/bash

# Check if the root directory argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <root_directory>"
    exit 1
fi

ROOT_DIR="$1"

# Iterate over subdirectories in the specified root directory
for d in "$ROOT_DIR"/*/; do
    if [ -d "$d" ]; then
        # Run the find/awk command in each directory
        total=$(find "$d" -type f -name '*.bin' -printf '%s\n' \
            | awk '{sum+=$1} END {if (sum == "") sum=0; print sum / 4 / (8192*576)}')
        echo "$d: $total"
    fi
done