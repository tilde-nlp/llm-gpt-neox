for d in */; do
    if [ -d "$d" ]; then
        # Run the find/awk command in each directory
        total=$(find "$d" -type f -name '*.bin' -printf '%s\n' \
            | awk '{sum+=$1} END {if (sum == "") sum=0; print sum / 4 / (8192*512)}')
        echo "$d: $total"
    fi
done