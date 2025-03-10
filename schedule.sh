#!/bin/bash

# Check if a primary job script, number of jobs, and bench script are provided as arguments
if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <job_script.sh> <n> <bench_script.sh>"
    exit 1
fi

# Get the primary job script, number of jobs, and bench job script from arguments
job_script="$1"
n="$2"
bench_script="$3"

# Check if the primary job script exists and is executable
if [[ ! -f "$job_script" ]]; then
    echo "Error: Job script '$job_script' not found."
    exit 1
fi

if [[ ! -x "$job_script" ]]; then
    echo "Warning: Job script '$job_script' is not executable. Making it executable."
    chmod +x "$job_script"
fi

# Check if the bench job script exists and is executable
if [[ ! -f "$bench_script" ]]; then
    echo "Error: Bench script '$bench_script' not found."
    exit 1
fi

if [[ ! -x "$bench_script" ]]; then
    echo "Warning: Bench script '$bench_script' is not executable. Making it executable."
    chmod +x "$bench_script"
fi

# Check if n is a positive integer
if ! [[ "$n" =~ ^[0-9]+$ ]] || [[ "$n" -le 0 ]]; then
    echo "Error: n must be a positive integer."
    exit 1
fi

# Initialize the previous primary job ID to none
prev_job_id=""

# Log file
log_file="job_log.txt"

# Clear the log file
> "$log_file"

# Loop to submit n primary jobs
for (( i=1; i<=n; i++ )); do
    # Submit the primary job
    if [[ -z "$prev_job_id" ]]; then
        # Submit the first primary job without dependency
        primary_job_id=$(sbatch --parsable "$job_script")
    else
        # Submit subsequent primary jobs with dependency on the previous one
        primary_job_id=$(sbatch --dependency=afterok:$prev_job_id --parsable "$job_script")
    fi

    # Check if primary job submission was successful
    if [[ -n "$primary_job_id" ]]; then
        echo "$(date): Primary Job $i submitted with Job ID: $primary_job_id" | tee -a "$log_file"
    else
        echo "$(date): Failed to submit Primary Job $i" | tee -a "$log_file"
        exit 1
    fi

    # Submit the bench job that depends on the current primary job finishing
    bench_job_id=$(sbatch --dependency=afterok:$primary_job_id --parsable "$bench_script")

    if [[ -n "$bench_job_id" ]]; then
        echo "$(date): Bench job for Primary Job $i submitted with Job ID: $bench_job_id" | tee -a "$log_file"
    else
        echo "$(date): Failed to submit Bench job for Primary Job $i" | tee -a "$log_file"
        exit 1
    fi

    # Update the previous primary job ID for chaining
    prev_job_id="$primary_job_id"
done

echo "$(date): All jobs submitted successfully." | tee -a "$log_file"
