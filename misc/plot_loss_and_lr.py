import os
import re
import matplotlib.pyplot as plt

def parse_logs(folder_path):
    # Regex to match the iteration lines and extract relevant fields
    pattern = re.compile(
        r"samples/sec: .*? \| iteration\s+(\d+)\/.*? \| elapsed time per iteration \(ms\): .*? \| learning rate: ([\d.E+-]+) \| .*? lm_loss: ([\d.E+-]+)"
    )

    # Dictionary to store data
    data = []

    # Get all the log files sorted in ascending order by number
    log_files = sorted(
        [f for f in os.listdir(folder_path) if f.startswith("slurm-") and f.endswith(".out")],
        key=lambda x: int(re.search(r"slurm-(\d+).out", x).group(1))
    )

    total_iteration_count = 0

    # Parse each file
    for log_file in log_files:
        with open(os.path.join(folder_path, log_file), 'r') as file:
            for line in file:
                match = pattern.search(line)
                if match:
                    print(match)
                    iteration = int(match.group(1))
                    learning_rate = float(match.group(2))
                    lm_loss = float(match.group(3))

                    # Store the data with the global iteration count
                    data.append({
                        'iteration': iteration,
                        'learning_rate': learning_rate,
                        'lm_loss': lm_loss
                    })

    return data

def plot_data(data):
    # Extract values for plotting
    print(len(data))
    iterations = [entry['iteration'] for entry in data]
    learning_rates = [entry['learning_rate'] for entry in data]
    lm_losses = [entry['lm_loss'] for entry in data]

    # Steps to annotate
    annotate_steps = [0, 1999, 4999, 9999, 12999]
    #annotate_steps= [0, 1999, 2055]

    # Plot learning rate vs iteration
    plt.figure(figsize=(12, 6))
    plt.plot(iterations, learning_rates, label="Learning Rate", color="blue")

    # Annotate points
    for step in annotate_steps:
        lr = learning_rates[step]
        plt.text(step, lr, f"{lr:.2e}", fontsize=8, ha="center", color="red")

    plt.xlabel("Iteration")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate vs Iteration")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot lm_loss vs iteration
    plt.figure(figsize=(12, 6))
    plt.plot(iterations, lm_losses, label="LM Loss", color="red")

    # Annotate points
    for step in annotate_steps:
        lr = lm_losses[step]
        plt.text(step, lr, f"{lr:.2e}", fontsize=8, ha="center", color="blue")



    plt.xlabel("Iteration")
    plt.ylabel("LM Loss")
    plt.title("LM Loss vs Iteration")
    plt.legend()
    plt.grid(True)
    plt.show()

# Path to the folder containing slurm log files
# TODO: add as arg
folder_path = "full_pipe_slurms_merged"

# Parse the logs and gather data
data = parse_logs(folder_path)

# Plot the data
plot_data(data)
