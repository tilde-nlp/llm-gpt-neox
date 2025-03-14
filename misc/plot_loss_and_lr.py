import os
import re
import matplotlib.pyplot as plt


def parse_logs(folder_path):
    # Regex to match the iteration lines and extract relevant fields
    # pattern = re.compile(
    #     r"samples/sec: .*? \| iteration\s+(\d+)\/.*? \| elapsed time per iteration \(ms\): .*? \| learning rate: ([\d.E+-]+) \| .*? lm_loss: ([\d.E+-]+)"
    # )
    pattern = re.compile(
        r"samples/sec: .*? \| iteration\s+(\d+)(?:\s*\[\s*\d+\s*\])?\/.*? \| elapsed time per iteration \(ms\): .*? \| learning rate: ([\d.E+-]+) \| .*? lm_loss: ([\d.E+-]+)"
    )

    val_pattern = re.compile(
        r"iteration (\d+) \| lm_loss value: ([\d.E+-]+) \| lm_loss_ppl value: ([\d.E+-]+)"
    )

    # Dictionary to store data
    data = []
    val_data = []
    # Get all the log files sorted in ascending order by number
    log_files = sorted(
        [f for f in os.listdir(folder_path) if f.startswith("slurm-") and f.endswith(".out")],
        key=lambda x: int(re.search(r"slurm-(\d+).out", x).group(1))
    )

    total_iteration_count = 0

    offsets = [0, 2000, 2000, 20000, 20000, 40000, 40000, 60000,
               60000]  # on restart, we reset "local" iteration number. This brings it back to global.
    # offsets = [0,0,0,0,0,0,0,0,0,0,0,0]
    offsets = [0, 2000]
    offsets = [0, 2000,2000, 20000, 20000]

    # Parse each file
    for n, log_file in enumerate(log_files):
        with open(os.path.join(folder_path, log_file), 'r') as file:
            for line in file:
                match = pattern.search(line)
                if match:
                    # print(match)
                    iteration = int(match.group(1))
                    learning_rate = float(match.group(2))
                    lm_loss = float(match.group(3))

                    # Store the data with the global iteration count
                    data.append({
                        'iteration': iteration + offsets[n],
                        'learning_rate': learning_rate,
                        'lm_loss': lm_loss
                    })
                val_match = val_pattern.search(line)
                if val_match:
                    print(val_match)
                    iteration = int(val_match.group(1))
                    lm_loss = float(val_match.group(2))
                    pp = float(val_match.group(3))
                    # Store the data with the global iteration count
                    val_data.append({
                        'iteration': iteration + offsets[n],
                        'lm_loss': lm_loss,
                        'perplexity': pp
                    })

    return data, val_data


def plot_data(data, val_data):
    # Extract values for plotting
    print(len(data))
    iterations = [entry['iteration'] for entry in data]
    learning_rates = [entry['learning_rate'] for entry in data]
    lm_losses = [entry['lm_loss'] for entry in data]

    # Steps to annotate
    annotate_steps = [0, 1999]

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

    # ------ validation ---- plotting
    print(len(val_data))
    iterations = [entry['iteration'] for entry in val_data]
    lm_losses = [entry['lm_loss'] for entry in val_data]
    print(iterations)

    # Plot lm_loss vs iteration
    plt.figure(figsize=(12, 6))
    plt.plot(iterations, lm_losses, label="LM validation Loss", color="orange")

    # # Annotate points
    # for step in annotate_steps:
    #     lr = lm_losses[step]
    #     plt.text(step, lr, f"{lr:.2e}", fontsize=8, ha="center", color="blue")

    plt.xlabel("Iteration")
    plt.ylabel("LM validation Loss")
    plt.title("LM validation Loss vs Iteration")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ------- PERPLEXITY --------------

    print(len(val_data))
    iterations = [entry['iteration'] for entry in val_data][100:]
    pps = [entry['perplexity'] for entry in val_data][100:]

    # iterations2 = []
    # pps2 = []
    # for i, p in zip(iterations, pps):
    #     if i % 500 == 0:
    #         iterations2.append(i)
    #         pps2.append(p)

    # Plot lm_loss vs iteration
    plt.figure(figsize=(12, 6))
    plt.plot(iterations, pps, label="Perplexity", color="blue")

    # # Annotate points
    # for step in annotate_steps:
    #     lr = lm_losses[step]
    #     plt.text(step, lr, f"{lr:.2e}", fontsize=8, ha="center", color="blue")

    plt.xlabel("Iteration")
    plt.ylabel("LM validation Perplexity")
    plt.title("LM validation Perplexity vs Iteration")
    plt.legend()
    plt.grid(True)
    plt.show()


# Path to the folder containing log files

folder_path = "full_pipe_slurms_merged_fixed_nobias_refix"

# Parse the logs and gather data
data, val_data = parse_logs(folder_path)

# Plot the data
plot_data(data, val_data)
