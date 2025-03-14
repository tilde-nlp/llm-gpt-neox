import sys
import json
import numpy as np
import matplotlib
from matplotlib import pyplot as plt


# reads json file of the structure
# {"lang": List[str], "U1" : {"short": List[str], "long": List[str]}, ....}
def read_json_file(filename):
    with open(filename) as json_file:
        data = json.load(json_file)
    return data

# this is mainly gpt written, so proceed with care
def plot_data_distribution_long_short_separate(data, scale_to_b=10 ** 9):
    # Get categories
    categories = data['names']

    # Extract phase data and scale
    u1_short = np.array(list(map(float, data['U1']["short"]))) / scale_to_b
    u1_long = np.array(list(map(float, data['U1']["long"]))) / scale_to_b

    u2_short = np.array(list(map(float, data['U2']["short"]))) / scale_to_b
    u2_long = np.array(list(map(float, data['U2']["long"]))) / scale_to_b

    n_short = np.array(list(map(float, data['N']["short"]))) / scale_to_b
    n_long = np.array(list(map(float, data['N']["long"]))) / scale_to_b

    u3_short = np.array(list(map(float, data['U3']["short"]))) / scale_to_b
    u3_long = np.array(list(map(float, data['U3']["long"]))) / scale_to_b

    ext_short = np.array(list(map(float, data['Ext']["short"]))) / scale_to_b
    ext_long = np.array(list(map(float, data['Ext']["long"]))) / scale_to_b

    # Compute totals for percentages
    sum1 = np.sum(u1_short + u1_long)
    sum2a = np.sum(u2_short + u2_long)
    sum2b = np.sum(n_short + n_long)
    sum3 = np.sum(u3_short + u3_long)
    total_sum = sum1 + sum2a + sum2b + sum3

    sum_ext = np.sum(ext_short + ext_long)

    # Compute totals for percentages
    total_short = u1_short + u2_short + n_short + u3_short
    total_long = u1_long + u2_long + n_long + u3_long

    # Setup figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    def plot_stacked_bars(ax, categories, short, long, color, hatch_color, label_short, label_long, title):
        """Helper function to plot stacked bars and add labels."""
        bars_short = ax.bar(categories, short, color=color, label=label_short)
        bars_long = ax.bar(categories, long, color=color, label=label_long, bottom=short, hatch='///',
                           edgecolor=hatch_color)

        # Add numerical labels to bars
        for bars in [bars_short, bars_long]:
            for bar in bars:
                height = bar.get_height()
                bottom = bar.get_y()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,  # X position (center of bar)
                    bottom + height / 2,  # Y position (middle of bar)
                    f'{height:.1f}',  # Label text
                    ha='center', va='center', color='black', fontsize=8, rotation=45
                )

        ax.set_title(title)
        ax.legend()

    # Plot U1
    title1 = f"U1: {sum1:.0f}B [{100 * sum1 / total_sum:.2f}%]"
    plot_stacked_bars(axes[0], categories, u1_short, u1_long, 'skyblue', 'royalblue', 'U1_short', 'U1_long', title1)

    # Plot U2 + N
    title2 = f"U2+N: {sum2a:.0f}B + {sum2b:.0f}B [{100 * sum2a / total_sum:.0f}%, {100 * sum2b / total_sum:.0f}%]"
    bottom_stack = np.zeros_like(u2_short)
    axes[1].bar(categories, u2_short, color='salmon', label='U2_short', bottom=bottom_stack)
    bottom_stack += u2_short
    axes[1].bar(categories, u2_long, color='salmon', label='U2_long', bottom=bottom_stack, hatch='///',
                edgecolor='darkred')
    bottom_stack += u2_long
    axes[1].bar(categories, n_short, color='orange', label='N_short', bottom=bottom_stack)
    bottom_stack += n_short
    axes[1].bar(categories, n_long, color='orange', label='N_long', bottom=bottom_stack, hatch='///',
                edgecolor='darkorange')
    axes[1].set_title(title2)
    axes[1].legend()

    # Plot U3
    title3 = f"U3: {sum3:.0f}B [{100 * sum3 / total_sum:.2f}%]"
    plot_stacked_bars(axes[2], categories, u3_short, u3_long, 'lightgreen', 'forestgreen', 'U3_short', 'U3_long',
                      title3)

    # Formatting
    for ax in axes:
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories, rotation=45, ha='right')

    plt.tight_layout()
    plt.show()

    # --- EXTRA SEPARATE PLOTS ---
    def create_separate_plot(categories, short, long, color, hatch_color, label_short, label_long, title, grey=False):
        matplotlib.use("TkAgg")
        if grey:
            plt.rcParams.update({
                "figure.facecolor": "#3E3E3E",  # Dark grey background
                "axes.facecolor": "#4F4F4F",  # Slightly lighter grey for plot background
                "axes.edgecolor": "white",  # White border
                "xtick.color": "white",  # White x-axis ticks
                "ytick.color": "white",  # White y-axis ticks
                "text.color": "white",  # White text
                "grid.color": "#6E6E6E",  # Light grey grid lines
                "legend.facecolor": "#5F5F5F"  # Medium grey for legend background
            })
        fig, ax = plt.subplots(figsize=(24, 12))
        plot_stacked_bars(ax, categories, short, long, color, hatch_color, label_short, label_long, title)
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories, rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    # Separate U1 Plot
    create_separate_plot(categories, u1_short, u1_long, 'skyblue', 'royalblue', 'U1_short', 'U1_long', title1)

    # Separate U2 + N Plot
    fig, ax = plt.subplots(figsize=(24, 12))
    bottom_stack = np.zeros_like(u2_short)
    bars2a = ax.bar(categories, u2_short, color='salmon', label='U2_short', bottom=bottom_stack)
    bottom_stack += u2_short
    bars2b = ax.bar(categories, u2_long, color='salmon', label='U2_long', bottom=bottom_stack, hatch='///', edgecolor='darkred')
    bottom_stack += u2_long
    bars3a = ax.bar(categories, n_short, color='orange', label='N_short', bottom=bottom_stack)
    bottom_stack += n_short
    bars3b = ax.bar(categories, n_long, color='orange', label='N_long', bottom=bottom_stack, hatch='///', edgecolor='darkorange')
    ax.set_title(title2)
    ax.legend()
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=45, ha='right')

    # Add labels
    for bars in [bars2a, bars2b, bars3a, bars3b]:
        for bar in bars:
            height = bar.get_height()
            bottom = bar.get_y()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bottom + height / 2,
                f'{height:.1f}',
                ha='center', va='center', color='black', fontsize=8, rotation=45
            )

    plt.tight_layout()
    plt.show()

    # Separate U3 Plot
    create_separate_plot(categories, u3_short, u3_long, 'lightgreen', 'forestgreen', 'U3_short', 'U3_long', title3)

    title_ext = f"Extension phase - {np.sum(ext_short):.0f}B Short + {np.sum(ext_long):.0f}B Long [{100 * np.sum(ext_short) / sum_ext:.2f}% Short, {100 * np.sum(ext_long) / sum_ext:.2f}% Long]"
    create_separate_plot(categories, ext_short, ext_long, 'lightpink', '#C71585', 'ext_short', 'ext_long', title_ext)

    # Separate TOTAL Plot (All phases, Short and Long separated)
    title4 = f"Total: {np.sum(total_short):.0f}B Short + {np.sum(total_long):.0f}B Long [{100 * np.sum(total_short) / total_sum:.2f}% Short, {100 * np.sum(total_long) / total_sum:.2f}% Long]"
    create_separate_plot(categories, total_short, total_long, 'peachpuff', '#D2691E', 'Total_Short', 'Total_Long', title4)


# this is actually human written, as evidenced by lack of comments
def plot_data_distribution_long_short(data, scale_to_b=10 ** 9):
    # get lang ids
    categories = data['names']

    # extract phase data
    u1_short = np.array(list(map(float, data['U1']["short"]))) / scale_to_b
    u1_long = np.array(list(map(float, data['U1']["long"]))) / scale_to_b

    u2_short = np.array(list(map(float, data['U2']["short"]))) / scale_to_b
    u2_long = np.array(list(map(float, data['U2']["long"]))) / scale_to_b

    n_short = np.array(list(map(float, data['N']["short"]))) / scale_to_b
    n_long = np.array(list(map(float, data['N']["long"]))) / scale_to_b

    u3_short = np.array(list(map(float, data['U3']["short"]))) / scale_to_b
    u3_long = np.array(list(map(float, data['U3']["long"]))) / scale_to_b

    ext_short = np.array(list(map(float, data['Ext']["short"]))) / scale_to_b
    ext_long = np.array(list(map(float, data['Ext']["long"]))) / scale_to_b

    # composite phase data
    u1 = u1_short + u1_long
    u2 = u2_short + u2_long
    n = n_short + n_long
    u3 = u3_short + u3_long
    ext = ext_short + ext_long

    n_composite = u2 + n

    # Define fixed horizontal lines
    line1_y = np.max(u1)
    line2_y = np.max(u1 + n_composite)

    # Compute offsets for stacked bars, considering alignment with the horizontal lines
    y1a_offset = np.zeros_like(u1_short)  # dataset 1.1 (u1_short) starts from 0
    y1b_offset = y1a_offset + u1_short  # dataset 1.2 is stacked on top of 1.1

    y2a_offset = np.maximum(line1_y, y1b_offset)  # First part of Dataset 2 starts at line1_y or above Dataset 1
    y2b_offset = y2a_offset + u2_short  # Second part of Dataset 2 is stacked on top of the first part

    y3a_offset = y2b_offset + u2_long  # 3.1 is stacked on top of 2.2
    y3b_offset = y3a_offset + n_short  # 3.2 is stacked on top of 3.1

    y4a_offset = np.maximum(line2_y, y3b_offset)  # Dataset 4.1 starts at line2_y or above Dataset 3
    y4b_offset = y4a_offset + u3_short  # Dataset 4.2 is stacked on top of 3.1

    x = np.arange(len(categories))  # Base positions

    # Create the plot
    matplotlib.use("TkAgg")
    fig, ax = plt.subplots(figsize=(24, 12))

    # Plot Dataset 1 (as two stacked components)
    bars1a = ax.bar(categories, u1_short, color='skyblue', label='U1_short', bottom=y1a_offset)
    bars1b = ax.bar(categories, u1_long, color='skyblue', label='U1_long', bottom=y1b_offset, hatch='///',
                    edgecolor='royalblue')

    # Plot Dataset 2 (as two stacked components)
    bars2a = ax.bar(categories, u2_short, color='salmon', label='U2_short', bottom=y2a_offset)
    bars2b = ax.bar(categories, u2_long, color='salmon', label='U2_long', bottom=y2b_offset, hatch='///',
                    edgecolor='darkred')

    # Plot Dataset 3 (as two stacked components)
    bars3a = ax.bar(categories, n_short, color='orange', label='N_short', bottom=y3a_offset)
    bars3b = ax.bar(categories, n_long, color='orange', label='N_long', bottom=y3b_offset, hatch='///',
                    edgecolor='darkorange')

    # Plot Dataset 4 (ast two stacked components)
    bars4a = ax.bar(categories, u3_short, color='lightgreen', label='U3_short', bottom=y4a_offset)
    bars4b = ax.bar(categories, u3_long, color='lightgreen', label='U3_long', bottom=y4b_offset, hatch='///',
                    edgecolor='forestgreen')

    # Add labels for each bar
    for bars in [bars1a, bars1b, bars2a, bars2b, bars3a, bars3b, bars4a, bars4b]:
        for bar in bars:
            height = bar.get_height()
            bottom = bar.get_y()
            ax.text(
                bar.get_x() + bar.get_width() / 2,  # X position (center of bar)
                bottom + height / 2,  # Y position (middle of bar)
                f'{height:.1f}',  # Label text
                ha='center', va='center', color='black', fontsize=8, rotation=45
            )

    # Draw horizontal lines at y=8 and y=16
    for y_line in [line1_y, line2_y]:
        ax.axhline(y=y_line, color='black', linestyle='--', linewidth=2)

    # Compute sums
    sum1 = np.sum(u1)
    sum2a = np.sum(u2)
    sum2b = np.sum(n)
    sum3 = np.sum(u3)
    total_sum = sum1 + sum2a + sum2b + sum3  # Compute the total

    # Formatting
    ax.set_ylabel('Total tokens')
    ax.set_title(
        f"U1: {sum1:.0f}B [{100 * sum1 / total_sum:.2f}%]       U2+N: {sum2a:.0f}B + {sum2b:.0f}B [{100 * sum2a / total_sum:.2f}%,{100 * sum2b / total_sum:.2f}%]     U3: {sum3:.0f}B [{100 * sum3 / total_sum:.2f}%]       ∑: {total_sum:.0f}B")
    ax.legend(loc='best')

    # Rotate x-axis labels
    ax.set_xticks(range(len(categories)))
    plt.xticks(rotation=45, ha='right')  # 'ha' ensures labels are right-aligned

    # Show plot
    plt.tight_layout()
    plt.show()

    return 0  # returns 0


def plot_data_distrbution(data):

    return 0  # returns 0


def main(json_in):
    data = read_json_file(json_in)



    #plot_data_distribution_long_short(data)
    plot_data_distribution_long_short_separate(data)


if __name__ == '__main__':
    path_to_json_in = sys.argv[1]
    main(path_to_json_in)
