import json
import math
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import zipfile

def parse_args():
  parser = argparse.ArgumentParser(
    description="Plots outputs of needle_tester.py"
  )
  
  parser.add_argument("--results",
                      type=str,
                      required=True,
                      help="Path to folder with needle_tester.py output")
  
  parser.add_argument("--baseline",
                      type=str,
                      default=None,
                      help="Path to folder with needle_tester.py output for a baseline model. " +
                           "This is optional. Graphs will be differences rather than absolute values, " +
                           "if provided.")
  
  parser.add_argument("--needle-sets",
                      type=str,
                      default=None,
                      help="String describing needle subsets. Example: '0,1,4|2,3|5' " +
                           "will plot one average for each of the needle subsets {0, 1, 4}, {2, 3} and {5} respectively. " + 
                           "This is useful if you have different types of needles.")
  
  parser.add_argument("--output-folder",
                      type=str,
                      default="./plots/",
                      help="Output folder to save to. (Will create folder path)")

  parser.add_argument("--scatter-averaging",
                      type=int,
                      default=7,
                      help="How many points to average over in the scatter plot when drawing the line. Defaults to 7.")

  parser.add_argument("--zip",
                      action="store_true",
                      default=False,
                      help="If flag is used, then will create a zip file with the contents of the output folder at --output-folder + '.zip'.")

  args = parser.parse_args()
  
  # Verify args.
  
  return args
  

class NeedleData():
  """
  Just a class instended to store needle_tester.py results.
  """
  def __init__(s, folder_path):
    s.results_path = os.path.join(folder_path, "results.json")
    s.debug_path = os.path.join(folder_path, "debug.json")
    
    with open(s.results_path, "r") as f:
      obj = json.load(f)
    
    s.scores = np.array(obj["scores"]) #[context, depth, hay, needle]
    s.target_depths = obj["target_depths"]
    s.target_contexts = obj["target_contexts"]
    s.needles = obj["needles"]
    s.hay_count = s.scores.shape[2]
    s.needle_count = s.scores.shape[3]


def plot_abs(heatmap, target_contexts, target_depths, title, save_path):
  """
  Function intended for drawing needle performance heatmaps in 0..1 range.
  """
  
  fig = plt.Figure(figsize=(6, 4))
  ax = fig.add_subplot(111)

  # Draw the image
  im = ax.imshow(heatmap)

  # Colorbar – must be created from the figure, not pyplot
  cbar = fig.colorbar(im, ax=ax)

  # Y‑axis ticks/labels
  ax.set_yticks(np.linspace(0, len(target_contexts) - 1, len(target_contexts)))
  ax.set_yticklabels([str(i) for i in target_contexts])
  ax.set_ylabel("Context Size")

  # X‑axis ticks/labels
  ax.set_xticks(np.linspace(0, len(target_depths) - 1, len(target_depths)))
  ax.set_xticklabels([f"{d:.2f}" for d in target_depths])
  ax.set_xlabel("Needle Position")

  # Title
  ax.set_title(title)

  # Save to disk
  fig.savefig(save_path, bbox_inches="tight")


def plot_diff(diff, target_contexts, target_depths, title, save_path):
  """
  Function intended for drawing the difference between two needle heatmaps of two different models.
  Should be in range -1..1
  """
  
  fig = plt.Figure(figsize=(6, 4))
  ax = fig.add_subplot(111)

  # Get a fancy red -> white -> green colorbar.
  cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "insert name here",
    ["red", "white", "green"],
    N=256
  )
  range_ = max(np.max(np.abs(diff)), 0.1)
  norm = matplotlib.colors.TwoSlopeNorm(vmin=-range_, vcenter=0, vmax=range_)

  # Draw the image
  im = ax.imshow(diff, cmap=cmap, norm=norm)

  # Colorbar – must be created from the figure, not pyplot
  cbar = fig.colorbar(im, ax=ax)

  # Y‑axis ticks/labels
  ax.set_yticks(np.linspace(0, len(target_contexts) - 1, len(target_contexts)))
  ax.set_yticklabels([str(i) for i in target_contexts])
  ax.set_ylabel("Context Size")

  # X‑axis ticks/labels
  ax.set_xticks(np.linspace(0, len(target_depths) - 1, len(target_depths)))
  ax.set_xticklabels([f"{d:.2f}" for d in target_depths])
  ax.set_xlabel("Needle Position")

  # Title
  ax.set_title(title)

  # Save to disk
  fig.savefig(save_path, bbox_inches="tight")


def _t95(df: int) -> float:
  "Hardcoded t statistic, because I don't want more dependencies"
  _tbl = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045
  }
  return _tbl.get(df, 1.960)   # z‑score takeover for df ≥ 30


def scatter(
        heatmap,
        target_contexts,
        target_depths,
        title,
        save_path,
        window=5,
        measurement_error=0.05
):
    """
    Scatter plot of retrieval performance vs. needle distance, with an
    overlaid sliding‑window average.
    """
    # Get points.
    distances = []
    performances = []
    
    for ctx_idx, ctx in enumerate(target_contexts):
        for dpt_idx, dpt in enumerate(target_depths):
            distances.append((1 - dpt) * ctx)
            performances.append(heatmap[ctx_idx, dpt_idx])

    # Plot points
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(distances, performances, alpha=0.6, label="Context-Depth Pairs")

    # Get moving average. splides from left to right, averages over
    if len(distances) >= window:
        # Sort the points.
        order = np.argsort(distances)
        dist_sorted = np.asarray(distances)[order]
        perf_sorted = np.asarray(performances)[order]

        # Convolution (i.e. same as just taking the average over that window.)
        kernel = np.ones(window) / window
        dist_avg = np.convolve(dist_sorted, kernel, mode="valid")
        perf_avg = np.convolve(perf_sorted, kernel, mode="valid")

        # Plot the line.
        ax.plot(dist_avg, perf_avg, linewidth=2,
                label=f"{window}-Point Moving Average", color="orange")

        # Geep's code for plotting a 95% conf interval
        stderr = [
            perf_sorted[i:i + window].std(ddof=1) / np.sqrt(window)
            for i in range(len(perf_sorted) - window + 1)
        ]
        # For account for measurement error.
        stderr = np.sqrt(np.asarray(stderr) ** 2 + (measurement_error ** 2) / (12 * window)) 
        # For accounting for smol sample sizes.
        t_mult = _t95(window - 1) 
        ci95 = t_mult * stderr
        
        # Draw confidence intervals
        ax.fill_between(
            dist_avg,
            perf_avg - ci95,
            perf_avg + ci95,
            color="orange",
            alpha=0.25,
            linewidth=0,
            label="95 % CI"
        )

    # Add labels
    ax.set_xlabel("Needle Distance")
    ax.set_ylabel("Retrieval Percent")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    # Save.
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def main():
  args = parse_args()
  
  # Make output folder.
  output_folder = args.output_folder
  os.makedirs(output_folder, exist_ok=True)

  # Load data
  results = NeedleData(args.results)
  
  if args.baseline is not None:
    # We will be comparing two models.
    baseline = NeedleData(args.baseline)
    data = results.scores - baseline.scores
    comparison = True
  else:
    # We will just plot the results for a single model.
    data = results.scores
    comparison = False

  target_contexts = results.target_contexts
  target_depths = results.target_depths
  needle_count = results.needle_count
  hay_count = results.hay_count
  
  # Perform plotting and metrics
  metrics_text = ""
  metrics_json = {}

  metrics_text+= "Metrics calculated for setup:\n"
  metrics_text+= str(args.__dict__) + "\n"
  metrics_text+= "\n---------------------------"
  metrics_text+="\n\n"
  metrics_json["setup"] = args.__dict__

  ## Heatmap plots.
  ### Functional programming hack.
  if not comparison:
    plot = plot_abs
  else:
    plot = plot_diff
  
  
  ### Plot heatmap of average over hay and needle.
  tmp_data = np.mean(np.mean(data, axis=3), axis=2)
  plot(tmp_data, target_contexts, target_depths, "Average", 
       os.path.join(output_folder, "Mean.png"))
       
  avg = np.mean(tmp_data)
  metrics_text+= "Average over all needles and hay (avg_all): " + str(avg)[:5] + "\n\n\n"
  metrics_json["avg_all"] = avg
  
  ### Plot heatmap by needle.
  for needle_idx in range(needle_count):
    tmp_data = np.mean(data, axis=2)[:, :, needle_idx]
    plot(tmp_data, 
         target_contexts, 
         target_depths, 
         results.needles[needle_idx]["needle"], 
         os.path.join(output_folder, "Needle" + str(needle_idx) + ".png"))
  
    avg = np.mean(tmp_data)
    key = "needle_" + str(needle_idx)
    metrics_text+= "Average over needle " + str(needle_idx) + " (" + key + "): " + str(avg)[:5] + "\n"
    metrics_text+= "Needle: " + results.needles[needle_idx]["needle"] + "\n"
    metrics_text+= "Prompt: " + results.needles[needle_idx]["answer_prompt"] + "\n\n"
    metrics_json[key] = avg
  metrics_text+= "\n"
  
  ### Plot by haystack.
  for hay_idx in range(hay_count):
    tmp_data = np.mean(data, axis=3)[:, :, hay_idx]
    plot(tmp_data,
         target_contexts,
         target_depths,
         "Haystack #" + str(hay_idx),
         os.path.join(output_folder, "Haystack" + str(hay_idx) + ".png"))
  
    avg = np.mean(tmp_data)
    key = "hay_" + str(hay_idx)
    metrics_text+= "Average over hay " + str(hay_idx) + " (" + key + "): " + str(avg)[:5] + "\n"
    metrics_json[key] = avg
  metrics_text+= "\n"
  
  ### Plot needle subsets.
  #### First parse the subsets.
  if args.needle_sets is not None:
    needle_sets = []
    for set_ in args.needle_sets.split("|"):
      tmp = [int(needleidx) for needleidx in set_.split(",")]
      needle_sets.append(tmp)
    
    #### Now the actual subset calculation and plot.
    for set_idx, set_ in enumerate(needle_sets):
      # Calculate mean.
      result = 0
      for needle_idx in set_:
        result+= data[:, :, :, needle_idx]
      result/= len(set_)
    
      # Plot.
      plot(np.mean(result, axis=2), 
           target_contexts, 
           target_depths, 
           "Needle set: " + str(set_), 
           os.path.join(output_folder, "Needleset" + str(set_idx) + ".png"))

      avg = np.mean(result)
      key = "needle_set_" + str(hay_idx)
      metrics_text+= "Average over needle set " + str(set_idx) + " (" + key + "): " + str(avg)[:5] + "\n"
      metrics_json[key] = avg
    metrics_text+= "\n"
  
  ## Scatters
  scatter(np.mean(np.mean(data, axis=3), axis=2), target_contexts, target_depths, 
          "Distance-Performance", 
          os.path.join(output_folder, "scatter.png"), args.scatter_averaging,
          1 / needle_count / hay_count)
  
  # Print readable and json to files.
  with open(os.path.join(output_folder, "Metrics.txt"), "w") as f:
    print(metrics_text, file=f)
  
  with open(os.path.join(output_folder, "Metrics.json"), "w") as f:
    json.dump(metrics_json, f)
  
  print(metrics_text)
  
  # Zip outputs.
  if args.zip:
    # Get output file name.
    if output_folder[-1] in ["/", "\\"]:
      zip_path = output_folder[:-1] + ".zip"
    else:
      zip_path = output_folder + ".zip"
      
      # Create zip file
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
      for root, _, files in os.walk(output_folder):
        for file in files:
          file_path = os.path.join(root, file)
          zip_file_path = file
          zf.write(file_path, zip_file_path)


if __name__ == "__main__":
  main()
