import argparse
import torch
import os
import re
import shutil

def parse_args():
  parser = argparse.ArgumentParser(
    description="Creates new GPT-NeoX checkpoints with averaged RMSNorm scale/weight tensors between TP. " +
                "Doesn't work for all possible neox configs. But works for our 30B model. " +
                "Also note the code is hardcoded for bf16 checkpoints."
  )
  
  parser.add_argument(
    "--input-folder",
    type=str,
    required=True,
    help="Path to checkpoint folder (e.g. /a/b/c/global_step1337)."
  )
  
  parser.add_argument(
    "--output-folder",
    type=str,
    default=None,
    help="Optional path for output. If not provided will just overwrite the checkpoint files."
  )
  
  return parser.parse_args()

def zero_pad(num):
  if num < 10:
    return "0" + str(num)
  else:
    return str(num)

def main():
  args = parse_args()
  
  # Detect number of ranks.
  print("Detecting rank count. :3", flush=True)
  highest_rank = -1
  for file_name in os.listdir(args.input_folder):
    match = re.match(r"^mp_rank_(\d+)_model_states\.pt$", file_name)
    if match:
      rank = int(match.group(1))
      if rank > highest_rank:
        highest_rank = rank
        highest_file = file_name
  rank_count = highest_rank + 1
  
  if highest_rank == -1:
    raise Exception("Didn't find any mp_rank_XX_model_states.pt file in the --input-folder!")
  
  print("Detected " + str(rank_count) + " ranks. :3", flush=True)
  

  # Load checkpoint ranks.
  print("Loading ranks. :3", flush=True)
  ranks = []
  for rank_idx in range(rank_count):
    rank_path = os.path.join(args.input_folder, "mp_rank_" + zero_pad(rank_idx) + "_model_states.pt")
    ranks.append(torch.load(rank_path, map_location="cpu"))
  
  # Average layernorms.
  print("Averaging layernorms. :3", flush=True)
  for tensor_name in ranks[0]["module"]:
    if not ("norm.scale" in tensor_name):
      # Not an rmsnorm scale vector, skipping.
      continue
    
    # Calc mean.
    total = ranks[0]["module"][tensor_name].to(torch.float32)
    for rank_idx in range(1, rank_count):
      total+= ranks[rank_idx]["module"][tensor_name].to(torch.float32)
      avg = (total / rank_count).to(torch.bfloat16)
    
    # Overwrite previous values.
    for rank_idx in range(0, rank_count):
      ranks[rank_idx]["module"][tensor_name] = avg
  
  # Save.
  print("Saving new ranks. :3", flush=True)
  if args.output_folder is not None:
    output_folder = args.output_folder
    os.makedirs(args.output_folder, exist_ok=True)
  else:
    print("Overwriting old ranks. Because you didn't provide the --output-folder ^_^''", flush=True)
    output_folder = args.input_folder

  for rank_idx in range(rank_count):
    output_rank_path = os.path.join(output_folder, "mp_rank_" + zero_pad(rank_idx) + "_model_states.pt")
    torch.save(ranks[rank_idx], output_rank_path)
  
  if args.output_folder is not None:
    print("Oh also copying optimizer states. :3", flush=True)
    pattern = re.compile(r"^mp_rank_(\d+)_model_states\.pt$")

    for file_name in os.listdir(args.input_folder):
      # Build full paths
      source_path = os.path.join(args.input_folder, file_name)
      destination_path = os.path.join(output_folder, file_name)

      # Check if file matches our pattern
      if pattern.match(file_name) is None:
        # The file does not match, so we copy it
        if os.path.isdir(source_path):
          shutil.copytree(source_path, destination_path, dirs_exist_ok=True)
        else:
          shutil.copy2(source_path, destination_path)


  print("Done! :d", flush=True)

if __name__ == "__main__":
  main()
