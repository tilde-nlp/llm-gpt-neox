import os
import yaml
import json
import subprocess

from collections import defaultdict
import logging

# set logging level
logging.basicConfig(level=logging.INFO)

# warmup
# tokens per iteration
# max_datapack_size
# how many tokens per U1
# how many tokens per N
# how many tokens per U2
# how many tokens over a big job
# out_dir = "/scratch/project_465001281/MK/serious_test/"  # with ending slash, hardcoded

# TODO: add as args
out_dir = "/scratch/project_465001281/tokenized/1B_gucci_sliced/"

data_distribution_json = "/scratch/project_465001281/MK/full_pipe_test/1B_clean.json"

path_to_first_state = "/scratch/project_465001281/MK/full_pipe_test/state.0.yaml"

state_number = int(path_to_first_state.split("/")[-1].split(".")[1])

out_all = "sliced"
# out_warmup = "warmup_slices"
# out_U1 = "U1_slices"
# out_N = "N_slices"
# out_U3 = "U3_slices"

# os.makedirs(out_dir + out_warmup, exist_ok=True)
# os.makedirs(out_dir + out_U1, exist_ok=True)
# os.makedirs(out_dir + out_N, exist_ok=True)
# os.makedirs(out_dir + out_U3, exist_ok=True)
os.makedirs(out_dir + out_all, exist_ok=True)

tokens_per_iter = 400000 #260000
warmup_iters = 2000 # 500
max_tokens_per_pack = 8 * 10 ** 9

# read in the input file
# token_distribution = {
#     "en": {
#         "total_tokens": "2.453554787313994",
#         "novel_tokens": "2.453554787313994",
#         "upsampled_tokens": "0",
#         "uspampling_%_of_total_tokens": "0.0",
#         "tokens_per_word": "1.5008266947241846",
#         "%_of_ALL_tokens": 16.369090289446135,
#         "U1_tokens": 0.125,
#         "U2_tokens": 0.125,
#         "U3_tokens": 0.25,
#         "N_tokens": 1.953554787313994
#     },
#     "fr": {
#         "total_tokens": "1.3484820687546417",
#         "novel_tokens": "1.3484820687546417",
#         "upsampled_tokens": "0",
#         "uspampling_%_of_total_tokens": "0.0",
#         "tokens_per_word": "1.5220373039793824",
#         "%_of_ALL_tokens": 8.996507781800346,
#         "U1_tokens": 0.125,
#         "U2_tokens": 0.125,
#         "U3_tokens": 0.25,
#         "N_tokens": 0.8484820687546417
#     }
# }


with open(data_distribution_json, "r", encoding="utf-8") as f:
    token_distribution = json.load(f)

# For each language extract number of tokens per phase
U1 = {}
N = {}  # U2 + N
U3 = {}

for lang in token_distribution:
    U1[lang] = token_distribution[lang]["U1_tokens"] * 10 ** 9
    N[lang] = (token_distribution[lang]["N_tokens"] + token_distribution[lang]["U2_tokens"]) * 10 ** 9
    U3[lang] = token_distribution[lang]["U3_tokens"] * 10 ** 9

# for each language calculate relative % of tokens w.r.t to other languages per phase
U1_ratio = {}
N_ratio = {}
U3_ratio = {}

U1_sum = sum([U1[lang] for lang in U1])
N_sum = sum([N[lang] for lang in N])
U3_sum = sum([U3[lang] for lang in U3])

for lang in token_distribution:
    U1_ratio[lang] = U1[lang] / U1_sum
    N_ratio[lang] = N[lang] / N_sum
    U3_ratio[lang] = U3[lang] / U3_sum

logging.info("U1_ratio: {}".format(U1_ratio))
logging.info("N_ratio: {}".format(N_ratio))
logging.info("U3_ratio: {}".format(U3_ratio))

# ------- prepare total per language slices ---------
# {'en': [{'warmup': 69}, {'U1': 420}, {'N': 1337}, {'N': 1488} .... ]
all_slices_per_language = defaultdict(list)

# ------------- first slice the warmup from U1 ---------------
# NOTE: warmup is always one datapack

# calculate tokens per language for warmup
total_warmup_tokens = warmup_iters * tokens_per_iter
logging.info("total_warmup_tokens: {}".format(total_warmup_tokens))

# some sanity checks
assert U1_sum > total_warmup_tokens

for lang in U1_ratio:
    all_slices_per_language[lang].append({"warmup": total_warmup_tokens * U1_ratio[lang]})

# ---- SLICE REMAINING U1 -------

# determine remaining number of tokens
remaining_U1_tokens = U1_sum - total_warmup_tokens
logging.info("remaining U1 tokens: {}".format(remaining_U1_tokens))
# determine how many slices
slices = int(remaining_U1_tokens // max_tokens_per_pack) + 1  # int() needed cause 0 is returned as float

logging.info("U1 slices: {}".format(slices))
# handle last slice
last_slice_tokens = remaining_U1_tokens % max_tokens_per_pack

for n in range(slices):

    # determine tokens to be sliced
    temp_target_slice_tokens = max_tokens_per_pack
    if n == len(range(slices)) - 1:
        temp_target_slice_tokens = last_slice_tokens

    for lang in U1_ratio:
        all_slices_per_language[lang].append({"U1": temp_target_slice_tokens * U1_ratio[lang]})

# ---- SLICE N ----------
# determine remaining number of tokens
remaining_N_tokens = N_sum
logging.info("remaining N tokens: {}".format(remaining_N_tokens))
# determine how many slices
slices = int(remaining_N_tokens // max_tokens_per_pack) + 1
logging.info("N slices: {}".format(slices))
# handle last slice
last_slice_tokens = remaining_N_tokens % max_tokens_per_pack

for n in range(slices):

    # determine tokens to be sliced
    temp_target_slice_tokens = max_tokens_per_pack
    if n == len(range(slices)) - 1:
        temp_target_slice_tokens = last_slice_tokens

    for lang in N_ratio:
        all_slices_per_language[lang].append({"N": temp_target_slice_tokens * N_ratio[lang]})

# ----- SLICE U3 ------------
# determine remaining number of tokens
remaining_U3_tokens = U3_sum
logging.info("remaining U3 tokens: {}".format(remaining_U3_tokens))
# determine how many slices
slices = int(remaining_U3_tokens // max_tokens_per_pack) + 1
logging.info("U3 slices: {}".format(slices))
# handle last slice
last_slice_tokens = remaining_U3_tokens % max_tokens_per_pack

for n in range(slices):

    # determine tokens to be sliced
    temp_target_slice_tokens = max_tokens_per_pack
    if n == len(range(slices)) - 1:
        temp_target_slice_tokens = last_slice_tokens

    # calculate language distribution
    for lang in U3_ratio:
        all_slices_per_language[lang].append({"U3": temp_target_slice_tokens * U3_ratio[lang]})

# create token counts.yaml and slices.yaml

token_counts = defaultdict(list)
slices = defaultdict(list)

for lang in all_slices_per_language:
    tokens_to_slice = all_slices_per_language[lang]

    for slice in tokens_to_slice:
        assert (len(list(slice.keys())) == 1)
        key = list(slice.keys())[0]
        token_counts[lang].append(slice[key])
        slices[lang].append(key)

token_counts = dict(token_counts)
slices = dict(slices)

logging.info("Created token counts file:")
logging.info(token_counts)
logging.info("Created slice counts file:")
logging.info(slices)

# dump the token yaml
with open(out_dir + out_all + f"/tokens.yaml", "w", encoding="utf-8") as f:
    yaml.dump(token_counts, f, sort_keys=False)

# dump the state yaml
with open(out_dir + out_all + f"/slices.yaml", "w", encoding="utf-8") as f:
    yaml.dump(slices, f, sort_keys=False)

cmd = [
    "python",
    "slicer_multi.py",
    "--tokens", out_dir + out_all + f"/tokens.yaml",
    "--state", path_to_first_state,
    "--slice", out_dir + out_all + f"/slices.yaml",
    "--out_dir", out_dir + out_all
]
subprocess.run(cmd, check=True)
