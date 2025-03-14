import os
import yaml
import json
import subprocess


from collections import defaultdict
import logging

# set logging level
log_filename = "datapack.log"

# Configure logging to log both to file and console
logging.basicConfig(
    level=logging.INFO,  # Set logging level
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
    handlers=[
        logging.FileHandler(log_filename, mode='w'),  # Log to a file
        logging.StreamHandler()  # Log to console
    ]
)

# loads state file
def load_state(path_to_state):
    with open(path_to_state, "r", encoding="utf-8") as file:
        state_data = yaml.safe_load(file)

    return state_data

def plot_bar(categories, values, phase_name, total_tokens):
    values = [v / 10 ** 9 for v in values]
    logging.info(f"{phase_name}: {sum(values):.1f} B tokens [{100 * sum(values) / total_tokens:.2f} %]")

    # # Create the bar chart
    # plt.figure(figsize=(24, 12))  # Set figure size
    # plt.bar(categories, values, color='skyblue', edgecolor='black')
    #
    # # Labels and title
    # plt.xlabel("Language")
    # plt.ylabel("Tokens [B]")
    # plt.title(f"{phase_name}: {sum(values):.1f} B tokens [{100 * sum(values) / total_tokens:.2f} %]")
    #
    # # Show the plot
    # plt.show()


def sanity_check(slices, tokens):
    assert list(sorted(slices.keys())) == list(sorted(tokens.keys()))

    out = defaultdict(list)

    for key in slices:
        lang_slices = slices[key]
        lang_tokens = tokens[key]

        temp_sum = defaultdict(int)
        for slice, token in zip(lang_slices, lang_tokens):
            temp_sum[slice] += token

        for dset in temp_sum:
            out[dset].append(temp_sum[dset])

    total_sum = 0
    for dset in out:
        if dset == "E":
            continue
        total_sum += sum(out[dset])

    total_sum = total_sum / 10 ** 9

    for dset in out:
        categories = list(slices.keys())
        values = out[dset]
        plot_bar(categories, values, dset, total_sum)


def ascii_bar_chart(categories, values, scale=100):
    max_value = max(values)  # Get the maximum value to scale bars
    for category, value in zip(categories, values):
        bar = "#" * max(int((value * scale)), 1)  # Scale the bars for better visibility
        logging.info(f"{category}: {bar} ({value * 100} %)")


def main(args):
    logging.info(f"Setting: {args}")

    tokens_per_iter = args.tokens_per_iter
    warmup_iters = args.warmup_iters
    cd_phase = args.cd_phase
    max_tokens_per_pack = args.max_tokens_per_pack

    out_dir = args.out_dir
    path_to_first_state = args.state_file
    data_distribution_json = args.slices_json

    logging.info(f"Tokens per iter: {tokens_per_iter}")
    logging.info(f"Warmup iters: {warmup_iters}")
    logging.info(f"CD phase: {cd_phase}")
    logging.info(f"Max tokens per pack: {max_tokens_per_pack}")
    logging.info(f"Output dir: {out_dir}")
    logging.info(f"Data distribution json: {data_distribution_json}")
    logging.info(f"State: {path_to_first_state}")

    assert tokens_per_iter * warmup_iters <= max_tokens_per_pack
    os.makedirs(out_dir, exist_ok=True)

    # read in the input file

    # token_distribution = {"names": ["en", "lv", "lt"],
    #                        "U1": [69, 420, 1337].
    #                         "U2": [ 4188, ...],
    #                         ....
    # }

    logging.info("Reading data distribution from {}".format(data_distribution_json))
    with open(data_distribution_json, "r", encoding="utf-8") as f:
        token_distribution = json.load(f)

    # For each language extract number of tokens per phase
    langs = token_distribution["names"]
    U1 = {}
    N = {}  # U2 + N
    U3 = {}
    Ext = {}

    for n, lang in enumerate(langs):
        U1[lang] = token_distribution["U1"]["total"][n]
        N[lang] = (token_distribution["N"]["total"][n] + token_distribution["U2"]["total"][n])
        U3[lang] = token_distribution["U3"]["total"][n]
        Ext[lang] = token_distribution["Ext"]["total"][n]

    # for each language calculate relative % of tokens w.r.t to other languages per phase
    U1_ratio = {}
    N_ratio = {}
    U3_ratio = {}
    Ext_ratio = {}

    U1_sum = sum([U1[lang] for lang in U1])
    N_sum = sum([N[lang] for lang in N])
    U3_sum = sum([U3[lang] for lang in U3])
    Ext_sum = sum([Ext[lang] for lang in Ext])

    train_sum = U1_sum + N_sum + U3_sum

    logging.info(f"U1 total tokens: {U1_sum} [{100 * U1_sum / train_sum:.2f} %]")
    logging.info(f"N total tokens: {N_sum} [{100 * N_sum / train_sum:.2f} %]")
    logging.info(f"U3 total tokens: {U3_sum} [{100 * U3_sum / train_sum:.2f} %]")

    logging.info("Total tokens up to EXTENSION PHASE: {}".format(train_sum))
    logging.info(f"Total iterations up to EXTENSION PHASE: {train_sum / tokens_per_iter:.1f}")

    cool_down_phase_tokens = int(train_sum * cd_phase / 100) + 1
    logging.info(f"Cool down phase tokens: {cool_down_phase_tokens} [{100 * cool_down_phase_tokens / train_sum:.2f} %]")

    for lang in langs:
        U1_ratio[lang] = U1[lang] / U1_sum
        N_ratio[lang] = N[lang] / N_sum
        U3_ratio[lang] = U3[lang] / U3_sum
        Ext_ratio[lang] = Ext[lang] / Ext_sum

    logging.info("")  # Empty line
    logging.info("U1_ratio:")
    ascii_bar_chart(list(U1_ratio.keys()), U1_ratio.values())

    logging.info("")  # Empty line
    logging.info("N_ratio:")
    ascii_bar_chart(list(N_ratio.keys()), N_ratio.values())

    logging.info("")  # Empty line
    logging.info("U3_ratio:")
    ascii_bar_chart(list(U3_ratio.keys()), U3_ratio.values())

    logging.info("")  # Empty line
    logging.info("Ext_ratio:")
    ascii_bar_chart(list(Ext_ratio.keys()), Ext_ratio.values())

    logging.info("")  # Empty line

    # ------- prepare total per language slices ---------
    # {'en': [{'warmup': 69}, {'U1': 420}, {'N': 1337}, {'N': 1488} .... ]
    all_slices_per_language = defaultdict(list)

    # ------------- first slice the warmup from U1 ---------------
    # NOTE: warmup is always one datapack

    # calculate tokens per language for warmup
    logging.info("Slicing warmup ...")
    total_warmup_tokens = warmup_iters * tokens_per_iter
    logging.info("total_warmup_tokens: {}".format(total_warmup_tokens))

    # some sanity checks
    assert U1_sum > total_warmup_tokens

    for lang in U1_ratio:
        all_slices_per_language[lang].append({"warmup": total_warmup_tokens * U1_ratio[lang]})

    # ---- SLICE REMAINING U1 -------

    logging.info("Slicing U1 ...")
    # determine remaining number of tokens
    remaining_U1_tokens = U1_sum - total_warmup_tokens
    logging.info(f"remaining U1 tokens: {remaining_U1_tokens} [{100 * remaining_U1_tokens / train_sum:.2f} %]")
    # determine how many slices
    slices = int(remaining_U1_tokens // max_tokens_per_pack) + 1  # int() needed cause 0 is returned as float

    logging.info("U1 slices: {}".format(slices))
    # handle last slice
    last_slice_tokens = remaining_U1_tokens % max_tokens_per_pack

    # final sanity check
    total_sliced_tokens = 0

    for n in range(slices):

        # determine tokens to be sliced
        temp_target_slice_tokens = max_tokens_per_pack
        if n == len(range(slices)) - 1:
            temp_target_slice_tokens = last_slice_tokens

        for lang in U1_ratio:
            all_slices_per_language[lang].append({"U1": temp_target_slice_tokens * U1_ratio[lang]})

        logging.info(
            f"U1 slice {n}: {temp_target_slice_tokens} tokens, ~ {temp_target_slice_tokens / tokens_per_iter:.1f} iterations")
        total_sliced_tokens += temp_target_slice_tokens

    assert total_sliced_tokens == remaining_U1_tokens

    # ---- SLICE N ----------
    logging.info("Slicing N ...")
    # determine remaining number of tokens
    remaining_N_tokens = N_sum
    logging.info(f"remaining U1 tokens: {remaining_N_tokens} [{100 * remaining_N_tokens / train_sum:.2f} %]")
    # determine how many slices
    slices = int(remaining_N_tokens // max_tokens_per_pack) + 1  # int() needed cause 0 is returned as float
    logging.info("N slices: {}".format(slices))
    # handle last slice
    last_slice_tokens = remaining_N_tokens % max_tokens_per_pack

    # final sanity check
    total_sliced_tokens = 0

    for n in range(slices):

        # determine tokens to be sliced
        temp_target_slice_tokens = max_tokens_per_pack
        if n == len(range(slices)) - 1:
            temp_target_slice_tokens = last_slice_tokens

        for lang in N_ratio:
            all_slices_per_language[lang].append({"N": temp_target_slice_tokens * N_ratio[lang]})

        logging.info(
            f"N slice {n}: {temp_target_slice_tokens} tokens, ~ {temp_target_slice_tokens / tokens_per_iter:.1f} iterations")
        total_sliced_tokens += temp_target_slice_tokens

    assert total_sliced_tokens == remaining_N_tokens

    # ----- SLICE U3 before cooldown------------
    logging.info("Slicing U3 (excluding cooldown)...")
    # determine remaining number of tokens
    remaining_U3_tokens = U3_sum - cool_down_phase_tokens
    assert remaining_U3_tokens > 0
    logging.info(f"remaining U3 tokens: {remaining_U3_tokens} [{100 * remaining_U3_tokens / train_sum:.2f} %]")
    # determine how many slices
    slices = int(remaining_U3_tokens // max_tokens_per_pack) + 1
    logging.info("U3 slices: {}".format(slices))
    # handle last slice
    last_slice_tokens = remaining_U3_tokens % max_tokens_per_pack

    # final sanity check
    total_sliced_tokens = 0

    for n in range(slices):

        # determine tokens to be sliced
        temp_target_slice_tokens = max_tokens_per_pack
        if n == len(range(slices)) - 1:
            temp_target_slice_tokens = last_slice_tokens

        # calculate language distribution
        for lang in U3_ratio:
            all_slices_per_language[lang].append({"U3": temp_target_slice_tokens * U3_ratio[lang]})

        logging.info(
            f"U3 slice {n}: {temp_target_slice_tokens} tokens, ~ {temp_target_slice_tokens / tokens_per_iter:.1f} iterations")
        total_sliced_tokens += temp_target_slice_tokens

    assert total_sliced_tokens == remaining_U3_tokens

    # ----- SLICE U3 COOLDOWN ------------
    logging.info(f"Slicing U3 cooldown ({cd_phase} %) ...")
    # determine remaining number of tokens
    remaining_U3_cd_tokens = cool_down_phase_tokens
    logging.info(f"remaining U3 cd tokens: {remaining_U3_cd_tokens} [{100 * remaining_U3_cd_tokens / train_sum:.2f} %]")
    # determine how many slices
    slices = int(remaining_U3_cd_tokens // max_tokens_per_pack) + 1
    logging.info("U3 cooldown slices: {}".format(slices))
    # handle last slice
    last_slice_tokens = remaining_U3_cd_tokens % max_tokens_per_pack

    # final sanity check
    total_sliced_tokens = 0

    for n in range(slices):

        # determine tokens to be sliced
        temp_target_slice_tokens = max_tokens_per_pack
        if n == len(range(slices)) - 1:
            temp_target_slice_tokens = last_slice_tokens

        # calculate language distribution
        for lang in U3_ratio:
            all_slices_per_language[lang].append({"U3_cd": temp_target_slice_tokens * U3_ratio[lang]})

        logging.info(
            f"U3_cd slice {n}: {temp_target_slice_tokens} tokens, ~ {temp_target_slice_tokens / tokens_per_iter:.1f} iterations")
        total_sliced_tokens += temp_target_slice_tokens

    assert total_sliced_tokens == remaining_U3_cd_tokens

    # ----- SLICE EXT Phase ------------
    logging.info("Slicing Ext ...")
    # determine remaining number of tokens
    remaining_Ext_tokens = Ext_sum
    logging.info("remaining Ext phase tokens: {}".format(remaining_Ext_tokens))
    # determine how many slices
    slices = int(remaining_Ext_tokens // max_tokens_per_pack) + 1
    logging.info("E slices: {}".format(slices))
    # handle last slice
    last_slice_tokens = remaining_Ext_tokens % max_tokens_per_pack

    # final sanity check
    total_sliced_tokens = 0

    for n in range(slices):

        # determine tokens to be sliced
        temp_target_slice_tokens = max_tokens_per_pack
        if n == len(range(slices)) - 1:
            temp_target_slice_tokens = last_slice_tokens

        # calculate language distribution
        for lang in U3_ratio:
            all_slices_per_language[lang].append({"E": temp_target_slice_tokens * Ext_ratio[lang]})

        logging.info(
            f"E slice {n}: {temp_target_slice_tokens} tokens, ~ {temp_target_slice_tokens / tokens_per_iter:.1f} iterations")
        total_sliced_tokens += temp_target_slice_tokens

    assert total_sliced_tokens == remaining_Ext_tokens

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

    logging.info("----- finished cutting -----")
    logging.info("Performing sanity checks")

    sanity_check(slices, token_counts)

    n = 0
    processes = []

    state = load_state(path_to_first_state)

    for key in token_counts:

        n += 1
        if n == 3:
            break

        neoxpath = "/project/project_465001281/IP/llm-gpt-neox"

        local_token_yaml = {}
        local_token_yaml[key] = token_counts[key]

        local_slice_yaml = {}
        local_slice_yaml[key] = slices[key]

        local_state_yaml = {}
        local_state_yaml[key] = state[key]

        local_out_dir = out_dir + "/" + key
        os.makedirs(local_out_dir, exist_ok=True)

        local_token_file = local_out_dir + f"/tokens.yaml"
        local_slice_file = local_out_dir + f"/slices.yaml"
        local_state_file = local_out_dir + f"/state.0.yaml"

        # dump the token yaml
        with open(local_token_file, "w", encoding="utf-8") as f:
            yaml.dump(local_token_yaml, f, sort_keys=False)

        # dump the slice yaml
        with open(local_slice_file, "w", encoding="utf-8") as f:
            yaml.dump(local_slice_yaml, f, sort_keys=False)

        # dump the state yaml
        with open(local_state_file, "w", encoding="utf-8") as f:
            yaml.dump(local_state_yaml, f, sort_keys=False)

        print("Slicing ", key)

        cmd = ["srun", "--account=project_465001281", "--partition=small-g", "--gpus-per-node=1",
               "--ntasks-per-node=1", "--cpus-per-task=7", "--mem-per-gpu=60G", "--time=4:00:00", "--nodes=1"]

        cmd += ["singularity", "exec", "-B", "/scratch:/scratch", "-B", "/project:/project",
                "/scratch/project_465001281/containers/rocm603_flash.sif"]

        cmd += ["bash", "-c",
                "$WITH_CONDA ; python " + neoxpath + f"/slicer_multi.py --tokens {local_token_file} --state {local_state_file} --slice {local_slice_file} --out_dir {out_dir}"]

        processes.append((key, subprocess.Popen(cmd)))


    for p in processes:
        exit_code = p[1].wait()
        if exit_code != 0:
            print("Slicing", p[0], "process crashed :(")
        else:
            print("Slicing", p[0], "sucessful :)")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--tokens-per-iter", type=int, required=True)
    parser.add_argument("--warmup-iters", type=int, required=True)
    parser.add_argument("--cd_phase", type=int, required=True)
    parser.add_argument("--max_tokens_per_pack", type=int, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--slices_json", type=str, required=True)
    parser.add_argument('--state_file', type=str, required=True)

    args = parser.parse_args()

    main(args)
