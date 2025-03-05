import numpy as np
import json
import matplotlib.pyplot as plt

# ----- Constants --------
# total uniform tokens per language
total_uniform_tokens_plpp = 30  # B
uniform_fractions = [1 / 4, 1 / 4, 2 / 4]  # phase1, phase2 (part of natural phase), phase3
sampling_factor = 3

# TODO: add me? does nothing currently
max_tokens = 2500  # B

# downscale w.r.t to max total tokens
scaler = 1 #75/2000

# tokens per word, per language
tokens_per_word = {
    'bg': 1.7977874967841523,  # bul
    'bs': 1.9001949528863857,  # bos
    'cnr': 1.75,  # not found in your list
    'cs': 1.9953637456356248,  # ces
    'da': 2.280225644153072,  # dan
    'de': 1.7478527299211413,  # deu
    'el': 6.24297991635645,  # ell
    'en': 1.5008266947241846,  # eng
    'es': 1.9850564580447467,  # spa
    'et': 2.309734513274336,  # est
    'fi': 2.438334148557054,  # fin
    'fr': 1.5220373039793824,  # fra
    'ga': 2.5559077079107504,  # gle
    'hr': 1.9466277961540617,  # hrv
    'hu': 3.345203702611874,  # hun
    'is': 2.979283764252449,  # isl
    'it': 2.1007367655084592,  # ita
    'lt': 2.23324717896993,  # lit
    'ltg': 1.75,  # not found
    'lv': 2.128191233140655,  # lav
    'mk': 1.7801523595275832,  # mkd
    'mt': 3.4809173468214736,  # mlt
    'nl': 2.0061224489795917,  # nld
    'no': 2.2127638089467836,  # nob
    'pl': 1.9630981801842282,  # pol
    'pt': 2.099803939874895,  # por
    'ro': 1.602320335159523,  # ron
    'ru': 1.9572221916753725,  # rus
    'sk': 1.9571787008642318,  # slk
    'sl': 1.9113744334625675,  # slv
    'sq': 1.75,  # not found
    'sr': 1.8692709443615434,  # srp
    'sv': 2.296843340087448,  # swe
    'tr': 2.5659872611464967,  # tur
    'uk': 2.016381193240602,  # ukr
    'P': 1.75,
    'C': 1.75,
    'M': 1.75
}

# flat new data in tokens
flat_new_data = {"en": 300, "fr": 120, "de": 80, "es": 130, "ru": 80}
flat_new_data = {"en": 300, "fr": 120, "de": 90, "es": 50, "ru": 20, "pt": 35, "it": 35, "cs": 10, "pl": 40}
add_flat_data =  True

# clip some languages (in tokens)
clip_data = {"pl": 0, "tr": -71}
clip = True

# real data
data = {
    'bg': 34.54, 'bs': 15.64, 'cnr': 0.33, 'cs': 100.31, 'da': 7.79, 'de': 74.76,
    'el': 0.05, 'en': 47.73, 'es': 27.62, 'et': 18.57, 'fi': 29.02, 'fr': 54.87,
    'ga': 0.14, 'hr': 27.04, 'hu': 15.31, 'is': 0.03, 'it': 28.93, 'lt': 20.25,
    'ltg': 0.01, 'lv': 14.93, 'mk': 4.92, 'mt': 0.08, 'nl': 25.69, 'no': 5.89,
    'pl': 142.85, 'pt': 29.86, 'ro': 42.74, 'ru': 61.2, 'sk': 25.67, 'sl': 21.66,
    'sq': 5.87, 'sr': 8.62, 'sv': 14.98, 'tr': 57.87, 'uk': 67.02, 'P': 30, 'C': 30, 'M': 30
}

# magically adjusted data

data = {
    "bg": 13.98,
    "bs": 7.29,
    "cnr": 0.18,
    "cs": 29.81,
    "da": 4.27,
    "de": 45.97,
    "el": 0.03,
    "en": 30.50,
    "es": 17.61,
    "et": 3.64,
    "fi": 10.63,
    "fr": 34.45,
    "ga": 0.09,
    "hr": 11.50,
    "hu": 8.61,
    "is": 0.02,
    "it": 17.83,
    "lt": 7.62,
    "ltg": 0.003,
    "lv": 4.79,
    "mk": 2.51,
    "mt": 0.05,
    "nl": 14.48,
    "no": 3.34,
    "pl": 62.00,
    "pt": 18.90,
    "ro": 22.47,
    "ru": 38.24,
    "sk": 11.50,
    "sl": 6.98,
    "sq": 3.45,
    "sr": 5.03,
    "sv": 8.48,
    "tr": 34.33,
    "uk": 34.80,
    'P': 30,
    'C': 30,
    'M': 30
}

# sanity
assert sum(uniform_fractions) <= 1

# convert to tokens
for key in data.keys():
    data[key] = tokens_per_word[key] * data[key]

    if add_flat_data and key in flat_new_data.keys():
        data[key] = data[key] + flat_new_data[key]

    if clip and key in clip_data.keys():
        data[key] = data[key] + clip_data[key]

# up-sample to total U per language per phase, but at most N times
up_sampled = {}

for key in data.keys():  # this is way more pythonic/js and way less C/C++, cringe af, not necessary either
    up_sampled[key] = max(0, min(total_uniform_tokens_plpp, sampling_factor * data[key]) - data[key])

print(up_sampled)

# calculate total_data
total_data = {}

for key in data.keys():

    total_data[key] = data[key] + up_sampled[key]

    # if add_flat_data and key in flat_new_data.keys():
    #     total_data[key] = total_data[key] + flat_new_data[key]
    #
    # if clip and key in clip_data.keys():
    #     total_data[key] = total_data[key] + clip_data[key]

total_data = dict(sorted(total_data.items(), key=lambda item: item[1], reverse=True))

categories = []
values1 = []
values2a = []
values2b = []
values3 = []

up_sampled_values = []
novel_values = []
total_values = []

for lang in total_data.keys():
    categories.append(lang)
    up_sampled_values.append(up_sampled[lang])
    novel_values.append(data[lang])
    total_values.append(total_data[lang])

    # split into uniform phases
    tokens_for_uniform = min(total_data[lang], total_uniform_tokens_plpp)

    values1.append(tokens_for_uniform * uniform_fractions[0])
    values2a.append(tokens_for_uniform * uniform_fractions[1])
    values2b.append(0)
    values3.append(tokens_for_uniform * uniform_fractions[2])

    if tokens_for_uniform < total_data[lang]:
        values2b[-1] += total_data[lang] - tokens_for_uniform

# categories = ['A', 'B', 'C', 'D', 'E']
# values1 = [5, 7, 3, 8, 4]  # Dataset 1
# values2a = [4, 5, 2, 3, 3]  # First part of Dataset 2
# values2b = [2, 4, 2, 2, 4]  # Second part of Dataset 2
# values3 = [8, 6, 5, 7, 9] # Dataset 3


# Sample data

values1 = np.array(values1) * scaler  # Dataset 1
values2a = np.array(values2a) * scaler  # First part of Dataset 2
values2b = np.array(values2b) * scaler # Second part of Dataset 2
values3 = np.array(values3)  * scaler # Dataset 3

up_sampled_values = np.array(up_sampled_values) * scaler
novel_values = np.array(novel_values) * scaler
total_values = np.array(total_values) * scaler

# Composite Dataset 2
values2 = values2a + values2b

# Define fixed horizontal lines
line1_y = np.max(values1)
line2_y = np.max(values2)

# Compute offsets for stacked bars, considering alignment with the horizontal lines
y1_offset = np.zeros_like(values1)  # First dataset starts from 0
y2a_offset = np.maximum(line1_y, values1)  # First part of Dataset 2 starts at line1_y or above Dataset 1
y2b_offset = y2a_offset + values2a  # Second part of Dataset 2 is stacked on top of the first part
y3_offset = np.maximum(line2_y, y2b_offset)  # Dataset 3 starts at line2_y or above Dataset 2

x = np.arange(len(categories))  # Base positions

# Create the plot
fig, ax = plt.subplots(figsize=(10, 8))

# Plot Dataset 1
bars1 = ax.bar(categories, values1, color='skyblue', label='U1', bottom=y1_offset)

# Plot Dataset 2 (as two stacked components)
bars2a = ax.bar(categories, values2a, color='salmon', label='U2', bottom=y2a_offset)
bars2b = ax.bar(categories, values2b, color='orange', label='Natural', bottom=y2b_offset)

# Plot Dataset 3
bars3 = ax.bar(categories, values3, color='lightgreen', label='U3', bottom=y3_offset)

# Add labels for each bar
for bars in [bars1, bars2a, bars2b, bars3]:
    for bar in bars:
        height = bar.get_height()
        bottom = bar.get_y()
        ax.text(
            bar.get_x() + bar.get_width() / 2,  # X position (center of bar)
            bottom + height / 2,  # Y position (middle of bar)
            f'{int(height)}',  # Label text
            ha='center', va='center', color='black', fontsize=10
        )

# Draw horizontal lines at y=8 and y=16
for y_line in [line1_y, line2_y]:
    ax.axhline(y=y_line, color='black', linestyle='--', linewidth=2)

# Compute sums
sum1 = np.sum(values1)
sum2a = np.sum(values2a)
sum2b = np.sum(values2b)
sum3 = np.sum(values3)
total_sum = sum1 + sum2a + sum2b + sum3  # Compute the total

# Formatting
ax.set_ylabel('Total tokens')
ax.set_title(
    f"U1: {sum1:.0f}B [{100 * sum1 / total_sum:.0f}%]       U2+N: {sum2a:.0f}B + {sum2b:.0f}B [{100 * sum2a / total_sum:.0f}%,{100 * sum2b / total_sum:.0f}%]     U3: {sum3:.0f}B [{100 * sum3 / total_sum:.0f}%]       ∑: {total_sum:.0f}B")
ax.legend(loc='best')

# Show plot
plt.tight_layout()
plt.show()

# ------ plot up-sampled vs natural ------

x_pct = (novel_values / total_values) * 100
y_pct = (up_sampled_values / total_values) * 100
print(total_data.keys())
# Create the plot
fig, ax = plt.subplots(figsize=(10, 8))

# Plot stacked bar chart
bars_x = ax.bar(categories, x_pct, label="novel", color="lightgreen")
bars_y = ax.bar(categories, y_pct, bottom=x_pct, label="up-sampled", color="salmon")

# # Add z values on top of each bar
# for i, value in enumerate(z):
#     ax.text(i, 105, str(value), ha="center", fontsize=12, fontweight="bold")


# Add labels for each bar
for bars in [bars_x, bars_y]:
    for bar in bars:
        height = bar.get_height()
        bottom = bar.get_y()
        ax.text(
            bar.get_x() + bar.get_width() / 2,  # X position (center of bar)
            bottom + height / 2,  # Y position (middle of bar)
            f'{int(height)}',  # Label text
            ha='center', va='center', color='black', fontsize=10
        )

# Labels and legend
ax.set_ylabel("Percentage (%)")
ax.set_title(
    "Total upsampled tokens: {:.0f}B       Total novel tokens: {:.0f}B       Novel tokens: {:.2f}% of all tokens".format(
        np.sum(up_sampled_values), np.sum(novel_values),
        100 * np.sum(novel_values) / np.sum(up_sampled_values + novel_values)
    )
)

ax.legend()

# Show plot
plt.tight_layout()
plt.ylim(0, 110)  # Ensure space for text on top
plt.show()

# --- plot individual language distribution ---

fig, ax = plt.subplots(figsize=(10, 8))
total_w_extra = values1 + values2a + values2b + values3
bars_x = ax.bar(categories, 100 * total_w_extra / np.sum(total_w_extra), label="novel", color="skyblue")

print(100 * total_w_extra / np.sum(total_w_extra))
res = 100 * total_w_extra / np.sum(total_w_extra)

# Add labels for each bar
for bars in [bars_x]:
    for bar in bars:
        height = bar.get_height()
        bottom = bar.get_y()
        ax.text(
            bar.get_x() + bar.get_width() / 2,  # X position (center of bar)
            bottom + height / 2,  # Y position (middle of bar)
            f'{int(height)}',  # Label text
            ha='center', va='center', color='black', fontsize=10
        )

ax.set_ylabel("Percentage (%)")

ax.set_title(
    "Percentage (%) of total tokens per language"
)

# Show plot
plt.tight_layout()
plt.show()


# ------ write the json

def kek(inp):
    inp = str(inp)
    out = inp
    # out = inp.encode("utf-8").hex()
    return out

with open("1B.json", "w", encoding="utf-8") as json_file:

    big = {}

    for n,lang in enumerate(total_data.keys()):
        temp_dict = {}
        temp_dict["total_tokens"] = kek(total_values[n])
        temp_dict["novel_tokens"] = kek(novel_values[n])
        temp_dict["upsampled_tokens"] = kek(up_sampled_values[n])
        temp_dict["uspampling_%_of_total_tokens"] = kek(100*up_sampled_values[n] / total_values[n])
        temp_dict["tokens_per_word"] = kek(tokens_per_word[lang])
        temp_dict["%_of_ALL_tokens"] = res[n]
        temp_dict["U1_tokens"] = values1[n]
        temp_dict["U2_tokens"] = values2a[n]
        temp_dict["U3_tokens"] = values3[n]
        temp_dict["N_tokens"] = values2b[n]

        big[lang] = temp_dict


    json.dump(big, json_file, indent=4, ensure_ascii=False)