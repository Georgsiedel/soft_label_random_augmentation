
import numpy as np
import matplotlib.pyplot as plt

# Given data for Method A and Method B
data_A = [
    [0.5, 0.2, None, None, None, None, None, None],
    [0.5, 0.4, "74,55", "75,40", "75,29", "74,95", "75,12", 75.06],
    [0.5, 0.6, "74,39", "74,54", "75,10", "74,90", "75,05", 74.80],
    [0.5, 0.8, "74,71", "75,32", "74,18", "74,52", "74,26", 74.60],
    [0.5, 1.0, "74,60", "74,98", "74,51", "74,79", "75,00", 74.78],
    [0.75, 0.2, None, None, None, None, None, None],
    [0.75, 0.4, "74,41", "75,01", "74,59", "74,84", "74,91", 74.75],
    [0.75, 0.6, "73,71", "73,49", "73,85", "73,84", "73,60", 73.70],
    [0.75, 0.8, "73,65", "72,86", "73,30", "73,69", "73,45", 73.39],
    [0.75, 1.0, "74,02", "73,55", "73,02", "73,07", "73,91", 73.51],
    [1.0, 0.2, None, None, None, None, None, None],
    [1.0, 0.4, "74,09", "74,07", "73,97", "74,03", "73,90", 74.01],
    [1.0, 0.6, "72,39", "73,42", "72,48", "73,04", "72,72", 72.81],
    [1.0, 0.8, "72,05", "71,77", "71,83", "72,51", "72,49", 72.13],
    [1.0, 1.0, "72,25", "71,98", "71,98", "72,28", "72,96", 72.29]
]

data_B = [
    [0.5, 0.2, None, None, None, None, None, None],
    [0.5, 0.4, "75,44", "75,78", "75,71", "75,88", "75,95", 75.75],
    [0.5, 0.6, "75,90", "76,05", "75,51", "76,35", "75,60", 75.88],
    [0.5, 0.8, "76,58", "75,63", "75,78", "75,51", "75,18", 75.74],
    [0.5, 1.0, "75,57", "75,21", "76,15", "75,10", "75,96", 75.60],
    [0.75, 0.2, None, None, None, None, None, None],
    [0.75, 0.4, "76,33", "75,96", "75,80", "75,68", "75,69", 75.89],
    [0.75, 0.6, "75,60", "75,34", "75,35", "75,60", "75,53", 75.48],
    [0.75, 0.8, "76,04", "75,61", "74,44", "75,94", "74,91", 75.39],
    [0.75, 1.0, "76,12", "75,07", "75,55", "75,53", "75,35", 75.52],
    [1.0, 0.2, None, None, None, None, None, None],
    [1.0, 0.4, "75,97", "76,01", "75,31", "75,77", "75,82", 75.78],
    [1.0, 0.6, "74,67", "74,97", "75,42", "74,61", "74,46", 74.83],
    [1.0, 0.8, "75,16", "74,77", "74,53", "74,96", "75,10", 74.90],
    [1.0, 1.0, "74,53", "74,50", "74,12", "74,42", "74,20", 74.35]
]

param2_levels = [0.5, 0.75, 1.0]
param1_vals = [0.2, 0.4, 0.6, 0.8, 1.0]

means_A = {0.5: [], 0.75: [], 1.0: []}
stds_A = {0.5: [], 0.75: [], 1.0: []}
means_B = {0.5: [], 0.75: [], 1.0: []}
stds_B = {0.5: [], 0.75: [], 1.0: []}

def to_float(val):
    return float(val.replace(",", "."))

# Compute means and stds for Method A
for row in data_A:
    re_p, re_max = row[0], row[1]
    acc_vals = row[2:7]
    if acc_vals[0] is None:
        means_A[re_p].append(np.nan)
        stds_A[re_p].append(np.nan)
    else:
        arr = np.array([to_float(v) for v in acc_vals])
        means_A[re_p].append(arr.mean())
        stds_A[re_p].append(arr.std(ddof=0))

# Compute means and stds for Method B
for row in data_B:
    re_p, re_max = row[0], row[1]
    acc_vals = row[2:7]
    if acc_vals[0] is None:
        means_B[re_p].append(np.nan)
        stds_B[re_p].append(np.nan)
    else:
        arr = np.array([to_float(v) for v in acc_vals])
        means_B[re_p].append(arr.mean())
        stds_B[re_p].append(arr.std(ddof=0))

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(8, 2.5), sharey=True)

for i, re_p in enumerate(param2_levels):
    ax = axes[i]
    x = np.array(param1_vals)
    yA = np.array(means_A[re_p]); eA = np.array(stds_A[re_p])
    yB = np.array(means_B[re_p]); eB = np.array(stds_B[re_p])

    validA = ~np.isnan(yA)
    validB = ~np.isnan(yB)

    ax.errorbar(x[validA], yA[validA], yerr=eA[validA],
                marker='o', linestyle='-', label='RE', capsize=3)
    ax.errorbar(x[validB], yB[validB], yerr=eB[validB],
                marker='s', linestyle='--', label='Soft RE', capsize=3)

    ax.set_title(f'Probability = {re_p}', fontsize='medium', fontweight='bold')
    if i == 0:
        ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_xticks(param1_vals)
    ax.grid(axis='y', alpha=0.3)  # horizontal gridlines only

# Remove individual x-labels
for ax in axes:
    ax.set_xlabel('')
    ax.legend(#handles, labels,
           loc='lower left',
           frameon=False)

# Shared X-axis label
fig.supxlabel('Maximum Area Ratio', fontsize='medium', fontweight='bold')

# Shared legend above figure in two rows
handles, labels = axes[1].get_legend_handles_labels()

fig.subplots_adjust(bottom=0.18)

#fig.tight_layout()
#plt.show()
file_name = "visualization/final_plots/RE-sweep.pdf"
plt.savefig(file_name, format='pdf')