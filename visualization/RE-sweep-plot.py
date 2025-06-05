import numpy as np
import matplotlib.pyplot as plt

# now including RE_p = 0.25. Each row is [RE_p, RE_max, ACC-0, ACC-1, ACC-2, ACC-3, ACC-4, avg].
data_A = [
    # RE_p = 0.25
    [0.25, 0.2,  "75,90", "74,74", "75,73", "75,16", "75,35", 75.38],
    [0.25, 0.4,  "75,80", "75,21", "75,41", "74,85", "75,26", 75.31],
    [0.25, 0.6,  "76,06", "75,67", "75,57", "75,26", "75,53", 75.62],
    [0.25, 0.8,  "75,81", "75,90", "75,32", "75,89", "75,55", 75.69],
    [0.25, 1.0,  "75,83", "75,57", "75,54", "75,18", "75,31", 75.49],
    # RE_p = 0.5
    [0.5,  0.2,  "75,22", "75,74", "75,47", "75,51", "74,97", 75.38],
    [0.5,  0.4,  "74,55", "75,40", "75,29", "74,95", "75,12", 75.06],
    [0.5,  0.6,  "74,39", "74,54", "75,10", "74,90", "75,05", 74.80],
    [0.5,  0.8,  "74,71", "75,32", "74,18", "74,52", "74,26", 74.60],
    [0.5,  1.0,  "74,60", "74,98", "74,51", "74,79", "75,00", 74.78],
    # RE_p = 0.75
    [0.75, 0.2,  "76,02", "75,92", "75,16", "74,92", "75,36", 75.48],
    [0.75, 0.4,  "74,41", "75,01", "74,59", "74,84", "74,91", 74.75],
    [0.75, 0.6,  "73,71", "73,49", "73,85", "73,84", "73,60", 73.70],
    [0.75, 0.8,  "73,65", "72,86", "73,30", "73,69", "73,45", 73.39],
    [0.75, 1.0,  "74,02", "73,55", "73,02", "73,07", "73,91", 73.51],
    # RE_p = 1.0
    [1.0,  0.2,  "75,26", "75,49", "75,00", "75,74", "75,63", 75.42],
    [1.0,  0.4,  "74,09", "74,07", "73,97", "74,03", "73,90", 74.01],
    [1.0,  0.6,  "72,39", "73,42", "72,48", "73,04", "72,72", 72.81],
    [1.0,  0.8,  "72,05", "71,77", "71,83", "72,51", "72,49", 72.13],
    [1.0,  1.0,  "72,25", "71,98", "71,98", "72,28", "72,96", 72.29]
]

data_B = [
    # RE_p = 0.25
    [0.25, 0.2,  "75,37", "75,22", "75,49", "75,56", "75,33", 75.39],
    [0.25, 0.4,  "76,39", "75,59", "75,51", "76,05", "75,67", 75.84],
    [0.25, 0.6,  "75,95", "75,87", "75,51", "76,01", "75,30", 75.73],
    [0.25, 0.8,  "75,92", "75,43", "75,98", "75,67", "75,67", 75.73],
    [0.25, 1.0,  "76,24", "75,69", "75,49", "75,51", "75,93", 75.77],
    # RE_p = 0.5
    [0.5,  0.2,  "75,84", "75,97", "75,83", "75,21", "76,02", 75.77],
    [0.5,  0.4,  "75,44", "75,78", "75,71", "75,88", "75,95", 75.75],
    [0.5,  0.6,  "75,90", "76,05", "75,51", "76,35", "75,60", 75.88],
    [0.5,  0.8,  "76,58", "75,63", "75,78", "75,51", "75,18", 75.74],
    [0.5,  1.0,  "75,57", "75,21", "76,15", "75,10", "75,96", 75.60],
    # RE_p = 0.75
    [0.75, 0.2,  "76,03", "76,68", "75,97", "76,13", "76,24", 76.21],
    [0.75, 0.4,  "76,33", "75,96", "75,80", "75,68", "75,69", 75.89],
    [0.75, 0.6,  "75,60", "75,34", "75,35", "75,60", "75,53", 75.48],
    [0.75, 0.8,  "76,04", "75,61", "74,44", "75,94", "74,91", 75.39],
    [0.75, 1.0,  "76,12", "75,07", "75,55", "75,53", "75,35", 75.52],
    # RE_p = 1.0
    [1.0,  0.2,  "76,24", "76,34", "75,71", "76,00", "75,96", 76.05],
    [1.0,  0.4,  "75,97", "76,01", "75,31", "75,77", "75,82", 75.78],
    [1.0,  0.6,  "74,67", "74,97", "75,42", "74,61", "74,46", 74.83],
    [1.0,  0.8,  "75,16", "74,77", "74,53", "74,96", "75,10", 74.90],
    [1.0,  1.0,  "74,53", "74,50", "74,12", "74,42", "74,20", 74.35]
]

param2_levels = [0.25, 0.5, 0.75, 1.0]
param1_vals   = [0.2, 0.4, 0.6, 0.8, 1.0]

# Initialize dictionaries for means and stds
means_A = {p: [] for p in param2_levels}
stds_A  = {p: [] for p in param2_levels}
means_B = {p: [] for p in param2_levels}
stds_B  = {p: [] for p in param2_levels}

def to_float(s):
    return float(s.replace(",", "."))

# Process Method A: use the provided average (last column) for the mean,
# compute std from ACC-0..ACC-4
for row in data_A:
    re_p, re_max = row[0], row[1]
    acc_vals     = row[2:7]
    arr = np.array([to_float(val) for val in acc_vals])
    means_A[re_p].append(row[7])            # provided avg
    stds_A[re_p].append(arr.std(ddof=0))    # population standard deviation

# Process Method B in the same way
for row in data_B:
    re_p, re_max = row[0], row[1]
    acc_vals     = row[2:7]
    arr = np.array([to_float(val) for val in acc_vals])
    means_B[re_p].append(row[7])            # provided avg
    stds_B[re_p].append(arr.std(ddof=0))    # population standard deviation

# Create 1×4 subplots (shared y-axis)
fig, axes = plt.subplots(1, 4, figsize=(9, 2.6), sharey=True)

for i, re_p in enumerate(param2_levels):
    ax = axes[i]
    x         = np.array(param1_vals)
    yA = np.array(means_A[re_p]); eA = np.array(stds_A[re_p])
    yB = np.array(means_B[re_p]); eB = np.array(stds_B[re_p])

    # Plot Method A (“RE”) and Method B (“Soft RE”)
    ax.errorbar(
        x, yA, yerr=eA,
        marker='o', linestyle='-',
        label='RE', capsize=3
    )
    ax.errorbar(
        x, yB, yerr=eB,
        marker='s', linestyle='--',
        label='Soft RE', capsize=3
    )

    ax.set_title(f'Probability = {re_p}', fontsize='medium', fontweight='bold')
    if i == 0:
        ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_xticks(param1_vals)
    ax.grid(axis='y', alpha=0.3)  # horizontal gridlines only

    # Remove individual x-axis label (shared below)
    ax.set_xlabel('')
    # Per‐subplot legend at lower left (no handles/labels explicitly passed)
    ax.legend(loc='lower left', frameon=False)

# Shared X‐axis label
fig.supxlabel('Maximum Area Ratio', fontsize='medium', fontweight='bold')

# Adjust bottom margin so legends don’t overlap the figure
fig.subplots_adjust(bottom=0.185, left=0.065, right=0.98)
#fig.tight_layout()

# Save to PDF (as in your code)
file_name = "visualization/final_plots/RE-sweep.pdf"
#plt.show()
plt.savefig(file_name, format='pdf')

