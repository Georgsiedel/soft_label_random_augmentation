import matplotlib.pyplot as plt

# Extended data with robustness values; ignore the repeated accuracy field
data_text = """
1	Rotate	0	no	65,73					65,73	39,19
1	ShearX	0	no	67,81					67,81	39,69
1	Solarize	0	no	70,53					70,53	44,80
1	TranslateX	0	no	70,92					70,92	36,92
1	Contrast	0	no	73,37					73,37	49,84
1	ShearY	0	no	73,08					73,08	42,60
1	Sharpness	0	no	73,55					73,55	50,66
1	Brightness	0	no	73,53					73,53	49,86
1	Color	0	no	73,46					73,46	44,30
1	Posterize	0	no	72,27					72,27	54,77
1	TranslateY	0	no	76,21					76,21	40,25

2	Rotate	0	no	70,23					70,23	44,38
2	ShearX	0	no	71,25					71,25	43,74
2	Solarize	0	no	73,05					73,05	47,26
2	TranslateX	0	no	72,76					72,76	39,14
2	Contrast	0	no	75,26					75,26	48,16
2	ShearY	0	no	75,44					75,44	45,80
2	Sharpness	0	no	74,60					74,60	46,78
2	Brightness	0	no	75,33					75,33	46,50
2	Color	0	no	74,56					74,56	45,20
2	Posterize	0	no	73,29					73,29	53,75
2	TranslateY	0	no	77,32					77,32	41,98

2	Rotate	0	no	70,23					70,23	44,38
2	Rotate,ShearX	0	no	72,62					72,62	45,76
2	Rotate,ShearX,Solarize	0	no	76,08					76,08	51,15
2	Rotate,ShearX,Solarize,TranslateX	0	no	77,42					77,42	50,88
2	Rotate,ShearX,Solarize,TranslateX,Contrast	0	no	77,99					77,99	53,75
2	Rotate,ShearX,Solarize,TranslateX,Contrast,ShearY	0	no	78,62					78,62	53,62
2	Rotate,ShearX,Solarize,TranslateX,Contrast,ShearY,Sharpness	0	no	78,99					78,99	56,36
2	Rotate,ShearX,Solarize,TranslateX,Contrast,ShearY,Sharpness,Brightness	0	no	78,80					78,80	55,50
2	Rotate,ShearX,Solarize,TranslateX,Contrast,ShearY,Sharpness,Brightness,Color	0	no	79,36					79,36	56,63
2	Rotate,ShearX,Solarize,TranslateX,Contrast,ShearY,Sharpness,Brightness,Color,Posterize	0	no	78,58					78,58	59,67
2	Rotate,ShearX,Solarize,TranslateX,Contrast,ShearY,Sharpness,Brightness,Color,Posterize,TranslateY	0	no	79,69					79,69	59,20
2	Rotate,ShearX,Solarize,TranslateX,Contrast,ShearY,Sharpness,Brightness,Color,Posterize,TranslateY,rest	0	no	79,27					79,27	60,87

1	Rotate	0		65,73					65,73	39,19
1	Rotate,ShearX	0		69,01					69,01	40,06
1	Rotate,ShearX,Solarize	0		74,38					74,38	47,70
1	Rotate,ShearX,Solarize,TranslateX	0		76,48					76,48	49,82
1	Rotate,ShearX,Solarize,TranslateX,Contrast	0		77,19					77,19	54,11
1	Rotate,ShearX,Solarize,TranslateX,Contrast,ShearY	0		78,20					78,20	54,97
1	Rotate,ShearX,Solarize,TranslateX,Contrast,ShearY,Sharpness	0		78,88					78,88	57,73
1	Rotate,ShearX,Solarize,TranslateX,Contrast,ShearY,Sharpness,Brightness	0		79,00					79,00	59,07
1	Rotate,ShearX,Solarize,TranslateX,Contrast,ShearY,Sharpness,Brightness,Color	0		79,84					79,84	59,41
1	Rotate,ShearX,Solarize,TranslateX,Contrast,ShearY,Sharpness,Brightness,Color,Posterize	0		79,01					79,01	63,38
1	Rotate,ShearX,Solarize,TranslateX,Contrast,ShearY,Sharpness,Brightness,Color,Posterize,TranslateY	0		80,04					80,04	63,54
1	Rotate,ShearX,Solarize,TranslateX,Contrast,ShearY,Sharpness,Brightness,Color,Posterize,TranslateY,rest	0	no	80,61					80,61	63,99
"""

# 1) Split raw text into blocks
lines = [l for l in data_text.strip().split('\n')]
blocks = []
cur = []
for l in lines:
    if l.strip() == "":
        if cur:
            blocks.append(cur)
            cur = []
    else:
        cur.append(l)
if cur:
    blocks.append(cur)

# 2) Parse each block to extract name, accuracy, and robustness
def parse_block_with_rob(block):
    out = []
    for line in block:
        parts = line.split('\t')
        name = parts[1]
        acc  = float(parts[4].replace(',', '.'))
        rob  = float(parts[-1].replace(',', '.'))
        out.append({'name': name, 'accuracy': acc, 'robustness': rob})
    return out

block1 = parse_block_with_rob(blocks[0])  # standard individual
block2 = parse_block_with_rob(blocks[1])  # soft individual
block3 = parse_block_with_rob(blocks[2])  # soft cumulative
block4 = parse_block_with_rob(blocks[3])  # standard cumulative

# 3) Prepare individual-transform data (Rotate is first)
indiv_names   = [e['name'] for e in block1]
std_acc_ind   = [e['accuracy'] for e in block1]
std_rob_ind   = [e['robustness'] for e in block1]
soft_acc_ind  = [e['accuracy'] for e in block2]
soft_rob_ind  = [e['robustness'] for e in block2]

# 4) Prepare cumulative data with "+" prefix for the newly added transform
def incr_label(n):
    if ',' in n:
        return '+ ' + n.split(',')[-1]
    else:
        return '+ ' + n

cum_names_std = [incr_label(e['name']) for e in block4]
cum_acc_std   = [e['accuracy']  for e in block4]
cum_rob_std   = [e['robustness'] for e in block4]

cum_names_soft = [incr_label(e['name']) for e in block3]
cum_acc_soft   = [e['accuracy']  for e in block3]
cum_rob_soft   = [e['robustness'] for e in block3]

# 5) Create a 2×2 grid: share x‐axis per column, share y‐axis per row, figure size 9×6
fig, axes = plt.subplots(2, 2, figsize=(9, 5),
                         sharex='col', sharey='row')

# Top-left: Individual Accuracy (markers only, no connecting lines)
ax1 = axes[0, 0]
ax1.plot(indiv_names, std_acc_ind, marker='o', linestyle='', label='Standard')
ax1.plot(indiv_names, soft_acc_ind, marker='s', linestyle='', label='Soft')
ax1.set_xticks(range(len(indiv_names)))
ax1.set_yticks(range(50, 101, 5))
ax1.set_xticklabels(indiv_names, rotation=45, ha='right')
ax1.set_ylabel('Accuracy (%)', fontweight='bold')
ax1.legend(frameon=False)
ax1.grid(True, axis='y', linewidth=0.2)  # only horizontal gridlines

# Top-right: Cumulative Accuracy (lines + markers)
ax2 = axes[0, 1]
ax2.plot(cum_names_std, cum_acc_std, marker='o', label='Standard')
ax2.plot(cum_names_soft, cum_acc_soft, marker='s', label='Soft')
ax2.set_xticks(range(len(cum_names_std)))
ax2.set_xticklabels(cum_names_std, rotation=45, ha='right')
ax2.legend(loc='lower right', frameon=False)
ax2.grid(True, axis='y', linewidth=0.2)

# Bottom-left: Individual Robustness (markers only)
ax3 = axes[1, 0]
ax3.plot(indiv_names, std_rob_ind, marker='o', linestyle='', label='Standard')
ax3.plot(indiv_names, soft_rob_ind, marker='s', linestyle='', label='Soft')
ax3.set_xticks(range(len(indiv_names)))
ax3.set_yticks(range(40, 101, 10))
ax3.set_xticklabels(indiv_names, rotation=37.5, fontsize=8.5, ha='right')
ax3.set_xlabel('Individual Transformations', fontweight='bold')
ax3.set_ylabel('Robustness (%)', fontweight='bold')
ax3.legend(frameon=False)
ax3.grid(True, axis='y', linewidth=0.2)

# Bottom-right: Cumulative Robustness (lines + markers)
ax4 = axes[1, 1]
ax4.plot(cum_names_std, cum_rob_std, marker='o', label='Standard')
ax4.plot(cum_names_soft, cum_rob_soft, marker='s', label='Soft')
ax4.set_xticks(range(len(cum_names_std)))
ax4.set_xticklabels(cum_names_std, rotation=37.5, fontsize=8.5, ha='right')
ax4.set_xlabel('Cumulative Transformations', fontweight='bold')
ax4.legend(loc='lower right', frameon=False)
ax4.grid(True, axis='y', linewidth=0.2)

plt.tight_layout()
#plt.show()
file_name = f"visualization/final_plots/TA-individual_plot.pdf"
plt.savefig(file_name, format='pdf')