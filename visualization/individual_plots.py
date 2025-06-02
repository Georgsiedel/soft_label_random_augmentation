import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from visualization_utils import compute_occlusion_visibility
from visualization_utils import plot_severity_vs_confidence
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter

def model_confidence(augmentation_type: str):
    data = pd.read_csv(f"visualization/non_linear_mapping_data/{augmentation_type}/{augmentation_type}_MAPPING_results.csv")

    data = data.sort_values(by="Severity")
    data.reset_index(drop=True, inplace=True)
    augmentation_magnitude = data["Severity"]
    augmentation_mean = data["Mean"]
    augmentation_std = data["Std"]
    model_accuracy = data["Accuracy"]
    return augmentation_magnitude, augmentation_mean, model_accuracy


def get_data(visibility_values: list, k: int = 2, chance: float = 0.1):
    confidence_rc_values = []

    confidence_rc_values = 1 - (1 - chance) * (visibility_values) ** k
    confidence_rc_values = np.clip(confidence_rc_values, chance, 1.0)

    return confidence_rc_values

def individual_plots(augmentation_type):
    
    """HVS Data"""
    rotation_hvs = [1., 0.9985, 0.997, 0.9955, 0.994, 0.9925, 0.991, 0.9895, 0.988, 0.9865, 0.985, 0.9835, 0.982, 0.9805, 0.979, 0.9775, 0.976, 0.9745, 0.973, 0.9715, 0.97, 0.964, 0.958, 0.952, 0.946, 0.94, 0.934, 0.9315, 0.936, 0.9405, 0.945]
    contrast_hvs = [0.32, 0.32, 0.64254054, 0.96603963, 0.96734732, 0.96865501, 0.9699627, 0.9712704, 0.97257809, 0.97388578, 0.97519347, 0.97650117, 0.97780886, 0.97911655, 0.98042424, 0.98173193, 0.98303963, 0.98434732, 0.98565501, 0.98696271, 0.9882704, 0.98957809, 0.99088578, 0.99219347, 0.99350117, 0.99480886, 0.99611655, 0.99742424, 0.99873194, 1., 1.]
    occlusion_hvs = [1., 0.9888205, 0.97764103, 0.96646153, 0.95528205, 0.94410256, 0.93292308, 0.92174358, 0.91056411, 0.89938461, 0.88820511, 0.87702564, 0.86584614, 0.85466667, 0.84348717, 0.83230768, 0.82112822, 0.80994873, 0.79876924, 0.78758975, 0.776, 0.764, 0.75466667, 0.72666669, 0.68000003, 0.68533333, 0.65333335, 0.58400002, 0.51066667, 0.38800001, 0.216]
    rotation_mirrored = rotation_hvs[::-1]
    rotation_hvs = rotation_mirrored + rotation_hvs
    occlusion_mirrored = occlusion_hvs[::-1]
    occlusion_hvs = occlusion_mirrored + occlusion_hvs
    """HVS Data"""

    num_bins = 31

    """Plotting Parameters"""
    plt.rcParams.update({'font.size': 12, 'font.family': 'DejaVu Sans'})
    main_data_color = '#377eb8'
    secondary_data_color = '#4daf4a'
    est_conf_color = 'red'
    metrics_color = '#333333'
    constk_color = '#984ea3'
    """Plotting Parameters"""

    sns.set_palette("colorblind")
    plt.figure(figsize=(12, 8))

    
    # """Occlusion"""
    if augmentation_type == 'occlusion':
        min_val, max_val = 0.0, 1.0

        visibility_values1 = [0.0, .05, .10, .15, .20, .25, .30, .35, 1.0]
        confidence_values1 = [0.22, 0.42, 0.44, 0.6, 0.56, 0.64, 0.62, 0.72, 1.0]
        confidence_values2 = [0.18, 0.42, 0.62, 0.64, 0.62, 0.73, 0.8, 0.72, 1.0]
        confidence_values3 = [0.22, 0.48, 0.62, 0.74, 0.72, 0.76, 0.78, 0.83, 1.0]
        confidence_values4 = [0.24, 0.47, 0.6, 0.74, 0.72, 0.8, 0.86, 0.87, 1.0]
        confidence_values5 = [0.22, 0.58, 0.64, 0.72, 0.78, 0.82, 0.76, 0.77, 1.0]
        confidence_values = np.mean([confidence_values1, confidence_values2, confidence_values3, confidence_values4, confidence_values5], axis=0)

        k1, k2, k3, k4 = 1, 2, 3, 4

        visibility_values_lim = np.linspace(min_val, max_val, num_bins)
        confidence_values_lim = np.interp(visibility_values_lim, visibility_values1, confidence_values)
        chance = 0.1

        estimated_confidence_values1 = 1 - (1 - chance) * (1 - visibility_values_lim) ** k1
        estimated_confidence_values2 = 1 - (1 - chance) * (1 - visibility_values_lim) ** k2
        estimated_confidence_values3 = 1 - (1 - chance) * (1 - visibility_values_lim) ** k3
        estimated_confidence_values4 = 1 - (1 - chance) * (1 - visibility_values_lim) ** k4
        
        plt.plot(visibility_values_lim, confidence_values_lim, '-', label=f'HVS Studies for Occlusion', color=main_data_color)
        plt.plot(visibility_values_lim, estimated_confidence_values1, '--', label=f'k={k1}', color='red')
        plt.plot(visibility_values_lim, estimated_confidence_values2, '--', label=f'k={k2}', color='green')
        plt.plot(visibility_values_lim, estimated_confidence_values3, '--', label=f'k={k3}', color='orange')
        plt.plot(visibility_values_lim, estimated_confidence_values4, '--', label=f'k={k4}', color='purple')
        plt.axhline(y=chance, color=main_data_color, linestyle='--', label=f'Min. Probability')

        plt.xlabel("Proportion Visible ($v$)", fontsize=12)
        plt.ylabel("Label Confidence ($p$)", fontsize=12)
        plt.yticks(np.arange(0.1, 1.2, 0.2))
        # #plt.title(f"HVS for {augmentation_type}")
        plt.legend(frameon=False, loc='upper left', fontsize=11)
        plt.tight_layout()
        filename = f"visualization/final_plots/Occlusion_plot.pdf"
        plt.savefig(filename, format='pdf')
        plt.show()

    
    """Rotate"""
    if augmentation_type == "Rotate":
        min_val, max_val = 0.0, 135.0

        # rotation_values1 = np.arange(0.0, 151.0, 30)
        # confidence_values1 = [1.0, 0.99, 0.98, 0.97, 0.93, 0.96]
        # confidence_values2 = [1.0, 0.96, 0.93, 0.91, 0.92, 0.86]
        # confidence_values3 = [1.0, 0.98, 0.97, 0.96, 0.96, 0.92]
        # confidence_values4 = [1.0, 1.0, 1.0, 0.98, 0.99, 0.98]
        # confidence_values = np.mean([
        #         confidence_values1,
        #         confidence_values2,
        #         confidence_values3,
        #         confidence_values4,
        #     ],
        #     axis=0,
        # )
        # rotation_values_another = np.arange(0.0, 181.0, 45)
        # confidence_values_another = [0.98, 0.99, 0.94, 0.94, 0.88]
        # rotation_values = np.concatenate((rotation_values1, rotation_values_another))
        # confidence_values = np.concatenate((confidence_values1, confidence_values_another))
        # unique_rot_vals, unique_indices = np.unique(rotation_values1, return_index=True)
        # rotation_values = unique_rot_vals.tolist()
        # confidence_values = confidence_values[unique_indices].tolist()

        # rotation_values_lim = np.linspace(min_val, max_val, num_bins)
        # confidence_values_lim = np.interp(rotation_values_lim, rotation_values, confidence_values)
        # confidence_values_lim[0] = 1.0

        augmentation_magnitude, augmentation_mean, model_accuracy = model_confidence(augmentation_type=augmentation_type)
        df_aug = pd.read_csv(f'visualization/non_linear_mapping_data/{augmentation_type}/{augmentation_type}_poly_k_results.csv')
        df_aug = df_aug.sort_values(by='severity')
        const_k = np.array(df_aug['mean_poly_k'])

        k1 = 2
        k2 = 1
        chance = 0.1
        chance_1 = 0.7
        chance_2 = min(rotation_hvs)

        estimated_confidence_values1 = 1 - (1 - chance_1) * (abs(augmentation_magnitude) / 135.0) ** k1
        estimated_confidence_values2 = 1 - (1 - chance_2) * (abs(augmentation_magnitude) / 135.0) ** k2

        ssim, ncc, uiq, scc, sift = plot_severity_vs_confidence(augmentation_type)
        
        #if you want to plot just one side, use the following lines
        #'''
        rotation_hvs = rotation_hvs[31:]
        const_k = const_k[31:]
        augmentation_magnitude = augmentation_magnitude[31:]
        model_accuracy = model_accuracy[31:]
        estimated_confidence_values1 = estimated_confidence_values1[31:]
        estimated_confidence_values2 = estimated_confidence_values2[31:]
        ssim = ssim[31:]
        ncc = ncc[31:]
        uiq = uiq[31:]
        scc = scc[31:]
        sift = sift[31:]
        #'''

        ssim = np.array(ssim)
        ncc = np.array(ncc)
        uiq = np.array(uiq)
        scc = np.array(scc)
        sift = np.array(sift)
        augmentation_magnitude = np.array(augmentation_magnitude)
        rotation_hvs = np.array(rotation_hvs)
        model_accuracy = np.array(model_accuracy)
        estimated_confidence_values1 = np.array(estimated_confidence_values1)
        estimated_confidence_values2 = np.array(estimated_confidence_values2)

        plt.plot(augmentation_magnitude, rotation_hvs, '-', label='Rotation HVS', color=main_data_color, linewidth=2)
        plt.plot(augmentation_magnitude, model_accuracy, "-", label="Model Accuracy", color=secondary_data_color, linewidth=2)
        plt.plot(augmentation_magnitude, estimated_confidence_values2, '--', label=f'k={k2}, min=HVS', color=est_conf_color, linewidth=2)
        plt.plot(augmentation_magnitude, estimated_confidence_values1, '-.', label=f'k={k1}, min={chance_1}', color=est_conf_color, linewidth=2)
        plt.plot(augmentation_magnitude, const_k, '-', label=f'k=2, min=chance', color=est_conf_color, linewidth=2)
        plt.plot(augmentation_magnitude, ssim, '-', label='SSIM', color=metrics_color, linewidth=1, alpha=0.8)
        plt.plot(augmentation_magnitude, ncc, '--', label='NCC', color=metrics_color, linewidth=1, alpha=0.8)
        plt.plot(augmentation_magnitude, uiq, '.-', label='UIQ', color=metrics_color, linewidth=1, alpha=0.8)
        plt.plot(augmentation_magnitude, scc, ':', label='SCC', color=metrics_color, linewidth=1, alpha=0.8)
        plt.plot(augmentation_magnitude, sift, '-.', label='SIFT', color=metrics_color, linewidth=1, alpha=0.8)
        plt.ylim(0.0, 1.2)
        ticks = np.linspace(0.0, 1.0, 6)
        plt.axhline(y=chance, color=est_conf_color, linestyle=':', label=f'Chance', linewidth=2, alpha=0.6)
        plt.axhline(y=chance_2, color=main_data_color, linestyle=':', label=f'HVS lower bound', linewidth=2, alpha=0.6)
        
        plt.yticks(list(ticks) + [chance, chance_2])
        plt.gca().get_yticklabels()[-2].set_color(est_conf_color)
        plt.gca().get_yticklabels()[-1].set_color(main_data_color)
        plt.xlabel(f"Augmentation Magnitude [$\circ$]", fontsize=16, fontweight='bold')
        plt.legend(fontsize=14, frameon=False, ncols=4, loc='upper center')

    """Contrast"""
    if augmentation_type == 'Contrast':
        augmentation_magnitude, _, model_accuracy = model_confidence(augmentation_type=augmentation_type)

        # contrast_values1 = [0.044, 0.061, 0.098, 0.956]
        # confidence_values1 = [0.0, 0.28, 0.96, 1.0]
        # confidence_values2 = [0.06, 0.3, 0.97, 1.0]
        # confidence_values3 = [0.2, 0.6, 0.94, 1.0]
        # confidence_values4 = [0.58, 0.9, 0.98, 1.0]
        # confidence_values5 = [0.76, 0.88, 0.98, 1.0]
        # confidence_values = np.mean([confidence_values1, confidence_values2,
        #                             confidence_values3, confidence_values4, confidence_values5], axis=0)
        # contrast_values_extended = augmentation_magnitude[31:].copy()
        # confidence_values_interpolated = np.interp(contrast_values_extended, contrast_values1, confidence_values)
        # contrast_values_mapped = augmentation_magnitude.copy()

        df_aug = pd.read_csv(f'visualization/non_linear_mapping_data/{augmentation_type}/{augmentation_type}_poly_k_results.csv')
        df_aug = df_aug.sort_values(by='severity')
        const_k = np.array(df_aug['mean_poly_k'])

        contrast_hvs = contrast_hvs + [1.0] * 31

        k1 = 2
        k2_neg, k2_pos = 5, 2
        chance = 0.1
        chance_1 = 0.7
        chance_2_neg = min(contrast_hvs)
        chance_2_pos = min(model_accuracy[31:])
        chance_2 = min(chance_2_neg, chance_2_pos)
        chance = 0.1

        estimated_confidence_values1 = 1 - (1 - chance_1) * (abs(augmentation_magnitude)) ** k1

        estimated_confidence_values2_neg = 1 - (1 - chance_2_neg) * (abs(augmentation_magnitude[:31])) ** k2_neg
        estimated_confidence_values2_pos = 1 - (1 - chance_2_pos) * (augmentation_magnitude[31:]) ** k2_pos
        estimated_confidence_values2 = np.hstack((np.array(estimated_confidence_values2_neg), np.array(estimated_confidence_values2_pos)))

        ssim, ncc, uiq, scc = plot_severity_vs_confidence(augmentation_type)

        contrast_hvs = np.array(contrast_hvs)
        model_accuracy = np.array(model_accuracy)
        ssim = np.array(ssim)
        ncc = np.array(ncc)
        uiq = np.array(uiq)
        scc = np.array(scc)
        augmentation_magnitude = np.array(augmentation_magnitude)
        estimated_confidence_values1 = np.array(estimated_confidence_values1)

        plt.plot(augmentation_magnitude, contrast_hvs, '-', label='Contrast HVS', color=main_data_color, linewidth=2)
        plt.plot(augmentation_magnitude, model_accuracy, "-", label="Model Accuracy", color=secondary_data_color, linewidth=2)
        plt.plot(augmentation_magnitude, estimated_confidence_values2, '--', label=f'k={k2_neg, k2_pos}, min=HVS', color=est_conf_color, linewidth=2)
        plt.plot(augmentation_magnitude, estimated_confidence_values1, '-.', label=f'k={k1}, min={chance_1}', color=est_conf_color, linewidth=2)
        plt.plot(augmentation_magnitude, const_k, '-', label=f'k=2, min=chance', color=est_conf_color, linewidth=2)
        plt.plot(augmentation_magnitude, ssim, '-', label='SSIM', color=metrics_color, linewidth=1, alpha=0.8)
        plt.plot(augmentation_magnitude, ncc, '--', label='NCC', color=metrics_color, linewidth=1, alpha=0.8)
        plt.plot(augmentation_magnitude, uiq, '.-', label='UIQ', color=metrics_color, linewidth=1, alpha=0.8)
        plt.plot(augmentation_magnitude, scc, ':', label='SCC', color=metrics_color, linewidth=1, alpha=0.8)
        plt.ylim(0.0, 1.0)
        ticks = np.linspace(0.0, 1.0, 6)
        plt.axhline(y=chance, color=est_conf_color, linestyle=':', label=f'Chance', linewidth=2, alpha=0.6)
        plt.axhline(y=chance_2, color=main_data_color, linestyle=':', label=f'HVS lower bound', linewidth=2, alpha=0.6)
        plt.yticks(list(ticks) + [chance, chance_2])
        plt.gca().get_yticklabels()[-2].set_color(est_conf_color)
        plt.gca().get_yticklabels()[-1].set_color(main_data_color)
        plt.xlabel(f"Augmentation Magnitude", fontsize=16, fontweight='bold')
        plt.legend(fontsize=14, frameon=False)


    """Brightness"""
    if augmentation_type == 'Brightness':
        augmentation_magnitude, _, model_accuracy = model_confidence(augmentation_type=augmentation_type)

        df_aug = pd.read_csv(f'visualization/non_linear_mapping_data/{augmentation_type}/{augmentation_type}_poly_k_results.csv')
        df_aug = df_aug.sort_values(by='severity')
        const_k = np.array(df_aug['mean_poly_k'])

        contrast_hvs = contrast_hvs + [1.0] * 31

        k1 = 2
        k2_neg, k2_pos = 5, 3
        chance = 0.1
        chance_1 = 0.7
        chance_2_neg = min(contrast_hvs)
        chance_2_pos = min(model_accuracy[31:])
        chance_2 = min(chance_2_neg, chance_2_pos)
        chance = 0.1

        estimated_confidence_values1 = 1 - (1 - chance_1) * (abs(augmentation_magnitude)) ** k1

        estimated_confidence_values2_neg = 1 - (1 - chance_2_neg) * (abs(augmentation_magnitude[:31])) ** k2_neg
        estimated_confidence_values2_pos = 1 - (1 - chance_2_pos) * (augmentation_magnitude[31:]) ** k2_pos
        estimated_confidence_values2 = np.hstack((np.array(estimated_confidence_values2_neg), np.array(estimated_confidence_values2_pos)))

        ssim, ncc, uiq, scc = plot_severity_vs_confidence(augmentation_type)

        contrast_hvs = np.array(contrast_hvs)
        model_accuracy = np.array(model_accuracy)
        ssim = np.array(ssim)
        ncc = np.array(ncc)
        uiq = np.array(uiq)
        scc = np.array(scc)
        augmentation_magnitude = np.array(augmentation_magnitude)
        estimated_confidence_values1 = np.array(estimated_confidence_values1)

        plt.plot(augmentation_magnitude, contrast_hvs, '-', label='Contrast HVS', color=main_data_color, linewidth=2)
        plt.plot(augmentation_magnitude, model_accuracy, "-", label="Model Accuracy", color=secondary_data_color, linewidth=2)
        plt.plot(augmentation_magnitude, estimated_confidence_values2, '--', label=f'k={k2_neg, k2_pos}, min=HVS, Model Accuracy', color=est_conf_color, linewidth=2)
        plt.plot(augmentation_magnitude, estimated_confidence_values1, '-.', label=f'k={k1}, min={chance_1}', color=est_conf_color, linewidth=2)
        plt.plot(augmentation_magnitude, const_k, '-', label=f'k=2, min=chance', color=est_conf_color, linewidth=2)
        plt.plot(augmentation_magnitude, ssim, '-', label='SSIM', color=metrics_color, linewidth=1, alpha=0.8)
        plt.plot(augmentation_magnitude, ncc, '--', label='NCC', color=metrics_color, linewidth=1, alpha=0.8)
        plt.plot(augmentation_magnitude, uiq, '.-', label='UIQ', color=metrics_color, linewidth=1, alpha=0.8)
        plt.plot(augmentation_magnitude, scc, ':', label='SCC', color=metrics_color, linewidth=1, alpha=0.8)
        plt.ylim(0.0, 1.0)
        ticks = np.linspace(0.0, 1.0, 6)
        plt.axhline(y=chance, color=est_conf_color, linestyle=':', label=f'Chance', linewidth=2, alpha=0.6)
        plt.axhline(y=chance_2, color=main_data_color, linestyle=':', label=f'HVS lower bound', linewidth=2, alpha=0.6)
        plt.yticks(list(ticks) + [chance, chance_2])
        plt.gca().get_yticklabels()[-2].set_color(est_conf_color)
        plt.gca().get_yticklabels()[-1].set_color(main_data_color)
        plt.xlabel(f"Augmentation Magnitude", fontsize=16, fontweight='bold')
        plt.legend(fontsize=14, frameon=False)

    """Color"""
    if augmentation_type == 'Color':

        augmentation_magnitude, augmentation_mean, model_accuracy = model_confidence(augmentation_type=augmentation_type)

        df_aug = pd.read_csv(f'visualization/non_linear_mapping_data/{augmentation_type}/{augmentation_type}_poly_k_results.csv')
        df_aug = df_aug.sort_values(by='severity')
        const_k = np.array(df_aug['mean_poly_k'])

        k1 = 2
        k2 = 3
        chance = 0.1
        chance_1 = 0.7
        chance_2 = min(model_accuracy)

        estimated_confidence_values1 = 1 - (1 - chance_1) * (abs(augmentation_magnitude)) ** k1
        estimated_confidence_values2 = np.where(
                    augmentation_magnitude < 0,
                    1 - (1 - chance_2) * (abs(augmentation_magnitude)) ** k2, 
                    1  # Always 1 for positive values
        )

        ssim, ncc, uiq, scc = plot_severity_vs_confidence(augmentation_type)

        ssim = np.array(ssim)
        ncc = np.array(ncc)
        uiq = np.array(uiq)
        scc = np.array(scc)
        augmentation_magnitude = np.array(augmentation_magnitude)
        model_accuracy = np.array(model_accuracy)
        estimated_confidence_values1 = np.array(estimated_confidence_values1)
        estimated_confidence_values2 = np.array(estimated_confidence_values2)

        plt.plot(augmentation_magnitude, model_accuracy, "-", label="Model Accuracy", color=secondary_data_color, linewidth=2)
        plt.plot(augmentation_magnitude, estimated_confidence_values2, '--', label=f'k={k2}, min=Model Accuracy', color=est_conf_color, linewidth=2)
        plt.plot(augmentation_magnitude, estimated_confidence_values1, '-.', label=f'k={k1}, min={chance_1}', color=est_conf_color, linewidth=2)
        plt.plot(augmentation_magnitude, const_k, '-', label=f'k=2, min=chance', color=est_conf_color, linewidth=2)
        plt.plot(augmentation_magnitude, ssim, '-', label='SSIM', color=metrics_color, linewidth=1, alpha=0.8)
        plt.plot(augmentation_magnitude, ncc, '--', label='NCC', color=metrics_color, linewidth=1, alpha=0.8)
        plt.plot(augmentation_magnitude, uiq, '.-', label='UIQ', color=metrics_color, linewidth=1, alpha=0.8)
        plt.plot(augmentation_magnitude, scc, ':', label='SCC', color=metrics_color, linewidth=1, alpha=0.8)
        plt.ylim(0.8, 1.0)
        ticks = np.linspace(0.8, 1.0, 3)
        #plt.axhline(y=chance, color=est_conf_color, linestyle=':', label=f'Chance', linewidth=2, alpha=0.6)
        plt.axhline(y=chance_2, color=secondary_data_color, linestyle=':', label=f'Model Accuracy lower bound', linewidth=2, alpha=0.6)
        plt.yticks(list(ticks) + [chance_2])
        #plt.gca().get_yticklabels()[-2].set_color(est_conf_color)
        plt.gca().get_yticklabels()[-1].set_color(secondary_data_color)
        plt.xlabel(f"Augmentation Magnitude", fontsize=16, fontweight='bold')
        plt.legend(fontsize=14, frameon=False)

    """Sharpness"""
    if augmentation_type == 'Sharpness':
        augmentation_magnitude, augmentation_mean, model_accuracy = model_confidence(augmentation_type=augmentation_type)

        df_aug = pd.read_csv(f'visualization/non_linear_mapping_data/{augmentation_type}/{augmentation_type}_poly_k_results.csv')
        df_aug = df_aug.sort_values(by='severity')
        const_k = np.array(df_aug['mean_poly_k'])
        
        
        k1 = 2
        k2_neg = 4
        k2_pos = 1
        chance = 0.1
        chance_1 = 0.7
        chance_2_neg = min(model_accuracy)
        chance_2_pos = min(model_accuracy[31:])
        chance_2 = min(chance_2_neg, chance_2_pos)
        estimated_confidence_values1 = 1 - (1 - chance_1) * (abs(augmentation_magnitude)) ** k1
        estimated_confidence_values2 = np.where(
                    augmentation_magnitude < 0,
                    1 - (1 - chance_2_neg) * (abs(augmentation_magnitude)) ** k2_neg, 
                    1 - (1 - chance_2_pos) * (abs(augmentation_magnitude)) ** k2_pos
        )

        ssim, ncc, uiq, scc = plot_severity_vs_confidence(augmentation_type)

        ssim = np.array(ssim)
        ncc = np.array(ncc)
        uiq = np.array(uiq)
        scc = np.array(scc)
        augmentation_magnitude = np.array(augmentation_magnitude)
        model_accuracy = np.array(model_accuracy)
        estimated_confidence_values1 = np.array(estimated_confidence_values1)
        estimated_confidence_values2 = np.array(estimated_confidence_values2)

        plt.plot(augmentation_magnitude, model_accuracy, "-", label="Model Accuracy", color=secondary_data_color, linewidth=2)
        plt.plot(augmentation_magnitude, estimated_confidence_values2, '--', label=f'k={k2_neg, k2_pos}, min=Model Accuracy', color=est_conf_color, linewidth=2)
        plt.plot(augmentation_magnitude, estimated_confidence_values1, '-.', label=f'k={k1}, min={chance_1}', color=est_conf_color, linewidth=2)
        plt.plot(augmentation_magnitude, const_k, '-', label=f'k=2, min=chance', color=est_conf_color, linewidth=2)
        plt.plot(augmentation_magnitude, ssim, '-', label='SSIM', color=metrics_color, linewidth=1, alpha=0.8)
        plt.plot(augmentation_magnitude, ncc, '--', label='NCC', color=metrics_color, linewidth=1, alpha=0.8)
        plt.plot(augmentation_magnitude, uiq, '.-', label='UIQ', color=metrics_color, linewidth=1, alpha=0.8)
        plt.plot(augmentation_magnitude, scc, ':', label='SCC', color=metrics_color, linewidth=1, alpha=0.8)
        plt.ylim(0.8, 1.0)
        ticks = np.linspace(0.8, 1.0, 3)
        #plt.axhline(y=chance, color=est_conf_color, linestyle=':', label=f'Chance', linewidth=2, alpha=0.6)
        plt.axhline(y=chance_2, color=secondary_data_color, linestyle=':', label=f'Model Accuracy lower bound', linewidth=2, alpha=0.6)
        plt.yticks(list(ticks) + [chance_2])
        #plt.gca().get_yticklabels()[-2].set_color(est_conf_color)
        plt.gca().get_yticklabels()[-1].set_color(secondary_data_color)
        plt.xlabel(f"Augmentation Magnitude", fontsize=16, fontweight='bold')
        plt.legend(fontsize=14, frameon=False)

    """ShearX"""
    if augmentation_type == "ShearX":

        augmentation_magnitude, augmentation_mean, model_accuracy = model_confidence(augmentation_type=augmentation_type)

        df_aug = pd.read_csv(f'visualization/non_linear_mapping_data/{augmentation_type}/{augmentation_type}_poly_k_results.csv')
        df_aug = df_aug.sort_values(by='severity')
        const_k = np.array(df_aug['mean_poly_k'])

        k1 = 2
        k2 = 1
        chance = 0.1
        chance_1 = 0.7
        chance_2 = min(rotation_hvs)

        estimated_confidence_values1 = 1 - (1 - chance_1) * (abs(augmentation_magnitude)) ** k1
        estimated_confidence_values2 = 1 - (1 - chance_2) * (abs(augmentation_magnitude)) ** k2

        ssim, ncc, uiq, scc, sift = plot_severity_vs_confidence(augmentation_type)
        
        #if you want to plot just one side, use the following lines
        #'''
        rotation_hvs = rotation_hvs[31:]
        const_k = const_k[31:]
        augmentation_magnitude = augmentation_magnitude[31:]
        model_accuracy = model_accuracy[31:]
        estimated_confidence_values1 = estimated_confidence_values1[31:]
        estimated_confidence_values2 = estimated_confidence_values2[31:]
        ssim = ssim[31:]
        ncc = ncc[31:]
        uiq = uiq[31:]
        scc = scc[31:]
        sift = sift[31:]
        #'''

        ssim = np.array(ssim)
        ncc = np.array(ncc)
        uiq = np.array(uiq)
        scc = np.array(scc)
        sift = np.array(sift)
        augmentation_magnitude = np.array(augmentation_magnitude)
        rotation_hvs = np.array(rotation_hvs)
        model_accuracy = np.array(model_accuracy)
        estimated_confidence_values1 = np.array(estimated_confidence_values1)
        estimated_confidence_values2 = np.array(estimated_confidence_values2)

        plt.plot(augmentation_magnitude, rotation_hvs, '-', label='Rotation HVS', color=main_data_color, linewidth=2)
        plt.plot(augmentation_magnitude, model_accuracy, "-", label="Model Accuracy", color=secondary_data_color, linewidth=2)
        plt.plot(augmentation_magnitude, estimated_confidence_values2, '--', label=f'k={k2}, min=HVS', color=est_conf_color, linewidth=2)
        plt.plot(augmentation_magnitude, estimated_confidence_values1, '-.', label=f'k={k1}, min={chance_1}', color=est_conf_color, linewidth=2)
        plt.plot(augmentation_magnitude, const_k, '-', label=f'k=2, min=chance', color=est_conf_color, linewidth=2)
        plt.plot(augmentation_magnitude, ssim, '-', label='SSIM', color=metrics_color, linewidth=1, alpha=0.8)
        plt.plot(augmentation_magnitude, ncc, '--', label='NCC', color=metrics_color, linewidth=1, alpha=0.8)
        plt.plot(augmentation_magnitude, uiq, '.-', label='UIQ', color=metrics_color, linewidth=1, alpha=0.8)
        plt.plot(augmentation_magnitude, scc, ':', label='SCC', color=metrics_color, linewidth=1, alpha=0.8)
        plt.plot(augmentation_magnitude, sift, '-.', label='SIFT', color=metrics_color, linewidth=1, alpha=0.8)
        plt.ylim(0.0, 1.2)
        ticks = np.linspace(0.0, 1.0, 6)
        plt.axhline(y=chance, color=est_conf_color, linestyle=':', label=f'Chance', linewidth=2, alpha=0.6)
        plt.axhline(y=chance_2, color=main_data_color, linestyle=':', label=f'HVS lower bound', linewidth=2, alpha=0.6)
        plt.yticks(list(ticks) + [chance, chance_2])
        plt.gca().get_yticklabels()[-2].set_color(est_conf_color)
        plt.gca().get_yticklabels()[-1].set_color(main_data_color)
        plt.xlabel(f"Augmentation Magnitude", fontsize=16, fontweight='bold')
        plt.legend(fontsize=14, frameon=False, ncols=4, loc='upper center')


    """ShearY"""
    if augmentation_type == "ShearY":

        augmentation_magnitude, augmentation_mean, model_accuracy = model_confidence(augmentation_type=augmentation_type)

        df_aug = pd.read_csv(f'visualization/non_linear_mapping_data/{augmentation_type}/{augmentation_type}_poly_k_results.csv')
        df_aug = df_aug.sort_values(by='severity')
        const_k = np.array(df_aug['mean_poly_k'])

        k1 = 2
        k2 = 1
        chance = 0.1
        chance_1 = 0.7
        chance_2 = min(rotation_hvs)

        estimated_confidence_values1 = 1 - (1 - chance_1) * (abs(augmentation_magnitude)) ** k1
        estimated_confidence_values2 = 1 - (1 - chance_2) * (abs(augmentation_magnitude)) ** k2

        ssim, ncc, uiq, scc, sift = plot_severity_vs_confidence(augmentation_type)
        
        #if you want to plot just one side, use the following lines
        #'''
        rotation_hvs = rotation_hvs[31:]
        const_k = const_k[31:]
        augmentation_magnitude = augmentation_magnitude[31:]
        model_accuracy = model_accuracy[31:]
        estimated_confidence_values1 = estimated_confidence_values1[31:]
        estimated_confidence_values2 = estimated_confidence_values2[31:]
        ssim = ssim[31:]
        ncc = ncc[31:]
        uiq = uiq[31:]
        scc = scc[31:]
        sift = sift[31:]
        #'''

        ssim = np.array(ssim)
        ncc = np.array(ncc)
        uiq = np.array(uiq)
        scc = np.array(scc)
        sift = np.array(sift)
        augmentation_magnitude = np.array(augmentation_magnitude)
        rotation_hvs = np.array(rotation_hvs)
        model_accuracy = np.array(model_accuracy)
        estimated_confidence_values1 = np.array(estimated_confidence_values1)
        estimated_confidence_values2 = np.array(estimated_confidence_values2)

        plt.plot(augmentation_magnitude, rotation_hvs, '-', label='Rotation HVS', color=main_data_color, linewidth=2)
        plt.plot(augmentation_magnitude, model_accuracy, "-", label="Model Accuracy", color=secondary_data_color, linewidth=2)
        plt.plot(augmentation_magnitude, estimated_confidence_values2, '--', label=f'k={k2}, min=HVS', color=est_conf_color, linewidth=2)
        plt.plot(augmentation_magnitude, estimated_confidence_values1, '-.', label=f'k={k1}, min={chance_1}', color=est_conf_color, linewidth=2)
        plt.plot(augmentation_magnitude, const_k, '-', label=f'k=2, min=chance', color=est_conf_color, linewidth=2)
        plt.plot(augmentation_magnitude, ssim, '-', label='SSIM', color=metrics_color, linewidth=1, alpha=0.8)
        plt.plot(augmentation_magnitude, ncc, '--', label='NCC', color=metrics_color, linewidth=1, alpha=0.8)
        plt.plot(augmentation_magnitude, uiq, '.-', label='UIQ', color=metrics_color, linewidth=1, alpha=0.8)
        plt.plot(augmentation_magnitude, scc, ':', label='SCC', color=metrics_color, linewidth=1, alpha=0.8)
        plt.plot(augmentation_magnitude, sift, '-.', label='SIFT', color=metrics_color, linewidth=1, alpha=0.8)
        plt.ylim(0.0, 1.2)
        ticks = np.linspace(0.0, 1.0, 6)
        plt.axhline(y=chance, color=est_conf_color, linestyle=':', label=f'Chance', linewidth=2, alpha=0.6)
        plt.axhline(y=chance_2, color=main_data_color, linestyle=':', label=f'HVS lower bound', linewidth=2, alpha=0.6)
        plt.yticks(list(ticks) + [chance, chance_2])
        plt.gca().get_yticklabels()[-2].set_color(est_conf_color)
        plt.gca().get_yticklabels()[-1].set_color(main_data_color)
        plt.xlabel(f"Augmentation Magnitude", fontsize=16, fontweight='bold')
        plt.legend(fontsize=14, frameon=False, ncols=4, loc='upper center')

    """TranslateX"""
    if augmentation_type == "TranslateX":

        augmentation_magnitude, augmentation_mean, model_accuracy = model_confidence(augmentation_type=augmentation_type)

        df_aug = pd.read_csv(f'visualization/non_linear_mapping_data/{augmentation_type}/{augmentation_type}_poly_k_results.csv')
        df_aug = df_aug.sort_values(by='severity')
        const_k = np.array(df_aug['mean_poly_k'])

        visibility = compute_occlusion_visibility(dim1=32, dim2=32, tx=0., ty=augmentation_magnitude)
        
        k1 = 2
        k2 = 4
        chance = 0.1
        chance_1 = 0.7
        chance_2 = min(occlusion_hvs)

        estimated_confidence_values1 = 1 - (1 - chance_1) * (1 - visibility) ** k1
        estimated_confidence_values2 = 1 - (1 - chance_2) * (1 - visibility) ** k2

        ssim, ncc, uiq, scc, sift = plot_severity_vs_confidence(augmentation_type)
        
        #if you want to plot just one side, use the following lines
        #'''
        occlusion_hvs = occlusion_hvs[31:]
        const_k = const_k[31:]
        augmentation_magnitude = augmentation_magnitude[31:]
        model_accuracy = model_accuracy[31:]
        estimated_confidence_values1 = estimated_confidence_values1[31:]
        estimated_confidence_values2 = estimated_confidence_values2[31:]
        ssim = ssim[31:]
        ncc = ncc[31:]
        uiq = uiq[31:]
        scc = scc[31:]
        sift = sift[31:]
        #'''

        ssim = np.array(ssim)
        ncc = np.array(ncc)
        uiq = np.array(uiq)
        scc = np.array(scc)
        sift = np.array(sift)
        augmentation_magnitude = np.array(augmentation_magnitude)
        occlusion_hvs = np.array(occlusion_hvs)
        model_accuracy = np.array(model_accuracy)
        estimated_confidence_values1 = np.array(estimated_confidence_values1)
        estimated_confidence_values2 = np.array(estimated_confidence_values2)

        plt.plot(augmentation_magnitude, occlusion_hvs, '-', label='Occlusion HVS', color=main_data_color, linewidth=2)
        plt.plot(augmentation_magnitude, model_accuracy, "-", label="Model Accuracy", color=secondary_data_color, linewidth=2)
        plt.plot(augmentation_magnitude, estimated_confidence_values2, '--', label=f'k={k2}, min=HVS', color=est_conf_color, linewidth=2)
        plt.plot(augmentation_magnitude, estimated_confidence_values1, '-.', label=f'k={k1}, min={chance_1}', color=est_conf_color, linewidth=2)
        plt.plot(augmentation_magnitude, const_k, '-', label=f'k=2, min=chance', color=est_conf_color, linewidth=2)
        plt.plot(augmentation_magnitude, ssim, '-', label='SSIM', color=metrics_color, linewidth=1, alpha=0.8)
        plt.plot(augmentation_magnitude, ncc, '--', label='NCC', color=metrics_color, linewidth=1, alpha=0.8)
        plt.plot(augmentation_magnitude, uiq, '.-', label='UIQ', color=metrics_color, linewidth=1, alpha=0.8)
        plt.plot(augmentation_magnitude, scc, ':', label='SCC', color=metrics_color, linewidth=1, alpha=0.8)
        plt.plot(augmentation_magnitude, sift, '-.', label='SIFT', color=metrics_color, linewidth=1, alpha=0.8)
        plt.ylim(0.0, 1.2)
        ticks = np.linspace(0.0, 1.0, 6)
        plt.axhline(y=chance, color=est_conf_color, linestyle=':', label=f'Chance', linewidth=2, alpha=0.6)
        plt.axhline(y=chance_2, color=main_data_color, linestyle=':', label=f'HVS lower bound', linewidth=2, alpha=0.6)
        plt.yticks(list(ticks) + [chance, chance_2])
        plt.gca().get_yticklabels()[-2].set_color(est_conf_color)
        plt.gca().get_yticklabels()[-1].set_color(main_data_color)
        plt.xlabel(f"Augmentation Magnitude [px]", fontsize=16, fontweight='bold')
        plt.legend(fontsize=14, frameon=False, ncols=4, loc='upper center')
        

    """TranslateY"""
    if augmentation_type == "TranslateY":

        augmentation_magnitude, augmentation_mean, model_accuracy = model_confidence(augmentation_type=augmentation_type)

        df_aug = pd.read_csv(f'visualization/non_linear_mapping_data/{augmentation_type}/{augmentation_type}_poly_k_results.csv')
        df_aug = df_aug.sort_values(by='severity')
        const_k = np.array(df_aug['mean_poly_k'])

        visibility = compute_occlusion_visibility(dim1=32, dim2=32, tx=0., ty=augmentation_magnitude)
        
        k1 = 2
        k2 = 4
        chance = 0.1
        chance_1 = 0.7
        chance_2 = min(occlusion_hvs)

        estimated_confidence_values1 = 1 - (1 - chance_1) * (1 - visibility) ** k1
        estimated_confidence_values2 = 1 - (1 - chance_2) * (1 - visibility) ** k2

        ssim, ncc, uiq, scc, sift = plot_severity_vs_confidence(augmentation_type)
        
        #if you want to plot just one side, use the following lines
        #'''
        occlusion_hvs = occlusion_hvs[31:]
        const_k = const_k[31:]
        augmentation_magnitude = augmentation_magnitude[31:]
        model_accuracy = model_accuracy[31:]
        estimated_confidence_values1 = estimated_confidence_values1[31:]
        estimated_confidence_values2 = estimated_confidence_values2[31:]
        ssim = ssim[31:]
        ncc = ncc[31:]
        uiq = uiq[31:]
        scc = scc[31:]
        sift = sift[31:]
        #'''

        ssim = np.array(ssim)
        ncc = np.array(ncc)
        uiq = np.array(uiq)
        scc = np.array(scc)
        sift = np.array(sift)
        augmentation_magnitude = np.array(augmentation_magnitude)
        occlusion_hvs = np.array(occlusion_hvs)
        model_accuracy = np.array(model_accuracy)
        estimated_confidence_values1 = np.array(estimated_confidence_values1)
        estimated_confidence_values2 = np.array(estimated_confidence_values2)

        plt.plot(augmentation_magnitude, occlusion_hvs, '-', label='Occlusion HVS', color=main_data_color, linewidth=2)
        plt.plot(augmentation_magnitude, model_accuracy, "-", label="Model Accuracy", color=secondary_data_color, linewidth=2)
        plt.plot(augmentation_magnitude, estimated_confidence_values2, '--', label=f'k={k2}, min=HVS', color=est_conf_color, linewidth=2)
        plt.plot(augmentation_magnitude, estimated_confidence_values1, '-.', label=f'k={k1}, min={chance_1}', color=est_conf_color, linewidth=2)
        plt.plot(augmentation_magnitude, const_k, '-', label=f'k=2, min=chance', color=est_conf_color, linewidth=2)
        plt.plot(augmentation_magnitude, ssim, '-', label='SSIM', color=metrics_color, linewidth=1, alpha=0.8)
        plt.plot(augmentation_magnitude, ncc, '--', label='NCC', color=metrics_color, linewidth=1, alpha=0.8)
        plt.plot(augmentation_magnitude, uiq, '.-', label='UIQ', color=metrics_color, linewidth=1, alpha=0.8)
        plt.plot(augmentation_magnitude, scc, ':', label='SCC', color=metrics_color, linewidth=1, alpha=0.8)
        plt.plot(augmentation_magnitude, sift, '-.', label='SIFT', color=metrics_color, linewidth=1, alpha=0.8)
        plt.ylim(0.0, 1.2)
        ticks = np.linspace(0.0, 1.0, 6)
        plt.axhline(y=chance, color=est_conf_color, linestyle=':', label=f'Chance', linewidth=2, alpha=0.6)
        plt.axhline(y=chance_2, color=main_data_color, linestyle=':', label=f'HVS lower bound', linewidth=2, alpha=0.6)
        plt.yticks(list(ticks) + [chance, chance_2])
        plt.gca().get_yticklabels()[-2].set_color(est_conf_color)
        plt.gca().get_yticklabels()[-1].set_color(main_data_color)
        plt.xlabel(f"Augmentation Magnitude [px]", fontsize=16, fontweight='bold')
        plt.legend(fontsize=14, frameon=False, ncols=4, loc='upper center')

    """Posterize"""
    if augmentation_type == 'Posterize':

        augmentation_magnitude, augmentation_mean, model_accuracy = model_confidence(augmentation_type=augmentation_type)

        df_aug = pd.read_csv(f'visualization/non_linear_mapping_data/{augmentation_type}/{augmentation_type}_poly_k_results.csv')
        df_aug = df_aug.sort_values(by='severity')
        const_k = np.array(df_aug['mean_poly_k'])

        unique_augmentation_magnitudes, unique_indices = np.unique(augmentation_magnitude, return_index=True)
        unique_model_accuracy = model_accuracy[unique_indices]
        unique_const_k = const_k[unique_indices]

        k1 = 2
        k2 = 10
        chance = 0.1
        chance_1 = 0.7
        chance_2 = min(model_accuracy)
        
        estimated_confidence_values1 = 1 - (1 - chance_1) * (1 - unique_augmentation_magnitudes / 8.0) ** k1
        estimated_confidence_values2 = 1 - (1 - chance_2) * (1 - (unique_augmentation_magnitudes - 2) / 6) ** k2

        ssim, ncc, uiq, scc = plot_severity_vs_confidence(augmentation_type)

        ssim = np.array(ssim)
        ncc = np.array(ncc)
        uiq = np.array(uiq)
        scc = np.array(scc)
        unique_augmentation_magnitudes = np.array(unique_augmentation_magnitudes)
        unique_model_accuracy = np.array(unique_model_accuracy)
        estimated_confidence_values1 = np.array(estimated_confidence_values1)
        estimated_confidence_values2 = np.array(estimated_confidence_values2)

        plt.plot(unique_augmentation_magnitudes, unique_model_accuracy, "-", label="Model Accuracy", color=secondary_data_color, linewidth=2)
        plt.plot(unique_augmentation_magnitudes, estimated_confidence_values2, '--', label=f'k={k2}, min=Model Accuracy', color=est_conf_color, linewidth=2)
        plt.plot(unique_augmentation_magnitudes, estimated_confidence_values1, '-.', label=f'k={k1}, min={chance_1}', color=est_conf_color, linewidth=2)
        plt.plot(unique_augmentation_magnitudes, unique_const_k, '-', label=f'k=2, min=chance', color=est_conf_color, linewidth=2)
        plt.plot(augmentation_magnitude, ssim, '-', label='SSIM', color=metrics_color, linewidth=1, alpha=0.8)
        plt.plot(augmentation_magnitude, ncc, '--', label='NCC', color=metrics_color, linewidth=1, alpha=0.8)
        plt.plot(augmentation_magnitude, uiq, '.-', label='UIQ', color=metrics_color, linewidth=1, alpha=0.8)
        plt.plot(augmentation_magnitude, scc, ':', label='SCC', color=metrics_color, linewidth=1, alpha=0.8)
        plt.ylim(0.5, 1.0)
        ticks = np.linspace(0.5, 1.0, 6)
        #plt.axhline(y=chance, color=est_conf_color, linestyle=':', label=f'Chance', linewidth=2, alpha=0.6)
        plt.axhline(y=chance_2, color=secondary_data_color, linestyle=':', label=f'Model Accuracy lower bound', linewidth=2, alpha=0.6)
        plt.yticks(list(ticks) + [chance_2])
        #plt.gca().get_yticklabels()[-2].set_color(est_conf_color)
        plt.gca().get_yticklabels()[-1].set_color(secondary_data_color)
        plt.xlabel(f"Augmentation Magnitude", fontsize=16, fontweight='bold')
        plt.legend(fontsize=14, frameon=False)


    if augmentation_type == 'Solarize':

        augmentation_magnitude, augmentation_mean, model_accuracy = model_confidence(augmentation_type=augmentation_type)

        df_aug = pd.read_csv(f'visualization/non_linear_mapping_data/{augmentation_type}/{augmentation_type}_poly_k_results.csv')
        df_aug = df_aug.sort_values(by='severity')
        const_k = np.array(df_aug['mean_poly_k']) 
        const_severity = np.array(df_aug['severity']) / 255.0

        k1 = 2
        k2 = 1.5
        chance = 0.1
        chance_1 = 0.7
        chance_2 = min(model_accuracy)
        
        estimated_confidence_values1 = 1 - (1 - chance_1) * (1 - augmentation_magnitude) ** k1
        estimated_confidence_values2 = 1 - (1 - chance_2) * (1 - augmentation_magnitude) ** k2

        ssim, ncc, uiq, scc = plot_severity_vs_confidence(augmentation_type)

        ssim = np.array(ssim)
        ncc = np.array(ncc)
        uiq = np.array(uiq)
        scc = np.array(scc)
        augmentation_magnitude = np.array(augmentation_magnitude)
        model_accuracy = np.array(model_accuracy)
        estimated_confidence_values1 = np.array(estimated_confidence_values1)
        estimated_confidence_values2 = np.array(estimated_confidence_values2)

        plt.plot(augmentation_magnitude, model_accuracy, "-", label="Model Accuracy", color=secondary_data_color, linewidth=2)
        plt.plot(augmentation_magnitude, estimated_confidence_values2, '--', label=f'k={k2}, min=Model Accuracy', color=est_conf_color, linewidth=2)
        plt.plot(augmentation_magnitude, estimated_confidence_values1, '-.', label=f'k={k1}, min={chance_1}', color=est_conf_color, linewidth=2)
        plt.plot(const_severity, const_k, '-', label=f'k=2, min=chance', color=est_conf_color, linewidth=2)
        plt.plot(augmentation_magnitude, ssim, '-', label='SSIM', color=metrics_color, linewidth=1, alpha=0.8)
        plt.plot(augmentation_magnitude, ncc, '--', label='NCC', color=metrics_color, linewidth=1, alpha=0.8)
        plt.plot(augmentation_magnitude, uiq, '.-', label='UIQ', color=metrics_color, linewidth=1, alpha=0.8)
        plt.plot(augmentation_magnitude, scc, ':', label='SCC', color=metrics_color, linewidth=1, alpha=0.8)
        plt.ylim(0.0, 1.0)
        ticks = np.linspace(0.0, 1.0, 6)
        plt.axhline(y=chance, color=est_conf_color, linestyle=':', label=f'Chance', linewidth=2, alpha=0.6)
        plt.axhline(y=chance_2, color=secondary_data_color, linestyle=':', label=f'Model Accuracy lower bound', linewidth=2, alpha=0.6)
        plt.yticks(list(ticks) + [chance, chance_2])
        plt.gca().get_yticklabels()[-2].set_color(est_conf_color)
        plt.gca().get_yticklabels()[-1].set_color(secondary_data_color)
        plt.xlabel(f"Augmentation Magnitude", fontsize=16, fontweight='bold')
        plt.legend(fontsize=14, frameon=False)


    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.ylabel("Label Confidence", fontsize=16, fontweight='bold')
    plt.tick_params(axis='both', labelsize=14)
    plt.grid(visible=True, axis='y', which='major', linestyle='--', linewidth=0.5, alpha=0.5)
    #plt.gca().grid(False)
    plt.tight_layout()
    file_name = f"visualization/final_plots/{augmentation_type}_plot.pdf"
    plt.savefig(file_name, format='pdf')
    plt.show()

if __name__ == "__main__":

    individual_plots("Contrast")