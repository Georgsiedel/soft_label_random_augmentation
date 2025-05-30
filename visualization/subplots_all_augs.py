import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from visualization_utils import compute_occlusion_visibility
from visualization_utils import plot_severity_vs_confidence
import seaborn as sns
import os
from individual_plots import model_confidence

def get_data(visibility_values: list, k: int = 2, chance: float = 0.1):
    confidence_rc_values = []

    confidence_rc_values = 1 - (1 - chance) * (visibility_values) ** k
    confidence_rc_values = np.clip(confidence_rc_values, chance, 1.0)

    return confidence_rc_values


if __name__ == "__main__":

    """HVS Data"""
    rotation_hvs = [1., 0.9985, 0.997, 0.9955, 0.994, 0.9925, 0.991, 0.9895, 0.988, 0.9865, 0.985, 0.9835, 0.982, 0.9805, 0.979, 0.9775, 0.976, 0.9745, 0.973, 0.9715, 0.97, 0.964, 0.958, 0.952, 0.946, 0.94, 0.934, 0.9315, 0.936, 0.9405, 0.945]
    contrast_hvs = [0.32, 0.32, 0.64254054, 0.96603963, 0.96734732, 0.96865501, 0.9699627, 0.9712704, 0.97257809, 0.97388578, 0.97519347, 0.97650117, 0.97780886, 0.97911655, 0.98042424, 0.98173193, 0.98303963, 0.98434732, 0.98565501, 0.98696271, 0.9882704, 0.98957809, 0.99088578, 0.99219347, 0.99350117, 0.99480886, 0.99611655, 0.99742424, 0.99873194, 1., 1.]
    occlusion_hvs = [1., 0.9888205, 0.97764103, 0.96646153, 0.95528205, 0.94410256, 0.93292308, 0.92174358, 0.91056411, 0.89938461, 0.88820511, 0.87702564, 0.86584614, 0.85466667, 0.84348717, 0.83230768, 0.82112822, 0.80994873, 0.79876924, 0.78758975, 0.776, 0.764, 0.75466667, 0.72666669, 0.68000003, 0.68533333, 0.65333335, 0.58400002, 0.51066667, 0.38800001, 0.216]
    
    #only if you want to plot two sides of the rotation-like transforms one 
    rotation_mirrored = rotation_hvs[::-1]
    rotation_hvs = rotation_mirrored + rotation_hvs
    
    #only if you want to plot two sides of the occlusion-like transforms one 
    occlusion_mirrored = occlusion_hvs[::-1]
    occlusion_hvs = occlusion_mirrored + occlusion_hvs
    contrast_hvs = contrast_hvs + [1.0] * 31
    """HVS Data"""

    """Plotting Parameters"""
    plt.rcParams.update({'font.size': 12, 'font.family': 'DejaVu Sans'})
    # plt.style.use(['science', 'no-latex'])
    # plt.rcParams['font.family'] = 'DejaVu Sans'
    main_data_color = '#377eb8'
    secondary_data_color = '#4daf4a'
    highlight_color = '#e41a1c'
    est_conf_color = 'red'
    metrics_color = '#333333'
    constk_color = '#984ea3'
    """Plotting Parameters"""
    
    num_bins = 31

    sns.set_palette("colorblind")
    fig, ax = plt.subplots(4, 3, figsize=(8, 10))
    ax[3, 2].remove()
    
    """Rotate"""
    augmentation_type = "Rotate"
    min_val, max_val = 0.0, 135.0

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
    '''
    
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
    '''

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

    ax[0, 2].plot(augmentation_magnitude, rotation_hvs, '-', label='Rotation HVS', color=main_data_color, linewidth=2)
    ax[0, 2].plot(augmentation_magnitude, model_accuracy, "-", label="Model Accuracy", color=secondary_data_color, linewidth=2)
    ax[0, 2].plot(augmentation_magnitude, estimated_confidence_values2, '--', label=f'k={k2}, min=HVS', color=est_conf_color, linewidth=2)
    ax[0, 2].plot(augmentation_magnitude, estimated_confidence_values1, '-.', label=f'k={k1}, min={chance_1}', color=est_conf_color, linewidth=2)
    ax[0, 2].plot(augmentation_magnitude, const_k, '-', label=f'k=2, min=chance', color=est_conf_color, linewidth=2)
    ax[0, 2].plot(augmentation_magnitude, ssim, '-', label='SSIM', color=metrics_color, linewidth=1, alpha=0.8)
    ax[0, 2].plot(augmentation_magnitude, ncc, '--', label='NCC', color=metrics_color, linewidth=1, alpha=0.8)
    ax[0, 2].plot(augmentation_magnitude, uiq, '.-', label='UIQ', color=metrics_color, linewidth=1, alpha=0.8)
    ax[0, 2].plot(augmentation_magnitude, scc, ':', label='SCC', color=metrics_color, linewidth=1, alpha=0.8)
    ax[0, 2].plot(augmentation_magnitude, sift, '-.', label='SIFT', color=metrics_color, linewidth=1, alpha=0.8)
    ax[0, 2].set_ylim(0.0, 1.0)
    ax[0, 2].axhline(y=chance, color=est_conf_color, linestyle=':', label=f'Chance', linewidth=2, alpha=0.6)
    ax[0, 2].axhline(y=chance_2, color=main_data_color, linestyle=':', label=f'HVS lower bound', linewidth=2, alpha=0.6)
    ax[0, 2].set_title(f"{augmentation_type}", fontsize=9, fontweight='bold')
    ax[0, 2].grid(visible=True, axis='y', which='major', linestyle='--', linewidth=0.5, alpha=0.5)
    ax[0, 2].tick_params(axis='both', labelsize=7)
    
    """Contrast"""
    augmentation_type = 'Contrast'
    augmentation_magnitude, _, model_accuracy = model_confidence(augmentation_type=augmentation_type)

    df_aug = pd.read_csv(f'visualization/non_linear_mapping_data/{augmentation_type}/{augmentation_type}_poly_k_results.csv')
    df_aug = df_aug.sort_values(by='severity')
    const_k = np.array(df_aug['mean_poly_k'])

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

    ax[2, 0].plot(augmentation_magnitude, contrast_hvs, '-', label='Contrast HVS', color=main_data_color, linewidth=2)
    ax[2, 0].plot(augmentation_magnitude, model_accuracy, "-", label="Model Accuracy", color=secondary_data_color, linewidth=2)
    ax[2, 0].plot(augmentation_magnitude, estimated_confidence_values2, '--', label=f'k={k2_neg, k2_pos}, min=HVS', color=est_conf_color, linewidth=2)
    ax[2, 0].plot(augmentation_magnitude, estimated_confidence_values1, '-.', label=f'k={k1}, min={chance_1}', color=est_conf_color, linewidth=2)
    ax[2, 0].plot(augmentation_magnitude, const_k, '-', label=f'k=2, min=chance', color=est_conf_color, linewidth=2)
    ax[2, 0].plot(augmentation_magnitude, ssim, '-', label='SSIM', color=metrics_color, linewidth=1, alpha=0.8)
    ax[2, 0].plot(augmentation_magnitude, ncc, '--', label='NCC', color=metrics_color, linewidth=1, alpha=0.8)
    ax[2, 0].plot(augmentation_magnitude, uiq, '.-', label='UIQ', color=metrics_color, linewidth=1, alpha=0.8)
    ax[2, 0].plot(augmentation_magnitude, scc, ':', label='SCC', color=metrics_color, linewidth=1, alpha=0.8)
    ax[2, 0].set_ylim(0.0, 1.0)
    ax[2, 0].axhline(y=chance, color=est_conf_color, linestyle=':', label=f'Chance', linewidth=2, alpha=0.6)
    ax[2, 0].axhline(y=chance_2, color=main_data_color, linestyle=':', label=f'HVS lower bound', linewidth=2, alpha=0.6)
    ax[2, 0].set_title(f"{augmentation_type}", fontsize=9, fontweight='bold')
    ax[2, 0].grid(visible=True, axis='y', which='major', linestyle='--', linewidth=0.5, alpha=0.5)
    ax[2, 0].tick_params(axis='both', labelsize=7)
    
    #plt.xlabel(f"Augmentation Magnitude", fontsize=16, fontweight='bold')
    #ax[0, 1].set_xticks(np.arange(-1.0, 1.1, 0.4))

    """Brightness"""
    augmentation_type = 'Brightness'
    augmentation_magnitude, _, model_accuracy = model_confidence(augmentation_type=augmentation_type)

    df_aug = pd.read_csv(f'visualization/non_linear_mapping_data/{augmentation_type}/{augmentation_type}_poly_k_results.csv')
    df_aug = df_aug.sort_values(by='severity')
    const_k = np.array(df_aug['mean_poly_k'])

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

    ax[1, 2].plot(augmentation_magnitude, contrast_hvs, '-', label='Contrast HVS', color=main_data_color, linewidth=2)
    ax[1, 2].plot(augmentation_magnitude, model_accuracy, "-", label="Model Accuracy", color=secondary_data_color, linewidth=2)
    ax[1, 2].plot(augmentation_magnitude, estimated_confidence_values2, '--', label=f'k={k2_neg, k2_pos}, min=HVS, Model Accuracy', color=est_conf_color, linewidth=2)
    ax[1, 2].plot(augmentation_magnitude, estimated_confidence_values1, '-.', label=f'k={k1}, min={chance_1}', color=est_conf_color, linewidth=2)
    ax[1, 2].plot(augmentation_magnitude, const_k, '-', label=f'k=2, min=chance', color=est_conf_color, linewidth=2)
    ax[1, 2].plot(augmentation_magnitude, ssim, '-', label='SSIM', color=metrics_color, linewidth=1, alpha=0.8)
    ax[1, 2].plot(augmentation_magnitude, ncc, '--', label='NCC', color=metrics_color, linewidth=1, alpha=0.8)
    ax[1, 2].plot(augmentation_magnitude, uiq, '.-', label='UIQ', color=metrics_color, linewidth=1, alpha=0.8)
    ax[1, 2].plot(augmentation_magnitude, scc, ':', label='SCC', color=metrics_color, linewidth=1, alpha=0.8)
    ax[1, 2].set_ylim(0.0, 1.0)
    ax[1, 2].axhline(y=chance, color=est_conf_color, linestyle=':', label=f'Chance', linewidth=2, alpha=0.6)
    ax[1, 2].axhline(y=chance_2, color=main_data_color, linestyle=':', label=f'HVS lower bound', linewidth=2, alpha=0.6)
    ax[1, 2].set_title(f"{augmentation_type}", fontsize=9, fontweight='bold')
    ax[1, 2].grid(visible=True, axis='y', which='major', linestyle='--', linewidth=0.5, alpha=0.5)
    ax[1, 2].tick_params(axis='both', labelsize=7)

    """Color"""
    augmentation_type = 'Color'
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

    ax[2, 2].plot(augmentation_magnitude, model_accuracy, "-", label="Model Accuracy", color=secondary_data_color, linewidth=2)
    ax[2, 2].plot(augmentation_magnitude, estimated_confidence_values2, '--', label=f'k={k2}, min=Model Accuracy', color=est_conf_color, linewidth=2)
    ax[2, 2].plot(augmentation_magnitude, estimated_confidence_values1, '-.', label=f'k={k1}, min={chance_1}', color=est_conf_color, linewidth=2)
    ax[2, 2].plot(augmentation_magnitude, const_k, '-', label=f'k=2, min=chance', color=est_conf_color, linewidth=2)
    ax[2, 2].plot(augmentation_magnitude, ssim, '-', label='SSIM', color=metrics_color, linewidth=1, alpha=0.8)
    ax[2, 2].plot(augmentation_magnitude, ncc, '--', label='NCC', color=metrics_color, linewidth=1, alpha=0.8)
    ax[2, 2].plot(augmentation_magnitude, uiq, '.-', label='UIQ', color=metrics_color, linewidth=1, alpha=0.8)
    ax[2, 2].plot(augmentation_magnitude, scc, ':', label='SCC', color=metrics_color, linewidth=1, alpha=0.8)
    ax[2, 2].set_ylim(0.8, 1.0)
    ax[2, 2].set_yticks([0.8, 0.9, 1.0])
    ax[2, 2].axhline(y=chance_2, color=secondary_data_color, linestyle=':', label=f'Model Accuracy lower bound', linewidth=2, alpha=0.6)
    ax[2, 2].set_title(f"{augmentation_type}", fontsize=9, fontweight='bold')
    ax[2, 2].grid(visible=True, axis='y', which='major', linestyle='--', linewidth=0.5, alpha=0.5)
    ax[2, 2].tick_params(axis='both', labelsize=7)

    #ax[0, 3].set_yticks(np.arange(0.90, 1.01, 0.01))
    #ax[0, 3].set_xticks(np.arange(-1.0, 1.1, 0.4))

    """Sharpness"""
    augmentation_type = 'Sharpness'
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

    ax[2, 1].plot(augmentation_magnitude, model_accuracy, "-", label="Model Accuracy", color=secondary_data_color, linewidth=2)
    ax[2, 1].plot(augmentation_magnitude, estimated_confidence_values2, '--', label=f'k={k2_neg, k2_pos}, min=Model Accuracy', color=est_conf_color, linewidth=2)
    ax[2, 1].plot(augmentation_magnitude, estimated_confidence_values1, '-.', label=f'k={k1}, min={chance_1}', color=est_conf_color, linewidth=2)
    ax[2, 1].plot(augmentation_magnitude, const_k, '-', label=f'k=2, min=chance', color=est_conf_color, linewidth=2)
    ax[2, 1].plot(augmentation_magnitude, ssim, '-', label='SSIM', color=metrics_color, linewidth=1, alpha=0.8)
    ax[2, 1].plot(augmentation_magnitude, ncc, '--', label='NCC', color=metrics_color, linewidth=1, alpha=0.8)
    ax[2, 1].plot(augmentation_magnitude, uiq, '.-', label='UIQ', color=metrics_color, linewidth=1, alpha=0.8)
    ax[2, 1].plot(augmentation_magnitude, scc, ':', label='SCC', color=metrics_color, linewidth=1, alpha=0.8)
    ax[2, 1].set_ylim(0.8, 1.0)
    ax[2, 1].set_yticks([0.8, 0.9, 1.0])
    ax[2, 1].axhline(y=chance_2, color=secondary_data_color, linestyle=':', label=f'Model Accuracy lower bound', linewidth=2, alpha=0.6)
    ax[2, 1].set_title(f"{augmentation_type}", fontsize=9, fontweight='bold')
    ax[2, 1].grid(visible=True, axis='y', which='major', linestyle='--', linewidth=0.5, alpha=0.5)
    ax[2, 1].tick_params(axis='both', labelsize=7)
    #ax[0, 3].set_yticks(np.arange(0.90, 1.01, 0.01))
    #ax[0, 3].set_xticks(np.arange(-1.0, 1.1, 0.4))

    """ShearX"""
    augmentation_type = "ShearX"

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
    '''
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
    '''

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

    ax[0, 0].plot(augmentation_magnitude, rotation_hvs, '-', label='Rotation HVS', color=main_data_color, linewidth=2)
    ax[0, 0].plot(augmentation_magnitude, model_accuracy, "-", label="Model Accuracy", color=secondary_data_color, linewidth=2)
    ax[0, 0].plot(augmentation_magnitude, estimated_confidence_values2, '--', label=f'k={k2}, min=HVS', color=est_conf_color, linewidth=2)
    ax[0, 0].plot(augmentation_magnitude, estimated_confidence_values1, '-.', label=f'k={k1}, min={chance_1}', color=est_conf_color, linewidth=2)
    ax[0, 0].plot(augmentation_magnitude, const_k, '-', label=f'k=2, min=chance', color=est_conf_color, linewidth=2)
    ax[0, 0].plot(augmentation_magnitude, ssim, '-', label='SSIM', color=metrics_color, linewidth=1, alpha=0.8)
    ax[0, 0].plot(augmentation_magnitude, ncc, '--', label='NCC', color=metrics_color, linewidth=1, alpha=0.8)
    ax[0, 0].plot(augmentation_magnitude, uiq, '.-', label='UIQ', color=metrics_color, linewidth=1, alpha=0.8)
    ax[0, 0].plot(augmentation_magnitude, scc, ':', label='SCC', color=metrics_color, linewidth=1, alpha=0.8)
    ax[0, 0].plot(augmentation_magnitude, sift, '-.', label='SIFT', color=metrics_color, linewidth=1, alpha=0.8)
    ax[0, 0].set_ylim(0.0, 1.0)
    ax[0, 0].axhline(y=chance, color=est_conf_color, linestyle=':', label=f'Chance', linewidth=2, alpha=0.6)
    ax[0, 0].axhline(y=chance_2, color=main_data_color, linestyle=':', label=f'HVS lower bound', linewidth=2, alpha=0.6)
    ax[0, 0].set_title(f"{augmentation_type}", fontsize=9, fontweight='bold')
    ax[0, 0].grid(visible=True, axis='y', which='major', linestyle='--', linewidth=0.5, alpha=0.5)
    ax[0, 0].tick_params(axis='both', labelsize=7)

    """ShearY"""
    augmentation_type = "ShearY"

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
    '''
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
    '''

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

    ax[0, 1].plot(augmentation_magnitude, rotation_hvs, '-', label='Rotation HVS', color=main_data_color, linewidth=2)
    ax[0, 1].plot(augmentation_magnitude, model_accuracy, "-", label="Model Accuracy", color=secondary_data_color, linewidth=2)
    ax[0, 1].plot(augmentation_magnitude, estimated_confidence_values2, '--', label=f'k={k2}, min=HVS', color=est_conf_color, linewidth=2)
    ax[0, 1].plot(augmentation_magnitude, estimated_confidence_values1, '-.', label=f'k={k1}, min={chance_1}', color=est_conf_color, linewidth=2)
    ax[0, 1].plot(augmentation_magnitude, const_k, '-', label=f'k=2, min=chance', color=est_conf_color, linewidth=2)
    ax[0, 1].plot(augmentation_magnitude, ssim, '-', label='SSIM', color=metrics_color, linewidth=1, alpha=0.8)
    ax[0, 1].plot(augmentation_magnitude, ncc, '--', label='NCC', color=metrics_color, linewidth=1, alpha=0.8)
    ax[0, 1].plot(augmentation_magnitude, uiq, '.-', label='UIQ', color=metrics_color, linewidth=1, alpha=0.8)
    ax[0, 1].plot(augmentation_magnitude, scc, ':', label='SCC', color=metrics_color, linewidth=1, alpha=0.8)
    ax[0, 1].plot(augmentation_magnitude, sift, '-.', label='SIFT', color=metrics_color, linewidth=1, alpha=0.8)
    ax[0, 1].set_ylim(0.0, 1.0)
    ax[0, 1].axhline(y=chance, color=est_conf_color, linestyle=':', label=f'Chance', linewidth=2, alpha=0.6)
    ax[0, 1].axhline(y=chance_2, color=main_data_color, linestyle=':', label=f'HVS lower bound', linewidth=2, alpha=0.6)
    ax[0, 1].set_title(f"{augmentation_type}", fontsize=9, fontweight='bold')
    ax[0, 1].grid(visible=True, axis='y', which='major', linestyle='--', linewidth=0.5, alpha=0.5)
    ax[0, 1].tick_params(axis='both', labelsize=7)

    """TranslateX"""
    augmentation_type = "TranslateX"

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
    '''
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
    '''

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

    ax[1, 0].plot(augmentation_magnitude, occlusion_hvs, '-', label='Occlusion HVS', color=main_data_color, linewidth=2)
    ax[1, 0].plot(augmentation_magnitude, model_accuracy, "-", label="Model Accuracy", color=secondary_data_color, linewidth=2)
    ax[1, 0].plot(augmentation_magnitude, estimated_confidence_values2, '--', label=f'k={k2}, min=HVS', color=est_conf_color, linewidth=2)
    ax[1, 0].plot(augmentation_magnitude, estimated_confidence_values1, '-.', label=f'k={k1}, min={chance_1}', color=est_conf_color, linewidth=2)
    ax[1, 0].plot(augmentation_magnitude, const_k, '-', label=f'k=2, min=chance', color=est_conf_color, linewidth=2)
    ax[1, 0].plot(augmentation_magnitude, ssim, '-', label='SSIM', color=metrics_color, linewidth=1, alpha=0.8)
    ax[1, 0].plot(augmentation_magnitude, ncc, '--', label='NCC', color=metrics_color, linewidth=1, alpha=0.8)
    ax[1, 0].plot(augmentation_magnitude, uiq, '.-', label='UIQ', color=metrics_color, linewidth=1, alpha=0.8)
    ax[1, 0].plot(augmentation_magnitude, scc, ':', label='SCC', color=metrics_color, linewidth=1, alpha=0.8)
    ax[1, 0].plot(augmentation_magnitude, sift, '-.', label='SIFT', color=metrics_color, linewidth=1, alpha=0.8)
    ax[1, 0].set_ylim(0.0, 1.0)
    ax[1, 0].axhline(y=chance, color=est_conf_color, linestyle=':', label=f'Chance', linewidth=2, alpha=0.6)
    ax[1, 0].axhline(y=chance_2, color=main_data_color, linestyle=':', label=f'HVS lower bound', linewidth=2, alpha=0.6)
    ax[1, 0].set_title(f"{augmentation_type}", fontsize=9, fontweight='bold')
    ax[1, 0].grid(visible=True, axis='y', which='major', linestyle='--', linewidth=0.5, alpha=0.5)
    ax[1, 0].tick_params(axis='both', labelsize=7)


    """TranslateY"""
    augmentation_type = "TranslateY"

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
    '''
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
    '''

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

    ax[1, 1].plot(augmentation_magnitude, occlusion_hvs, '-', label='Occlusion HVS', color=main_data_color, linewidth=2)
    ax[1, 1].plot(augmentation_magnitude, model_accuracy, "-", label="Model Accuracy", color=secondary_data_color, linewidth=2)
    ax[1, 1].plot(augmentation_magnitude, estimated_confidence_values2, '--', label=f'k={k2}, min=HVS', color=est_conf_color, linewidth=2)
    ax[1, 1].plot(augmentation_magnitude, estimated_confidence_values1, '-.', label=f'k={k1}, min={chance_1}', color=est_conf_color, linewidth=2)
    ax[1, 1].plot(augmentation_magnitude, const_k, '-', label=f'k=2, min=chance', color=est_conf_color, linewidth=2)
    ax[1, 1].plot(augmentation_magnitude, ssim, '-', label='SSIM', color=metrics_color, linewidth=1, alpha=0.8)
    ax[1, 1].plot(augmentation_magnitude, ncc, '--', label='NCC', color=metrics_color, linewidth=1, alpha=0.8)
    ax[1, 1].plot(augmentation_magnitude, uiq, '.-', label='UIQ', color=metrics_color, linewidth=1, alpha=0.8)
    ax[1, 1].plot(augmentation_magnitude, scc, ':', label='SCC', color=metrics_color, linewidth=1, alpha=0.8)
    ax[1, 1].plot(augmentation_magnitude, sift, '-.', label='SIFT', color=metrics_color, linewidth=1, alpha=0.8)
    ax[1, 1].set_ylim(0.0, 1.0)
    ax[1, 1].axhline(y=chance, color=est_conf_color, linestyle=':', label=f'Chance', linewidth=2, alpha=0.6)
    ax[1, 1].axhline(y=chance_2, color=main_data_color, linestyle=':', label=f'HVS lower bound', linewidth=2, alpha=0.6)
    ax[1, 1].set_title(f"{augmentation_type}", fontsize=9, fontweight='bold')
    ax[1, 1].grid(visible=True, axis='y', which='major', linestyle='--', linewidth=0.5, alpha=0.5)
    ax[1, 1].tick_params(axis='both', labelsize=7)

    """Posterize"""
    augmentation_type = 'Posterize'

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

    ax[3, 0].plot(unique_augmentation_magnitudes, unique_model_accuracy, "-", label="Model Accuracy", color=secondary_data_color, linewidth=2)
    ax[3, 0].plot(unique_augmentation_magnitudes, estimated_confidence_values2, '--', label=f'k={k2}, min=Model Accuracy', color=est_conf_color, linewidth=2)
    ax[3, 0].plot(unique_augmentation_magnitudes, estimated_confidence_values1, '-.', label=f'k={k1}, min={chance_1}', color=est_conf_color, linewidth=2)
    ax[3, 0].plot(unique_augmentation_magnitudes, unique_const_k, '-', label=f'k=2, min=chance', color=est_conf_color, linewidth=2)
    ax[3, 0].plot(augmentation_magnitude, ssim, '-', label='SSIM', color=metrics_color, linewidth=1, alpha=0.8)
    ax[3, 0].plot(augmentation_magnitude, ncc, '--', label='NCC', color=metrics_color, linewidth=1, alpha=0.8)
    ax[3, 0].plot(augmentation_magnitude, uiq, '.-', label='UIQ', color=metrics_color, linewidth=1, alpha=0.8)
    ax[3, 0].plot(augmentation_magnitude, scc, ':', label='SCC', color=metrics_color, linewidth=1, alpha=0.8)
    ax[3, 0].set_ylim(0.5, 1.0)
    ax[3, 0].axhline(y=chance_2, color=secondary_data_color, linestyle=':', label=f'Model Accuracy lower bound', linewidth=2, alpha=0.6)
    ax[3, 0].set_title(f"{augmentation_type}", fontsize=9, fontweight='bold')
    ax[3, 0].grid(visible=True, axis='y', which='major', linestyle='--', linewidth=0.5, alpha=0.5)
    ax[3, 0].tick_params(axis='both', labelsize=7)

    """Solarize: TBD again for ONLY positive augmentation range"""
    augmentation_type = 'Solarize'

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

    ax[3, 1].plot(augmentation_magnitude, model_accuracy, "-", label="Model Accuracy", color=secondary_data_color, linewidth=2)
    ax[3, 1].plot(augmentation_magnitude, estimated_confidence_values2, '--', label=f'k={k2}, min=Model Accuracy', color=est_conf_color, linewidth=2)
    ax[3, 1].plot(augmentation_magnitude, estimated_confidence_values1, '-.', label=f'k={k1}, min={chance_1}', color=est_conf_color, linewidth=2)
    ax[3, 1].plot(const_severity, const_k, '-', label=f'k=2, min=chance', color=est_conf_color, linewidth=2)
    ax[3, 1].plot(augmentation_magnitude, ssim, '-', label='SSIM', color=metrics_color, linewidth=1, alpha=0.8)
    ax[3, 1].plot(augmentation_magnitude, ncc, '--', label='NCC', color=metrics_color, linewidth=1, alpha=0.8)
    ax[3, 1].plot(augmentation_magnitude, uiq, '.-', label='UIQ', color=metrics_color, linewidth=1, alpha=0.8)
    ax[3, 1].plot(augmentation_magnitude, scc, ':', label='SCC', color=metrics_color, linewidth=1, alpha=0.8)
    ax[3, 1].set_ylim(0.0, 1.0)
    ax[3, 1].axhline(y=chance, color=est_conf_color, linestyle=':', label=f'Chance', linewidth=2, alpha=0.6)
    ax[3, 1].axhline(y=chance_2, color=secondary_data_color, linestyle=':', label=f'Model Accuracy lower bound', linewidth=2, alpha=0.6)
    ax[3, 1].set_title(f"{augmentation_type}", fontsize=9, fontweight='bold')
    ax[3, 1].grid(visible=True, axis='y', which='major', linestyle='--', linewidth=0.5, alpha=0.5)
    ax[3, 1].tick_params(axis='both', labelsize=7)

    # plt.tight_layout()

    legend_labels = [
    ('Model Accuracy', secondary_data_color, '-'),
    ('HVS', main_data_color, '-'),
    ('k=2, p$\geq$chance', est_conf_color, '-'),
    ('k=2, p$\geq$0.7', est_conf_color, '-.'),
    ('k=est. (HVS/Model Accuracy)', est_conf_color, '--'),
    ('Image Comparison Metrics', metrics_color, '-'),
    ('chance = 0.1', est_conf_color, ':'),
    ('min (HVS)', main_data_color, ':'),
    ('min (Model Accuracy)', secondary_data_color, ':')
]

    handles = []

    for label, color, linestyle in legend_labels:
        line = plt.Line2D([0], [0], color=color, linestyle=linestyle, linewidth=2, label=label)
        handles.append(line)

    fig.legend(handles=handles, handlelength=2.5, loc='center', bbox_to_anchor=(0.83, 0.15), fontsize=9.5, frameon=False)

    fig.text(0.5, 0.015, 'Augmentation Magnitude', ha='center', fontsize=11, fontweight='bold')  # X-axis label
    fig.text(0.015, 0.5, 'Label Confidence', va='center', rotation='vertical', fontsize=11, fontweight='bold')  # Y-axis label

    # Adjust tight layout to fit legend
    plt.tight_layout(rect=[0.02, 0.02, 1, 1])

    file_name = "visualization/final_plots/all_augmentations_subpot.pdf"
    plt.savefig(file_name, format='pdf')
    plt.show()