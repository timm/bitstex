import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Garamond', 'Computer Modern Roman']
plt.rcParams['mathtext.fontset'] = 'cm'
def generate_heatmap(file_path):
    df = pd.read_csv(file_path, sep='\t', header=None, names=['X', 'Y', 'Value'])
    df_filtered = df[df['Value'] >= 0]
    heatmap_data = df_filtered.pivot_table(
        index='X', columns='Y', values='Value', aggfunc='median')
    heatmap_data = heatmap_data.sort_index(ascending=False)
    data_matrix = heatmap_data.fillna(0).values
    sigma = 1.2
    blurred_data = gaussian_filter(data_matrix, sigma=sigma)
    blurred_df = pd.DataFrame(blurred_data, index=heatmap_data.index, columns=heatmap_data.columns)
    plt.figure(figsize=(3.5, 3))
    x_labels = [int(col) if int(col) % 2 == 0 else '' for col in blurred_df.columns]
    y_labels = [int(idx) if int(idx) % 10 == 0 else '' for idx in blurred_df.index]
    ax = sns.heatmap(blurred_df, cmap='viridis', annot=False,
                     xticklabels=x_labels, yticklabels=y_labels)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label('Wins', size=12, weight='bold')
    X_grid, Y_grid = np.meshgrid(np.arange(blurred_data.shape[1]) + 0.5,
                                 np.arange(blurred_data.shape[0]) + 0.5)
    contours = ax.contour(X_grid, Y_grid, blurred_data, levels=[60, 70, 80, 90],
                          colors='black', alpha=0.9, linewidths=1.5)
    ax.clabel(contours, inline=True, fontsize=11, fmt='%1.0f', colors='black')
    plt.xlabel('Check', fontsize=12, fontweight='bold')
    plt.ylabel('Budget', fontsize=12, fontweight='bold')
    plt.xticks(fontsize=10, rotation=0)
    plt.yticks(fontsize=10, rotation=0)
    plt.tight_layout()
    plt.savefig('heatmap_regenerated_final.png', dpi=300)
    plt.show()
if __name__ == "__main__":
    generate_heatmap('lua1000.log')
