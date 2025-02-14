import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import pandas as pd

def plot_cumulative_similarity_distributions(
    result_df: pd.DataFrame,
    ratios: Dict[str, List[float]],
    num_bins: int = 10,
    figsize: Tuple[int, int] = (12, 12)
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot cumulative distribution of similarity ratios for each sentiment category.
    
    Args:
        result_df (pd.DataFrame): DataFrame containing sentiment categories
        ratios (Dict[str, List[float]]): Dictionary of ratio scores per category
        num_bins (int): Number of bins for the histogram
        figsize (tuple): Size of the figure (width, height)
        
    Returns:
        tuple: (Figure, array of Axes)
    """
    def calculate_cumulative_percentages(data: List[float], bins: List[float]) -> Tuple[List[float], List[float]]:
        """Calculate cumulative percentages for histogram data."""
        hist, _ = np.histogram(data, bins)
        total_data = len(data)
        percentages = [(count / total_data) * 100 for count in hist]
        
        # Calcola valori cumulativi (dal più alto al più basso)
        cumulative = []
        current_sum = 0
        for p in percentages[::-1]:
            current_sum += p
            cumulative.append(current_sum)
        
        return cumulative[::-1], hist
    
    # Crea subplot grid
    n_categories = len(result_df)
    n_rows = (n_categories + 3) // 4
    fig, axs = plt.subplots(n_rows, 4, figsize=figsize)
    axs_flat = axs.flatten()
    
    # Crea i grafici per ogni categoria
    for i in range(n_categories):
        row, col = i // 4, i % 4
        sentiment_category = result_df['Sentiment Category'].iloc[i]
        ratio_scores = ratios[sentiment_category]
        
        # Calcola bins
        min_value = min(ratio_scores)
        max_value = max(ratio_scores)
        bin_width = (max_value - min_value) / num_bins
        bins = [min_value + i * bin_width for i in range(num_bins)]
        bins.append(max_value)
        
        # Calcola percentuali cumulative
        cumulative_percentages, hist = calculate_cumulative_percentages(ratio_scores, bins)
        
        # Stampa statistiche
        print(f"\nStatistiche per {sentiment_category}:")
        for j in range(num_bins):
            print(f"Intervallo {j + 1}: Range ({bins[j]:.2f} - {bins[j + 1]:.2f}): "
                  f"{hist[j]} dati ({(hist[j]/len(ratio_scores))*100:.2f}%)")
        
        # Crea il grafico a barre
        axs_flat[i].bar(bins[:-1], cumulative_percentages, 
                       width=bin_width, align='edge')
        axs_flat[i].set_xlabel('Similarity Score')
        axs_flat[i].set_ylabel('Cumulative Percentage')
        axs_flat[i].set_title(sentiment_category)
        axs_flat[i].grid(True)
    
    # Rimuovi subplot vuoti
    for i in range(n_categories, len(axs_flat)):
        fig.delaxes(axs_flat[i])
    
    plt.tight_layout()
    return fig, axs

def print_distribution_statistics(result_df: pd.DataFrame, 
                                ratios: Dict[str, List[float]], 
                                num_bins: int = 10) -> None:
    """
    Print detailed statistics about the distribution of similarity scores.
    
    Args:
        result_df (pd.DataFrame): DataFrame containing sentiment categories
        ratios (Dict[str, List[float]]): Dictionary of ratio scores per category
        num_bins (int): Number of bins for the histogram
    """
    for _, sentiment_category in enumerate(result_df['Sentiment Category']):
        ratio_scores = ratios[sentiment_category]
        print(f"\n{sentiment_category}:")
        print(f"Total comparisons: {len(ratio_scores)}")
        print(f"Mean similarity: {np.mean(ratio_scores):.2f}")
        print(f"Median similarity: {np.median(ratio_scores):.2f}")
        print(f"Std deviation: {np.std(ratio_scores):.2f}")