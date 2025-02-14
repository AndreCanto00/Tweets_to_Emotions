import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.cumulative_visualization import (
    plot_cumulative_similarity_distributions,
    print_distribution_statistics
)

@pytest.fixture
def sample_data():
    result_df = pd.DataFrame({
        'Sentiment Category': ['Happy', 'Sad', 'Angry'],
        'Min Ratio Score': [20, 30, 25],
        'Max Ratio Score': [80, 85, 75],
        'Average Ratio Score': [50, 55, 45]
    })
    
    # Genera dati di esempio con distribuzioni diverse
    ratios = {
        'Happy': np.concatenate([
            np.random.normal(40, 5, 50),
            np.random.normal(60, 5, 50)
        ]),
        'Sad': np.random.normal(55, 15, 100),
        'Angry': np.random.exponential(20, 100) + 40
    }
    
    return result_df, ratios

def test_plot_creation(sample_data):
    result_df, ratios = sample_data
    fig, axs = plot_cumulative_similarity_distributions(result_df, ratios)
    
    assert isinstance(fig, plt.Figure)
    
    # Verifica il numero corretto di subplot
    n_subplots = len([ax for ax in axs.flat if ax.has_data()])
    assert n_subplots == len(result_df)
    
    # Verifica le etichette degli assi
    for ax in axs.flat:
        if ax.has_data():
            assert ax.get_xlabel() == 'Similarity Score'
            assert ax.get_ylabel() == 'Cumulative Percentage'
            assert ax.get_title() in result_df['Sentiment Category'].values
    
    plt.close(fig)

def test_empty_input():
    empty_df = pd.DataFrame(columns=['Sentiment Category'])
    empty_ratios = {}
    
    fig, axs = plot_cumulative_similarity_distributions(empty_df, empty_ratios)
    plt.close(fig)

def test_print_statistics(sample_data, capsys):
    result_df, ratios = sample_data
    print_distribution_statistics(result_df, ratios)
    
    captured = capsys.readouterr()
    output = captured.out
    
    # Verifica che vengano stampate le statistiche per ogni categoria
    for category in result_df['Sentiment Category']:
        assert category in output
        assert "Total comparisons:" in output
        assert "Mean similarity:" in output
        assert "Median similarity:" in output
        assert "Std deviation:" in output

@pytest.mark.parametrize("num_bins", [5, 10, 20])
def test_different_bin_sizes(sample_data, num_bins):
    result_df, ratios = sample_data
    fig, axs = plot_cumulative_similarity_distributions(
        result_df, ratios, num_bins=num_bins
    )
    plt.close(fig)