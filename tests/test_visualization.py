from src.visualization import plot_empath_analysis
import matplotlib.pyplot as plt
import pytest
import pandas as pd
import numpy as np
from src.visualization import plot_similarity_distributions

def test_plot_empath_analysis():
    """Test Empath visualization function"""
    test_values = {
        'happiness': [0.1, 0.2, 0.3],
        'sadness': [0.2, 0.3, 0.4],
        'anger': [0.3, 0.4, 0.5],
        'fear': [0.4, 0.5, 0.6]
    }
    test_media = {cat: sum(vals)/len(vals) for cat, vals in test_values.items()}
    
    fig = plot_empath_analysis(test_values, test_media)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)  # Clean up



@pytest.fixture
def sample_visualization_data():
    result_df = pd.DataFrame({
        'Sentiment Category': ['Happy', 'Sad', 'Angry'],
        'Min Ratio Score': [20, 30, 25],
        'Max Ratio Score': [80, 85, 75],
        'Average Ratio Score': [50, 55, 45]
    })
    
    ratios = {
        'Happy': np.random.normal(50, 10, 100),
        'Sad': np.random.normal(55, 12, 100),
        'Angry': np.random.normal(45, 15, 100)
    }
    
    return result_df, ratios

def test_plot_creation(sample_visualization_data):
    result_df, ratios = sample_visualization_data
    fig, axs = plot_similarity_distributions(result_df, ratios)
    
    # Verifica che fig sia un oggetto Figure
    assert isinstance(fig, plt.Figure)
    
    # Verifica numero subplot
    n_subplots = len([ax for ax in axs.flat if ax.has_data()])
    assert n_subplots == len(result_df)
    
    # Verifica titoli
    for ax in axs.flat:
        if ax.has_data():
            assert ax.get_title() is not None
    
    plt.close(fig)

def test_empty_input():
    empty_df = pd.DataFrame(columns=['Sentiment Category'])
    empty_ratios = {}
    
    fig, axs = plot_similarity_distributions(empty_df, empty_ratios)
    plt.close(fig)

@pytest.mark.parametrize("n_categories", [1, 4, 7, 13])
def test_different_grid_sizes(n_categories):
    # Test con diversi numeri di categorie
    categories = [f'Cat{i}' for i in range(n_categories)]
    result_df = pd.DataFrame({
        'Sentiment Category': categories,
        'Min Ratio Score': [20] * n_categories,
        'Max Ratio Score': [80] * n_categories,
        'Average Ratio Score': [50] * n_categories
    })
    
    ratios = {cat: np.random.normal(50, 10, 100) for cat in categories}
    
    fig, axs = plot_similarity_distributions(result_df, ratios)
    plt.close(fig)