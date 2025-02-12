# tests/test_visualization.py

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