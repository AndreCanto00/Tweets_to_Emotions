import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from typing import Dict, List
import pandas as pd

def plot_content_distributions(df):
    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    sns.kdeplot(x=df['content_len'], hue=df['sentiment'], ax=ax[0], label='content len', legend=False)
    ax[0].set_title('Distribution of Content Length by Sentiment')
    sns.kdeplot(x=df['content_word'], hue=df['sentiment'], ax=ax[1], label='content word')
    ax[1].set_title('Distribution of Content Word Count by Sentiment')
    return fig



def plot_sentiment_wordclouds(data):
    """
    Genera e visualizza word cloud per ogni sentimento nel dataset.

    Args:
        data (pd.DataFrame): DataFrame con le colonne 'sentiment' e 'content'.
    """
    sentiments = data['sentiment'].unique().tolist()
    sentiments = sentiments + sentiments[:3]  # Estende la lista per riempire il grid
    
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(15, 15))

    for ax, sentiment in zip(axes.flatten(), sentiments):
        text = " ".join(data[data['sentiment'] == sentiment]['content'])
        cloud = WordCloud(width=800, height=800, background_color='white', min_font_size=10).generate(text)
        ax.imshow(cloud)
        ax.set_title(sentiment)
        ax.axis("off")

    plt.tight_layout()
    plt.show()
    

def plot_sentiment_histogram(data, category_counts):
    """
    Visualizza un istogramma delle categorie di sentiment.

    Parameters:
    data (DataFrame): Il DataFrame contenente i dati dei tweet.
    category_counts (dict): Un dizionario con le categorie di sentiment come chiavi e i conteggi come valori.
    """
    for sentiment_category in data['sentiment']:
        sentiment_category = sentiment_category.strip().lower()
        if sentiment_category in category_counts:
            category_counts[sentiment_category] += 1

    categories = list(category_counts.keys())
    counts = list(category_counts.values())

    plt.figure(figsize=(10, 6))
    plt.bar(categories, counts)
    plt.xlabel('Sentiment Categories')
    plt.ylabel('Number of Tweets')
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.show()


# src/visualization.py

def plot_empath_analysis(positive_values_of_cat, media):
    """
    Create subplot grid showing Empath analysis for each sentiment category
    
    Args:
        positive_values_of_cat: Dictionary containing positive values for each category
        media: Dictionary containing mean values for each category
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    sentiment_categories = list(positive_values_of_cat.keys())
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    
    for i, sentiment_category in enumerate(sentiment_categories):
        row = i // 4
        col = i % 4
        ax = axes[row, col]
        
        # Plot values and mean line
        ax.plot(positive_values_of_cat[sentiment_category])
        ax.axhline(media[sentiment_category], 
                  color='red', 
                  linestyle='--', 
                  label=f'Average: {media[sentiment_category]:.5f}')
        
        # Customize plot
        ax.set_title(f'{sentiment_category.capitalize()}')
        ax.set_xlabel('Empath Categories')
        ax.set_ylabel('Value')
        ax.legend()
    
    plt.tight_layout()
    return fig




def plot_similarity_distributions(result_df: pd.DataFrame, 
                                ratios: Dict[str, List[float]], 
                                figsize: tuple = (12, 12)) -> None:
    """
    Plot histograms of similarity ratio distributions for each sentiment category.
    
    Args:
        result_df (pd.DataFrame): DataFrame containing sentiment categories
        ratios (Dict[str, List[float]]): Dictionary with sentiment categories as keys
                                        and ratio scores as values
        figsize (tuple): Size of the figure (width, height)
    """
    # Calcola il numero di righe e colonne necessarie per la griglia
    n_categories = len(result_df)
    n_rows = (n_categories + 3) // 4  # Arrotonda per eccesso
    n_cols = 4
    
    # Crea la figura e i subplot
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Appiattisce l'array di axes per una più facile iterazione
    axs_flat = axs.flatten()
    
    # Crea gli istogrammi
    for i in range(n_categories):
        sentiment_category = result_df['Sentiment Category'].iloc[i]
        ratio_scores_category = ratios[sentiment_category]
        
        # Crea l'istogramma
        axs_flat[i].hist(ratio_scores_category, bins=20, edgecolor='k')
        axs_flat[i].set_xlabel('Ratio Scores')
        axs_flat[i].set_ylabel('Frequency')
        axs_flat[i].set_title(sentiment_category)
        axs_flat[i].grid(True)
    
    # Rimuovi i subplot vuoti in eccesso
    for i in range(n_categories, len(axs_flat)):
        fig.delaxes(axs_flat[i])
    
    # Aggiusta il layout
    plt.tight_layout()
    
    return fig, axs