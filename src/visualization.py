import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

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


def plot_empath_categories(positive_values_of_cat, media):
    """Plot Empath categories analysis"""
    sentiment_categories = list(positive_values_of_cat.keys())
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    
    for i, sentiment_category in enumerate(sentiment_categories):
        row = i // 4
        col = i % 4
        ax = axes[row, col]
        ax.plot(positive_values_of_cat[sentiment_category])
        ax.axhline(media[sentiment_category], color='red', linestyle='--', 
                  label=f'Average: {media[sentiment_category]:.5f}')
        ax.set_title(f'{sentiment_category.capitalize()}')
        ax.set_xlabel('Empath Categories')
        ax.set_ylabel('Value')
        ax.legend()
    
    plt.tight_layout()
    return fig