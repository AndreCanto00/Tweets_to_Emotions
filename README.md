# Progetto di Analisi dei Tweet

[![Run Python Tests](https://github.com/AndreCanto00/Tweets_to_Emotions/actions/workflows/test.yml/badge.svg)](https://github.com/AndreCanto00/Tweets_to_Emotions/actions/workflows/test.yml)

Questo repository contiene un progetto di data science per l'analisi dei tweet. Utilizziamo librerie come `pandas`, `matplotlib`, `seaborn`, `nltk` e altre per eseguire la pulizia, la tokenizzazione, la visualizzazione e l'analisi dei dati dei tweet.

## Struttura del Progetto

- `.github/workflows/`: Contiene i workflow di GitHub Actions per l'esecuzione automatica dei test.
- `notebooks/`: Contiene i notebook Jupyter per l'analisi esplorativa dei dati.
- `src/`: Contiene il codice sorgente per la pulizia, la tokenizzazione, la visualizzazione e l'analisi dei dati.
- `tests/`: Contiene i test unitari per il codice sorgente.
- `tweet_emotions.csv`, `concatenated_tweets_by_category.csv`, `detached_tweets_by_category.csv`: File CSV con i dati dei tweet.

## Requisiti

Per eseguire questo progetto, è necessario installare le seguenti dipendenze:

- `pandas`
- `matplotlib`
- `seaborn`
- `nltk`
- `scikit-learn`
- `wordcloud`
- `torch`
- `unidecode`
- `contractions`
- `emoji`
- `empath`
- `transformers`
- `xgboost`
- `pytest`
- `pytest-cov`
- `nbval`

## Installazione

1. Clona il repository:
    ```sh
    git clone https://github.com/tuo-username/progetto-analisi-tweet.git
    cd progetto-analisi-tweet
    ```

2. Crea un ambiente virtuale e installa le dipendenze:
    ```sh
    make install
    ```

## Esecuzione del Progetto

1. Per eseguire il preprocessing dei dati, utilizza il modulo [preprocessing](http://_vscodecontentref_/1):
    ```sh
    python src/preprocessing.py
    ```

2. Per visualizzare i dati, utilizza il modulo `visualization`:
    ```sh
    python src/visualization.py
    ```

3. Per eseguire i test, utilizza il comando:
    ```sh
    make test
    ```

## Test

I test sono scritti utilizzando [pytest](http://_vscodecontentref_/2). Per eseguire i test, utilizza il comando:
```sh
make test

