# Tweet Analysis Project.

[![Run Python Tests](https://github.com/AndreCanto00/Tweets_to_Emotions/actions/workflows/test.yml/badge.svg)](https://github.com/AndreCanto00/Tweets_to_Emotions/actions/workflows/test.yml)

This repository contains a data science project for tweet analysis. We use libraries such as `pandas`, `matplotlib`, `seaborn`, `nltk` and others to perform cleaning, tokenization, visualization and analysis of tweet data.

The file Project_requirements.pdf contains the project requests.
The file Paper.pdf contains the results of the project.

## Project Structure.

- `.github/workflows/`: Contains GitHub Actions workflows for automated test execution.
- `notebooks/`: Contains Jupyter notebooks for exploratory data analysis.
- `src/`: Contains source code for data cleaning, tokenization, visualization, and analysis.
- `tests/`: Contains the unit tests for the source code.
- `tweet_emotions.csv`, `concatenated_tweets_by_category.csv`, `detached_tweets_by_category.csv`: CSV files with tweet data.

## Requirements.

To run this project, you need to install the following dependencies:

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


## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/tuo-username/progetto-analisi-tweet.git
    cd project-analysis-tweet
    ```

2. Create a virtual environment and install dependencies:
    ``sh
    make install
    ```

## Running the Project

1. To perform data preprocessing, use the [preprocessing](http://_vscodecontentref_/1) module:
    ``sh
    python src/preprocessing.py
    ```

2. To visualize the data, use the `visualization` module:
    ```sh
    python src/visualization.py
    ```

3. To run tests, use the command:
    ``sh
    make test
    ```

## Test

Tests are written using [pytest](http://_vscodecontentref_/2). To run the tests, use the command:
``sh
make test


