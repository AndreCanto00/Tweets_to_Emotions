setup:
	python3 -m venv .venv
	. .venv/bin/activate

install: setup
	. .venv/bin/activate && \
	pip install --upgrade pip && \
	pip install -r requirements.txt

test: setup
	. .venv/bin/activate && \
	python -m pytest --nbval Tweets_to_Emotions.ipynb

format: setup
	. .venv/bin/activate && \
	black Tweets_to_Emotions.ipynb

lint: setup
	. .venv/bin/activate && \
	pylint --disable=R,C,W1203 Tweets_to_Emotions.ipynb

all: install lint test
