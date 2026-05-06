.PHONY: install train predict lint clean

install:
	python -m pip install -r requirements.txt

train:
	python -m src.train

predict:
	python -m src.predict --input data/customer_churn.csv --output data/predictions.csv

lint:
	python -m compileall src

clean:
	rm -f data/predictions*.csv model/churn_model.joblib model/metrics.json
