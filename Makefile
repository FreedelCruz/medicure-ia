.PHONY: install test demo app notebooks-smoke clean

install:
	pip install -r requirements.txt

test:
	python -m pytest

demo:            ## interface en mode SIMULATION (sans torch ni poids)
	python -m app.gradio_app --mock

app:             ## interface avec le vrai modèle
	python -m app.gradio_app

notebooks-smoke: ## exécute les 5 notebooks sur un faux dataset (aucune donnée à télécharger)
	bash scripts/smoke_notebooks.sh

clean:
	rm -rf .pytest_cache outputs artifacts/audit_log.jsonl
