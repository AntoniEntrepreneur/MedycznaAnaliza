# Analiza danych o udarze

Projekt na zajęcia z analizy danych w medycynie. Analizujemy *Stroke Prediction Dataset*
([Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset))
i budujemy prosty model przesiewowy dla osób dorosłych.

## Struktura

```text
data/raw/healthcare-dataset-stroke-data.csv  # dane
notebooks/stroke_risk_analysis.ipynb         # analiza i modele
app.py                                       # aplikacja Streamlit
```

## Uruchomienie

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

jupyter notebook notebooks/stroke_risk_analysis.ipynb   # notebook
streamlit run app.py                                    # aplikacja
```

Aplikacja ma dwa widoki: **Populacja** (opis kohorty dorosłych) oraz
**Kalkulator pacjenta** (Logistic Regression trenowana przy starcie).

Wynik ma charakter akademicki — nie jest diagnozą ani zaleceniem medycznym.
