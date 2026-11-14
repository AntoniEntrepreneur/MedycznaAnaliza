# Analiza danych o udarze

Projekt akademicki analizujący *Stroke Prediction Dataset* i budujący prosty
model przesiewowy dla dorosłej kohorty. Notebook i aplikacja korzystają z tego
samego kodu ładowania, kuracji i modelowania danych.

## Struktura

```text
data/raw/healthcare-dataset-stroke-data.csv  # wymagany, śledzony zbiór danych
notebooks/stroke_risk_analysis.ipynb         # liniowy raport analityczny
src/config.py                                # ścieżki i definicje cech
src/data.py                                  # walidacja i kohorty
src/ml.py                                    # preprocesor i modele
app.py                                       # dwa widoki Streamlit
scripts/setup_venv.sh                        # konfiguracja środowiska
```

## Dane i provenance

Plik `data/raw/healthcare-dataset-stroke-data.csv` jest lokalną kopią zbioru
*Stroke Prediction Dataset* autorstwa fedesoriano, opublikowanego w Kaggle:
<https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset>.
Repozytorium śledzi kopię, ale nie zmienia ani nie nadaje jej licencji.
Przed dalszą redystrybucją należy sprawdzić aktualne warunki Kaggle, licencję
źródła i wymagania atrybucji. Projekt nie pobiera danych automatycznie.

Brak pliku, uszkodzony CSV lub brak wymaganej kolumny kończy działanie
czytelnym błędem. Nie jest tworzony pusty ani zdalny zastępnik danych.

## Uruchomienie

Wymagany jest Python 3.10+.

```bash
./scripts/setup_venv.sh
source .venv/bin/activate
```

Notebook można uruchomić z katalogu projektu:

```bash
jupyter nbconvert --to notebook --execute --inplace \
  notebooks/stroke_risk_analysis.ipynb
```

Aplikację uruchamia się poleceniem:

```bash
streamlit run app.py
```

W aplikacji dostępne są dwa widoki:

- **Populacja** — opis dorosłej kohorty, częstość udarów i wykresy cech;
- **Kalkulator pacjenta** — trening modelu w pamięci i oszacowanie
  prawdopodobieństwa dla wprowadzonego profilu.

Widok populacji aplikacji używa kuracyjnej kohorty dorosłych (wiek >= 18,
bez rekordu `gender == "Other"`). Notebook najpierw opisuje cały zwalidowany
plik (w tym osoby pediatryczne i rekord `Other`), a później używa tej samej
kuracyjnej kohorty dorosłych do modelowania. Te zakresy są jawnie oznaczone,
aby statystyki populacyjne nie były mylone z kohortą modelową.

Model nie jest zapisywany ani ładowany z artefaktu. Każde uruchomienie trenuje
pipeline na śledzonym pliku z ustalonym `random_state=42`. Wynik ma charakter
akademicki i przesiewowy; nie jest diagnozą ani zaleceniem medycznym.
