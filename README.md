# Analiza Danych w Medycynie — Predykcja Ryzyka Udaru Mózgu

Projekt akademicki poświęcony zaawansowanej analizie statystycznej i uczeniu maszynowemu w medycynie na podstawie zbioru danych klinicznych dotyczących udarów mózgu (*Stroke Prediction Dataset*, ~5 110 rekordów).

## Struktura repozytorium

```text
MedycznaAnaliza/
├── data/
│   ├── raw/                       # Surowe dane źródłowe (healthcare-dataset-stroke-data.csv)
│   └── processed/                 # Przetworzone i zaimputowane dane tabelaryczne
├── notebooks/                     # Samodzielne notebooki badawcze (Jupyter / Google Colab)
│   └── 01_stroke_analysis_and_modeling.ipynb
├── src/                           # Moduły wielokrotnego użytku w Pythonie
│   ├── __init__.py
│   ├── config.py                  # Ścieżki, URL-e awaryjne, definicje kolumn klinicznych
│   ├── data.py                    # Automatyczny pipeline pobierania i ładowania danych
│   ├── biostatistics.py           # Narzędzia analizy biostatystycznej i testów hipotez
│   ├── modeling.py                # Pipeline'y ML, obsługa asymetrii klas i progi decyzyjne
│   └── app.py                     # Interaktywny kalkulator ryzyka klinicznego w Streamlit
├── docs/                          # Dokumentacja procesu i agentów
├── requirements.txt               # Zależności biblioteczne
└── README.md
```

## Źródło danych (Data Acquisition)

Pipeline używa wyłącznie śledzonego pliku
`data/raw/healthcare-dataset-stroke-data.csv`. Brak pliku, uszkodzony CSV lub
niezgodne kolumny kończą działanie konkretnym komunikatem naprawczym; nie ma
zdalnego fallbacku ani cichego pustego wyniku.

### Użycie w kodzie Python:

```python
from src.data import load_data

# Wczytanie i walidacja śledzonego pliku
df = load_data()
print(df.shape)  # (5110, 12)
```

## Instalacja środowiska lokalnego

Wymagany Python 3.10+.

```bash
# Utworzenie środowiska i instalacja zależności (macOS / Linux)
./scripts/setup_venv.sh
source .venv/bin/activate

# Uruchomienie aplikacji
streamlit run app.py
```

Skrypt tworzy środowisko `.venv` (pomijane przez Git) i instaluje zależności
z `requirements.txt`. Można wskazać inny interpreter, ustawiając
`PYTHON_BIN`, na przykład `PYTHON_BIN=python3.12 ./scripts/setup_venv.sh`.

Na Windows należy utworzyć środowisko poleceniem `py -m venv .venv`,
aktywować je przez `.\.venv\Scripts\activate`, a następnie wykonać
`python -m pip install -r requirements.txt`.
