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

## Automatyczne pobieranie danych (Data Acquisition)

Repozytorium zawiera wbudowany mechanizm automatycznego ładowania danych (`src/data.py`).
Dane są automatycznie wyszukiwane lokalnie w folderze `data/raw/`. W przypadku uruchomienia w odizolowanym środowisku (np. Google Colab) bez pobranego repozytorium, pipeline automatycznie pobiera zbiór z publicznego raw URL na GitHubie i ładuje go do pamięci podręcznej.

### Użycie w kodzie Python:

```python
from src.data import load_data

# Automatyczne wykrycie pliku lokalnego lub pobranie z raw GitHub
df = load_data()
print(df.shape)  # (5110, 12)
```

### Jednolinijkowe pobieranie w Google Colab:

Jeśli notebook jest uruchamiany w środowisku Colab bez klonowania całego repozytorium:

```python
import pandas as pd

url = "https://raw.githubusercontent.com/AntoniEntrepreneur/MedycznaAnaliza/main/data/raw/healthcare-dataset-stroke-data.csv"
df = pd.read_csv(url, na_values=["N/A", "NA", ""])
```

## Instalacja środowiska lokalnego

Wymagany Python 3.10+.

```bash
# Utworzenie i aktywacja wirtualnego środowiska
python3 -m venv venv
source venv/bin/activate  # macOS / Linux
# lub: .\venv\Scripts\activate na Windows

# Instalacja zależności
pip install -r requirements.txt
```
