# Streamlit App — Klasteryzacja + LLM (DeepSeek)

Aplikacja webowa zbudowana w **Streamlit**, uruchamiana na **Streamlit Community Cloud**.

## Funkcje

- **Filtrowanie użytkowników** — możliwość zawężania danych po branży, miejscu, edukacji, płci, hobby i motywacjach.  
- **Klasteryzacja (KMeans)** — grupowanie podobnych użytkowników na podstawie cech binarnych i liczbowych.  
- **Podsumowanie grupy** — wyświetlanie charakterystyki wybranego klastra.  
- **Integracja z LLM** — opis klastra generowany w czasie rzeczywistym przez model **DeepSeek (deepseek-chat)** w chmurze.  
]
## Wymagania

- Python 3.9+  
- Zawartość `requirements.txt` (Streamlit, pandas, scikit-learn, matplotlib, requests, numpy)

## Uruchamianie lokalne

```bash
git clone https://github.com/<twoje-repo>
cd <twoje-repo>
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

Deploy

Repozytorium jest połączone ze Streamlit Community Cloud.
W panelu wystarczy wskazać plik streamlit_app.py i dodać sekrety
DEEPSEEK_API_KEY = "TWÓJ_KLUCZ"

Projekt łączy klasyczne ML (klasteryzacja KMeans) z nowoczesnym LLM (DeepSeek) w aplikacji chmurowej.
export DEEPSEEK_API_KEY="TWÓJ_KLUCZ"
streamlit run streamlit_app.py
