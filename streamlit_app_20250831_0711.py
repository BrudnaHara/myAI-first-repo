import streamlit as st
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# twardy reset stylu
mpl.rcParams.update(mpl.rcParamsDefault)
plt.style.use("default")

# Zmień TYLKO kolory dla matplotlib (Streamlit sam się dostosuje)
mpl.rcParams.update({
    "figure.facecolor": "#0E1117",        # ciemne tło Streamlita
    "axes.facecolor": "#0E1117",          # ciemne tło wykresu
    "savefig.facecolor": "#0E1117",       # ciemne tło do zapisu
    "axes.edgecolor": "white",            # białe obramowania
    "text.color": "white",                # biały tekst
    "axes.labelcolor": "white",           # białe etykiety osi
    "xtick.color": "white",               # białe ticki na osi X
    "ytick.color": "white",               # białe ticki na osi Y
})

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
from pathlib import Path
import os
import requests
import json

DEEPSEEK_BASE = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"  # tryb czatowy
DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY")


st.set_page_config(
    page_title="grepuj nerdów", 
    layout="wide",
    page_icon="🖥️",
)

# ---------- Dane ----------
@st.cache_data
def load_data(default_path: str = "35__welcome_survey_cleaned.csv") -> pd.DataFrame:
    p = Path(default_path)
    if p.exists():
        return pd.read_csv(p, sep=";")
    return pd.DataFrame()

df = load_data()
if df.empty:
    st.warning("Nie znaleziono pliku `35__welcome_survey_cleaned.csv`. Wgraj CSV (separator ;).")
    up = st.file_uploader("Wgraj plik CSV", type=["csv"])
    if up:
        df = pd.read_csv(up, sep=";")
    else:
        st.stop()

# ---------- Normalizacja płci ----------
def _map_gender(x):
    if pd.isna(x):
        return x
    if isinstance(x, (int, float, np.integer, np.floating)):
        try:
            xi = int(x)
            return {0: "Kobieta", 1: "Mężczyzna", 2: "Inna/Brak"}.get(xi, x)
        except Exception:
            pass
    s = str(x).strip().lower()
    mapa = {
        "0": "Kobieta", "1": "Mężczyzna", "2": "Inna/Brak",
        "k": "Kobieta", "kobieta": "Kobieta", "f": "Kobieta", "female": "Kobieta",
        "m": "Mężczyzna", "mezczyzna": "Mężczyzna", "mężczyzna": "Mężczyzna", "male": "Mężczyzna",
    }
    return mapa.get(s, x)

if "gender" in df.columns:
    df["gender"] = df["gender"].apply(_map_gender)

# ---------- Sidebar: filtry ----------
st.sidebar.header("grepuj nerdów")
k_choice = st.sidebar.slider("liczba klastrów", 1, 10, 5, 1)

def multiselect_filter(df, col, label=None):
    if col in df.columns:
        opts = sorted(df[col].dropna().unique().tolist())
        sel = st.sidebar.multiselect(label or col, options=opts)
        if sel:
            return df[df[col].isin(sel)]
    return df

df_f = df.copy()
# kategoryczne - ORYGINALNE NAZWY
df_f = multiselect_filter(df_f, "industry", "industry")
df_f = multiselect_filter(df_f, "fav_place", "fav_place") 
df_f = multiselect_filter(df_f, "edu_level", "edu_level")
df_f = multiselect_filter(df_f, "gender", "gender")
df_f = multiselect_filter(df_f, "fav_animals", "fav_animals")

st.sidebar.markdown("---")  # oddzielenie

# bez expandera - bezpośrednio w sidebarze
raw_cols = [
    "hobby_movies","hobby_sport","hobby_art","hobby_other","hobby_video_games",
    "learning_pref_books","learning_pref_offline_courses","learning_pref_personal_projects",
    "learning_pref_teaching","learning_pref_teamwork","learning_pref_workshop","learning_pref_chatgpt",
    "motivation_challenges","motivation_career","motivation_creativity_and_innovation",
    "motivation_money_and_job","motivation_personal_growth","motivation_remote",
]
binary_cols = list(dict.fromkeys(raw_cols))

for col in binary_cols:  # 👈 USUŃ WCIIĘCIE - ma być na poziomie 0
    if col in df_f.columns:
        s = pd.to_numeric(df_f[col], errors="coerce")
        choice = st.sidebar.radio(
            col,
            ["Wszystko","tak","nie"],
            index=0,
            horizontal=True,
            key=f"radio_{col}"  # 👈 USUŃ _{i} bo nie ma enumerate
        )
        if choice != "Wszystko":
            want = 1 if choice == "tak" else 0
            df_f = df_f[df_f[col] == want]  # 👈 Popraw filtrowanie


if df_f.empty:
    st.write("you weirdo as fuck XD")
    st.stop()

# ---------- Klastrowanie ----------
@st.cache_resource
def prepare_clustering(data: pd.DataFrame, n_clusters: int = 5):
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in numeric_cols if c != "id"]
    if not features:
        return None
    clean = data.dropna(subset=features).copy()
    if clean.empty:
        return None

    scaler = StandardScaler()
    X = scaler.fit_transform(clean[features].astype(float))

    k = max(1, min(n_clusters, X.shape[0]))
    if k == 1:
        clusters = np.zeros(X.shape[0], dtype=int)
    else:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X)

    return {"clusters": clusters, "features": features, "kept_idx": clean.index.to_list(), "k": k}

df_enc = df_f.copy()
if "age" in df_enc.columns:
    age_map = {
        "<18": 16, "18-24": 21, "18–24": 21,
        "25-34": 29.5, "35-44": 39.5, "45-54": 49.5,
        "55-64": 59.5, ">=65": 70, "unknown": np.nan
    }
    df_enc["age_num"] = df_enc["age"].map(age_map)

cat_cols_cfg = ["edu_level", "fav_place", "gender", "industry", "city", "fav_animals"]
cat_cols = [c for c in cat_cols_cfg if c in df_enc.columns]

dummies = pd.get_dummies(df_enc[cat_cols], dummy_na=False) if cat_cols else pd.DataFrame(index=df_enc.index)

df_enc = df_enc.apply(lambda s: pd.to_numeric(s, errors="ignore"))
num_base = df_enc.select_dtypes(include=[np.number])
if "age_num" in df_enc.columns and "age_num" not in num_base.columns:
    num_base = pd.concat([num_base, df_enc[["age_num"]]], axis=1)

df_num = pd.concat([num_base, dummies], axis=1)
df_num = df_num.dropna(axis=1, how="all")
df_num = df_num.loc[:, df_num.nunique() > 1]

res = prepare_clustering(df_num, k_choice)
if res is None:
    st.write("you weirdo as fuck XD")
    st.stop()

df_view = df_f.copy()
df_view["cluster"] = np.nan
df_view.loc[res["kept_idx"], "cluster"] = res["clusters"]
df_clust = df_view.dropna(subset=["cluster"]).copy()
df_clust["cluster"] = df_clust["cluster"].astype(int)
if df_clust.empty:
    st.write("you weirdo as fuck XD")
    st.stop()

bin_cols = list(dict.fromkeys(raw_cols))
bin_present = [c for c in bin_cols if c in df_clust.columns]

# ===== OPIS KLASTRU — DeepSeek (tylko śmieszek-linuxiarz) =====

if not DEEPSEEK_KEY:
    st.warning("⚠️ Brak klucza API DeepSeek. Dodaj DEEPSEEK_API_KEY do zmiennych środowiskowych.")
    st.info("💡 Tip: Utwórz plik .env z DEEPSEEK_API_KEY=twój_klucz")
else:
    # Cache nazw grup na 24h
    @st.cache_data(ttl=86400, show_spinner=False)
    def generate_group_name(cluster_id, cluster_data, bin_present):
        """Generuje kreatywną nazwę dla grupy"""
        try:
            # Przygotuj dane dla promptu
            preview_rows = []
            for c in bin_present:
                s = pd.to_numeric(cluster_data[c], errors="coerce")
                p = float((s == 1).mean() * 100) if s.notna().any() else 0.0
                preview_rows.append({"feature": c, "share_of_1_pct": round(p, 1)})
            
            preview_rows.sort(key=lambda r: r["share_of_1_pct"], reverse=True)
            
            user_prompt = f"""
Grupa {cluster_id} ({len(cluster_data)} osób) ma następujące charakterystyki:
{chr(10).join([f"- {r['feature']}: {r['share_of_1_pct']}%" for r in preview_rows[:5]])}

Wymyśl nazwę dla tej grupy (max 2-3 słowa) tak jak by ją nazwał Linus Torvalds 
Nazwa powinna być po polsku i nawiązywać do cech grupy.
Odpowiedz tylko nazwą, bez dodatkowych komentarzy.
"""

            body = {
                "model": DEEPSEEK_MODEL,
                "stream": False,
                "temperature": 0.9,
                "max_tokens": 30,
                "messages": [
                    {"role": "system", "content": "Jesteś kreatywnym nazywaczem grup. Twórz zabawne, trafne nazwy."},
                    {"role": "user", "content": user_prompt},
                ],
            }

            response = requests.post(
                f"{DEEPSEEK_BASE}/chat/completions",
                headers={
                    "Authorization": f"Bearer {DEEPSEEK_KEY}",
                    "Content-Type": "application/json",
                },
                json=body,
                timeout=15,
            )
            
            response.raise_for_status()
            result = response.json()
            
            name = result["choices"][0]["message"]["content"].strip()
            # Usuń cudzysłowy i niechciane znaki
            name = name.replace('"', '').replace("'", "").strip()
            return name
            
        except Exception:
            # Fallback do numeru grupy jeśli API nie działa
            return f"Grupa {cluster_id}"

    # Cache odpowiedzi API na 1 godzinę
    @st.cache_data(ttl=3600, show_spinner=False)
    def generate_cluster_description(cluster_id, cluster_data, bin_present):
        """Generuje opis klastra w stylu linus torvalds"""
        try:
            # Przygotuj dane dla promptu
            preview_rows = []
            for c in bin_present:
                s = pd.to_numeric(cluster_data[c], errors="coerce")
                p = float((s == 1).mean() * 100) if s.notna().any() else 0.0
                preview_rows.append({"feature": c, "share_of_1_pct": round(p, 1)})
            
            preview_rows.sort(key=lambda r: r["share_of_1_pct"], reverse=True)
            
            user_prompt = f"""
Grupa {cluster_id} ({len(cluster_data)} osób) ma takie staty:
{chr(10).join([f"- {r['feature']}: {r['share_of_1_pct']}%" for r in preview_rows[:8]])}

Opisz tę grupę w stylu w jakim zrobił by to Linus Torvalds. Max 5 zdań. 
Pisz tak jakbyś gadał na IRCu czy forum linuxowym. Używaj języka mieszanego linuxowo polskiego. Na koniec dodaj ocenę w skali 1-10 jaką wystawiłby Linus Torvalds w formacie:
Linus Torvalds rate: X/10
"""

            # Wywołanie API
            body = {
                "model": DEEPSEEK_MODEL,
                "stream": False,
                "temperature": 0.85,
                "max_tokens": 300,
                "messages": [
                    {"role": "system", "content": "Jesteś symulacją Linusa Torvaldsa."},
                    {"role": "user", "content": user_prompt},
                ],
            }

            response = requests.post(
                f"{DEEPSEEK_BASE}/chat/completions",
                headers={
                    "Authorization": f"Bearer {DEEPSEEK_KEY}",
                    "Content-Type": "application/json",
                },
                json=body,
                timeout=30,
            )
            
            response.raise_for_status()
            result = response.json()
            
            return result["choices"][0]["message"]["content"].strip()
            
        except requests.exceptions.ConnectionError:
            return "❌ Brak połączenia z internetem"
        except requests.exceptions.Timeout:
            return "⏰ Timeout - API nie odpowiedziało w czasie"
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                return "🔑 Błąd autoryzacji - sprawdź klucz API"
            elif e.response.status_code == 402:
                return "💳 Wymagana płatność - dodaj kartę w DeepSeek"
            elif e.response.status_code == 429:
                return "🚫 Limit rate exceeded - poczekaj chwilę"
            else:
                return f"❌ Błąd HTTP {e.response.status_code}"
        except Exception as e:
            return f"⚠️ Nieoczekiwany błąd: {str(e)}"

    if st.button("Kliknij aby dowiedzieć się co Linus Torvalds myśli o tej grupie", type="primary"):
        with st.spinner("Linus myśli..."):
            description = generate_cluster_description(0, df_clust, bin_present)
            group_name = generate_group_name(0, df_clust, bin_present)
        
        # Wyświetl wynik
        if description.startswith(("❌", "⏰", "🔑", "💳", "🚫", "⚠️")):
            st.error(description)
        else:
            st.markdown(f"# 🎯 {group_name}")  # DUŻY NAPIS
            st.write(description)
 

# ---------- Tabela z osobami ----------
if df_clust.empty:
    st.write("you weirdo as fuck XD")
else:
    st.write(f"zgrepowano {len(df_clust)} nerdów podobnych do ciebie!")
    st.dataframe(df_clust)

# Tabelki pod opisem - dla tych co nie lubią czytać
raw_cols = [
    "hobby_movies","hobby_sport","hobby_art","hobby_other","hobby_video_games",
    "learning_pref_books","learning_pref_offline_courses","learning_pref_personal_projects",
    "learning_pref_teaching","learning_pref_teamwork","learning_pref_workshop","learning_pref_chatgpt",
    "motivation_challenges","motivation_career","motivation_creativity_and_innovation",
    "motivation_money_and_job","motivation_personal_growth","motivation_remote",
]
bin_cols = list(dict.fromkeys(raw_cols))
bin_present = [c for c in bin_cols if c in df_clust.columns]

if bin_present:
    # Przygotuj dane
    names = []
    values = []
    for c in bin_present:
        s = pd.to_numeric(df_clust[c], errors="coerce")
        p = float((s == 1).mean() * 100) if s.notna().any() else 0.0
        names.append(c)
        values.append(p)
    
    # Tworzymy DataFrame
    plot_df = pd.DataFrame({"param": names, "%": values})
    plot_df = plot_df.sort_values("%", ascending=True)
    
    # Tworzymy wykres - MAX SUROWE
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # USUŃ WSZYSTKO CO ZBĘDNE
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_facecolor('none')
    
    # USUŃ OSIE I TICKI
    ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)  # 👈 usuń oś X
    ax.tick_params(axis='y', which='both', left=False, labelleft=True)  # tylko etykiety Y
    
    # Słupki
    bars = ax.barh(plot_df["param"], plot_df["%"], color='#1f77b4', height=0.6)
    ax.set_xlim(0, 100)
    
    # Tylko wartości na słupkach - bez tytułów, bez osi
    for i, v in enumerate(plot_df["%"]):
        ax.text(v + 1, i, f"{v:.0f}%", va='center', fontweight='bold', fontsize=10)
    
    st.pyplot(fig)

else:
    st.info("Brak pól binarnych do podsumowania.")
    
# Easter egg
st.markdown("---")
with st.expander("🖥️ **konsola pomocy (wpisz komendę)**"):
    help_input = st.text_input("$", value="", key="help_input", placeholder="wpisz 'help' lub 'man'")
    
    if help_input.strip() == "help":
        st.code("""
Dostępne komendy:
- help          -> pokazuje tę wiadomość
- man           -> manual demona grupowania
- grepuj_nerdów -> uruchamia główną funkcję
- exit          -> zamyka pomoc (faktycznie nie zamyka, lol)
        """)
    
    elif help_input.strip() == "man":
        st.code("""
NAZWA:
    demony-grupowania – demon grupowania użytkowników na podstawie podobieństwa cech

SKŁADNIA:
    demony-grupowania [--algorithm=kmeans] [--cluster=5] [--verbose]

OPIS:
    Demon działający w tle, który grupuje użytkowników na klastry używając 
    niekontrolowanego uczenia maszynowego. Działa jak usługa systemowa.

OPCJE:
    --algorithm   wybór algorytmu (domyślnie: kmeans)  
    --cluster     liczba klastrów (domyślnie: 5)  
    --verbose     szczegółowe logi do /var/log/nerdapp/grupowanie.log

AUTOR:
    NerdApp 1.0 – napisane przez BrudnaHara na Debianie
        """)
    
    elif help_input.strip() == "grepuj_nerdów":
        st.success("uruchamiam grepowanie nerdów...")
        st.write("🔍 przełącz się na zakładkę 'grepuj nerdów' powyżej!")
    
    elif help_input.strip() == "exit":
        st.warning("nie ma wyjścia z pomocą – to jest Streamlit, nie prawdziwy terminal! 😉")
    
    elif help_input.strip() != "":
        st.error(f"Komenda nieznana: '{help_input}'. wpisz 'help' aby uzyskać pomoc.")

st.caption("Kto to czyta ten mieszka w piwnicy XD")
