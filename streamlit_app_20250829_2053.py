import streamlit as st
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# twardy reset stylu
mpl.rcParams.update(mpl.rcParamsDefault)
plt.style.use("default")

# białe tła i czarne fonty
mpl.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.edgecolor": "black",
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
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
    page_icon="🖥️"
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
st.sidebar.header("⚙️ Ustawienia")
k_choice = st.sidebar.slider("Liczba klastrów", 1, 10, 5, 1)

def multiselect_filter(df, col, label=None):
    if col in df.columns:
        opts = sorted(df[col].dropna().unique().tolist())
        sel = st.sidebar.multiselect(label or col, options=opts)
        if sel:
            return df[df[col].isin(sel)]
    return df

df_f = df.copy()
# kategoryczne
df_f = multiselect_filter(df_f, "industry", "Branża")
df_f = multiselect_filter(df_f, "fav_place", "Ulubione miejsce")
df_f = multiselect_filter(df_f, "edu_level", "Wykształcenie")
df_f = multiselect_filter(df_f, "gender", "Płeć")
df_f = multiselect_filter(df_f, "fav_animals", "Ulubione zwierzę")
df_f = multiselect_filter(df_f, "city", "Miasto")

with st.sidebar.expander("Parametry jądra"):
    raw_cols = [
        "hobby_movies","hobby_sport","hobby_art","hobby_other","hobby_video_games",
        "learning_pref_books","learning_pref_offline_courses","learning_pref_personal_projects",
        "learning_pref_teaching","learning_pref_teamwork","learning_pref_workshop","learning_pref_chatgpt",
        "motivation_challenges","motivation_career","motivation_creativity_and_innovation",
        "motivation_money_and_job","motivation_personal_growth","motivation_remote",
    ]
    binary_cols = list(dict.fromkeys(raw_cols))

    for i, col in enumerate(binary_cols):
        if col in df_f.columns:
            s = pd.to_numeric(df_f[col], errors="coerce")
            choice = st.sidebar.radio(
                col,
                ["Wszystko","tak","nie"],
                index=0,
                horizontal=True,
                key=f"radio_bin_{col}_{i}"
            )
            if choice != "Wszystko":
                want = 1 if choice == "tak" else 0
                df_f = df_f[s == want]

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

mode_series = df_clust["cluster"].mode()
if mode_series.empty:
    st.write("you weirdo as fuck XD")
    st.stop()

selected_cluster = int(mode_series.iloc[0])
same_cluster = df_clust[df_clust["cluster"] == selected_cluster]

st.sidebar.header("📌 Wybrana grupa")
st.sidebar.write(f"Automatycznie wybrano **grupę {selected_cluster}**.")
st.sidebar.metric("Liczba osób w grupie", len(same_cluster))

# ---------- Sekcja główna ----------
st.header("nerdy jak ty XD")
if same_cluster.empty:
    st.write("you weirdo as fuck XD")
else:
    st.write(f"Znaleziono {len(same_cluster)} osób podobnych do Ciebie!")
    st.dataframe(same_cluster)


# ---------- Charakterystyka grup ----------
st.header("demony grupowania")

cluster_data = df_clust[df_clust["cluster"] == selected_cluster]
st.write(f"**Grupa {int(selected_cluster)}** — {len(cluster_data)} osób")

# ===== OPIS KLASTRU — DeepSeek (tylko śmieszek-linuxiarz) =====
st.subheader("🧠 Opis klastru — DeepSeek")

if not DEEPSEEK_KEY:
    st.warning("⚠️ Brak klucza API DeepSeek. Dodaj DEEPSEEK_API_KEY do zmiennych środowiskowych.")
    st.info("💡 Tip: Utwórz plik .env z DEEPSEEK_API_KEY=twój_klucz")
else:
    # Cache nazw grup na dłużej - 24h
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

Wymyśl kreatywną, zabawną nazwę dla tej grupy (max 2-3 słowa). 
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
        """Generuje opis klastra w stylu linuxiarza-trolla"""
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

Opisz tę grupę w stylu zapalonego linuxiarza-trolla. Używaj spolszczonego żargonu IT, 
baw się stereotypami, bądź lekko ironiczny i zabawny. Max 3 zdania. 
Pisz tak jakbyś gadał na IRCu czy forum linuxowym. Używaj polskich odpowiedników 
angielskich terminów. Nie bądź miły, ale nie przesadzaj z hejtem.
"""

            # Wywołanie API
            body = {
                "model": DEEPSEEK_MODEL,
                "stream": False,
                "temperature": 0.85,  # Wyższa temperatura dla kreatywności
                "max_tokens": 150,
                "messages": [
                    {"role": "system", "content": "Jesteś zapalonym linuxiarzem z poczuciem humoru. Mówisz spolszczonym żargonem IT, jesteś lekko trollujący ale w granicach rozsądku. Używasz wyrażeń jak 'noł łej', 'debić', 'apt-get install życie', 'kernel panic', 'RTFM' etc."},
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

    if st.button("🎯 Generuj opis grupy", type="primary"):
        with st.spinner("🔮 DeepSeek analizuje grupę..."):
            description = generate_cluster_description(selected_cluster, cluster_data, bin_present)
        
        # Wyświetl wynik
        if description.startswith(("❌", "⏰", "🔑", "💳", "🚫", "⚠️")):
            st.error(description)
        else:
            st.success("✅ Gotowe!")
            st.write(description)
            st.caption(f"🤖 Wygenerowano dla grupy {selected_cluster}")

# Tabelki pod opisem - dla tych co nie lubią czytać
raw_cols = [
    "hobby_movies","hobby_sport","hobby_art","hobby_other","hobby_video_games",
    "learning_pref_books","learning_pref_offline_courses","learning_pref_personal_projects",
    "learning_pref_teaching","learning_pref_teamwork","learning_pref_workshop","learning_pref_chatgpt",
    "motivation_challenges","motivation_career","motivation_creativity_and_innovation",
    "motivation_money_and_job","motivation_personal_growth","motivation_remote",
]
bin_cols = list(dict.fromkeys(raw_cols))
bin_present = [c for c in bin_cols if c in cluster_data.columns]

if bin_present:
    rows = []
    for c in bin_present:
        s = pd.to_numeric(cluster_data[c], errors="coerce")
        p = float((s == 1).mean() * 100) if s.notna().any() else 0.0
        rows.append((c, f"{p:.0f}%"))
    st.table(pd.DataFrame(rows, columns=["feature", "share_of_1"]))
else:
    st.info("Brak pól binarnych do podsumowania.")

st.markdown("---")
st.caption("Kto to czyta ten mieszka w piwnicy XD")
st.markdown("---")