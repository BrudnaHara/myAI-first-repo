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
clusters_available = sorted(df_clust["cluster"].unique().tolist())
default_idx = clusters_available.index(selected_cluster) if selected_cluster in clusters_available else 0
cluster_desc = st.selectbox("Wybierz grupę do opisania:", options=clusters_available, index=default_idx)

cluster_data = df_clust[df_clust["cluster"] == cluster_desc]
st.write(f"**Grupa {int(cluster_desc)}** — {len(cluster_data)} osób")

st.subheader("🔧 Preferencje i motywacje (udział 1 = TAK)")

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
if st.button("man demony-grupowania"):
    with st.expander("📖 MANUAL: demony-grupowania", expanded=True):
        st.markdown("""
        ### NAZWA
        **demony-grupowania** – demon grupowania użytkowników na podstawie podobieństwa cech

        ### SKŁADNIA
        `demony-grupowania [--algorithm=kmeans] [--cluster=5] [--verbose]`

        ### OPIS
        Demon działający w tle, który grupuje użytkowników na klastry używając niekontrolowanego uczenia maszynowego.  
        Działa jak usługa systemowa – nie wymaga interakcji użytkownika.

        ### OPCJE
        - `--algorithm`   wybór algorytmu (domyślnie: kmeans)  
        - `--cluster`     liczba klastrów (domyślnie: 5)  
        - `--verbose`     szczegółowe logi do `/var/log/nerdapp/grupowanie.log`

        ### PRZYKŁADY
        `demony-grupowania --cluster=5 --verbose`  
        Grupuje użytkowników na 5 klastrów z pokazywaniem logów.

        ### AUTOR
        NerdApp 1.0 – napisane przez BrudnaHara na Debianie
        """)
        st.markdown("""
<style>
    .stExpander {
        background-color: black;
        color: #00ff00;
        font-family: Monospace;
    }
</style>
""", unsafe_allow_html=True)

# ===== HYBRYDA LLM (Analiza vs Śmieszek) =====
st.header("🧠 Opis klastru — DeepSeek (hybryda)")

if not DEEPSEEK_KEY:
    st.info("Brak DEEPSEEK_API_KEY w środowisku.")
else:
    mode = st.radio("Tryb odpowiedzi", ["Analiza", "Śmieszek"], horizontal=True)

    # 🔥 TUTAJ ZMIENIASZ PROMPT
    if mode == "Analiza":
        SYS_PROMPT = (
            "Jesteś poważnym analitykiem danych. "
            "Podaj szczegółowy, techniczny opis klastra. "
            "Bez żartów. Po polsku. Max 10 zdań."
        )
        TEMPERATURE = 0.25   # 🔥 zmień tu temperaturę
        MAX_TOKENS = 500     # 🔥 zmień tu max tokeny
    else:
        SYS_PROMPT = (
            "Opisz grupę użytkowników w zabawny, ironiczny sposób. "
            "Po polsku. Max 2 zdania. "
            "Nie bądź poważny."
        )
        TEMPERATURE = 0.7    # 🔥 zmień tu temperaturę
        MAX_TOKENS = 150     # 🔥 zmień tu max tokeny

    preview_rows = []
    for c in bin_present:
        s = pd.to_numeric(cluster_data[c], errors="coerce")
        p = float((s == 1).mean() * 100) if s.notna().any() else 0.0
        preview_rows.append({"feature": c, "share_of_1_pct": round(p, 1)})
    preview_rows.sort(key=lambda r: r["share_of_1_pct"], reverse=True)
    payload_user = {
        "cluster_id": int(cluster_desc),
        "n_in_group": int(len(cluster_data)),
        "top_binary_features": preview_rows[:12],
    }

    if st.button("Generuj opis klastru (stream)"):
        body = {
            "model": DEEPSEEK_MODEL,
            "stream": True,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "messages": [
                {"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": json.dumps(payload_user, ensure_ascii=False)},
            ],
        }

        placeholder = st.empty()
        buf = []

        try:
            # 🔧 debug – sprawdzamy czy wchodzimy do bloku
            st.write("START STREAM")

            with requests.post(
                f"{DEEPSEEK_BASE}/chat/completions",
                headers={
                    "Authorization": f"Bearer {DEEPSEEK_KEY}",
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",   # ważne
                },
                json=body,
                stream=True,
                timeout=90,
            ) as r:
                r.raise_for_status()
                st.caption(f"TE={r.headers.get('Transfer-Encoding')} CT={r.headers.get('Content-Type')}")
                for line in r.iter_lines(chunk_size=1, decode_unicode=True):
                    if not line:
                        continue
                    if line.startswith("data: "):
                        chunk = line[len("data: "):].strip()
                        if chunk == "[DONE]":
                            break
                        try:
                            obj = json.loads(chunk)
                            delta = obj["choices"][0]["delta"].get("content", "")
                            if delta:
                                buf.append(delta)
                                placeholder.write("".join(buf) + "▌")
                        except Exception:
                            continue
            # finalny tekst
            placeholder.write("".join(buf))
        except requests.HTTPError as e:
            st.error(f"HTTP {e.response.status_code}: {e.response.text[:300]}")
        except Exception as e:
            st.error(f"Błąd: {e}")

# Easter egg
st.markdown("---")
with st.expander("🖥️ **Konsola pomocy (wpisz komendę)**"):
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
        st.success("Uruchamiam grepowanie nerdów...")
        st.write("🔍 Przełącz się na zakładkę 'grepuj nerdów' powyżej!")
    
    elif help_input.strip() == "exit":
        st.warning("Nie ma wyjścia z pomocą – to jest Streamlit, nie prawdziwy terminal! 😉")
    
    elif help_input.strip() != "":
        st.error(f"Komenda nieznana: '{help_input}'. Wpisz 'help' aby uzyskać pomoc.")
