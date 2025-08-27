
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
from pathlib import Path
from collections import Counter

st.set_page_config(page_title="👥 Znajdź nerdów jak ty", layout="wide")
st.title("👥 Znajdź nerdów jak ty")

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

# ---------- Sidebar: filtry ----------
st.sidebar.header("⚙️ Ustawienia")
k_choice = st.sidebar.slider("Liczba klastrów", 1, 10, 5, 1)

# Gender filter
gender_opts = []
if "gender_num" in df.columns:
    g_map = {"Kobieta (1.0)":1.0, "Mężczyzna (0.0)":0.0}
    gender_opts = st.sidebar.multiselect("Płeć (numerycznie)", options=list(g_map.keys()))

def multiselect_filter(df, col, label=None):
    if col in df.columns:
        opts = sorted(df[col].dropna().unique().tolist())
        sel = st.sidebar.multiselect(label or col, options=opts)
        if sel:
            return df[df[col].isin(sel)]
    return df
df_f = df.copy()
df_f = multiselect_filter(df_f, "industry", "Branża")
df_f = multiselect_filter(df_f, "fav_place", "Ulubione miejsce")

# ---------- Filtry binarne (0/1 → tak/nie w UI, filtrowanie po liczbie) ----------
binary_cols = ["hobby_movies", "hobby_sport", "learning_pref_chatgpt", "motivation_challenges"]

for col in binary_cols:
    if col in df_f.columns:
        s = pd.to_numeric(df_f[col], errors="coerce")  # działa dla 0,1 oraz "0","1"
        choice = st.sidebar.radio(f"{col}", ["Wszystko", "tak", "nie"], index=0)
        if choice != "Wszystko":
            want = 1 if choice == "tak" else 0
            df_f = df_f[s == want]


# ---------- Klastrowanie (wariant minimalny) ----------
@st.cache_resource
def prepare_clustering(data: pd.DataFrame, n_clusters: int = 5):
    # tylko cechy numeryczne (bez 'id' jeśli istnieje)
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in numeric_cols if c != "id"]
    if not features:
        return None
    # usuń wiersze z brakami w używanych cechach
    clean = data.dropna(subset=features).copy()
    if clean.empty:
        return None

    # standaryzacja
    scaler = StandardScaler()
    X = scaler.fit_transform(clean[features].astype(float))

    # k nie może przekraczać liczby próbek
    k = max(1, min(n_clusters, X.shape[0]))
    if k == 1:
        clusters = np.zeros(X.shape[0], dtype=int)
    else:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X)

    return {"clusters": clusters, "features": features, "kept_idx": clean.index.to_list(), "k": k}

res = prepare_clustering(df_f, k_choice)
if res is None:
    st.info("Brak danych do klastrowania po filtrach lub brak cech numerycznych.")
    st.stop()

# wstrzyknij etykiety do tabeli
df_view = df_f.copy()
df_view["cluster"] = np.nan
df_view.loc[res["kept_idx"], "cluster"] = res["clusters"]
df_clust = df_view.dropna(subset=["cluster"]).copy()
df_clust["cluster"] = df_clust["cluster"].astype(int)

# ---------- Auto-wybór grupy ----------
mode_series = df_clust["cluster"].mode()
if mode_series.empty:
    st.info("Brak przypisanych klastrów po filtrach.")
    st.stop()

selected_cluster = int(mode_series.iloc[0])
same_cluster = df_clust[df_clust["cluster"] == selected_cluster]

st.sidebar.header("📌 Wybrana grupa")
st.sidebar.write(f"Automatycznie wybrano **grupę {selected_cluster}**.")
st.sidebar.metric("Liczba osób w grupie", len(same_cluster))


# ---------- Sekcja główna ----------
st.header("👋 Nerdy jak Ty XD")
if same_cluster.empty:
    st.write("you weirdo as fuck XD")
else:
    st.write(f"Znaleziono {len(same_cluster)} osób podobnych do Ciebie!")
    st.dataframe(same_cluster)

# ---------- Charakterystyka grup ----------
st.header("📊 Charakterystyka grup")
clusters_available = sorted(df_clust["cluster"].unique().tolist())
default_idx = clusters_available.index(selected_cluster) if selected_cluster in clusters_available else 0
cluster_desc = st.selectbox("Wybierz grupę do opisania:", options=clusters_available, index=default_idx)

cluster_data = df_clust[df_clust["cluster"] == cluster_desc]
st.write(f"**Grupa {int(cluster_desc)}** — {len(cluster_data)} osób")

col1, col2 = st.columns(2)
with col1:
    if "industry" in cluster_data.columns and cluster_data["industry"].notna().any():
        ind_counts = cluster_data["industry"].value_counts()
        fig, ax = plt.subplots()
        ax.pie(ind_counts.values, labels=ind_counts.index, autopct="%1.1f%%")
        ax.set_title(f"Branże — Grupa {int(cluster_desc)}")
        st.pyplot(fig)
    else:
        st.info("Brak kolumny 'industry' lub danych.")

with col2:
    if "fav_place" in cluster_data.columns and cluster_data["fav_place"].notna().any():
        place_counts = cluster_data["fav_place"].value_counts().head(10)
        fig, ax = plt.subplots()
        ax.bar(place_counts.index.astype(str), place_counts.values)
        ax.set_title(f"Ulubione miejsca — Grupa {int(cluster_desc)}")
        ax.tick_params(axis="x", rotation=45)
        st.pyplot(fig)
    else:
        st.info("Brak kolumny 'fav_place' lub danych.")

# ---------- Śmieszne podsumowanie ----------
def funny_summary(df_subset: pd.DataFrame) -> str:
    bits = []
    if "industry" in df_subset.columns and not df_subset["industry"].dropna().empty:
        top_ind = df_subset["industry"].mode().iloc[0]
        bits.append(f"klan {top_ind.lower()}")
    if "fav_place" in df_subset.columns and not df_subset["fav_place"].dropna().empty:
        top_pl = df_subset["fav_place"].mode().iloc[0]
        bits.append(f"wyznawcy miejscówki „{top_pl}”")
    if not bits:
        bits = ["zjadacze tokenów na ChatGPT", "fascynaci kotów łażących po górach"]
    core = ", ".join(bits[:3])
    punch = [
        "najbliżej Ci do ekipy zjadaczy tokenów 🧠",
        "wygląda, że to plemię prompt wizardów 🔮",
        "statystyka szepcze: to Twoje klimaty 😎",
    ]
    return f"➡️ Podsumowanie: {core}. {np.random.choice(punch)}"

st.subheader("😼 TL;DR Twojej grupy")
target_df = same_cluster if not same_cluster.empty else df_clust
st.write(funny_summary(target_df))

st.markdown("---")
st.caption("Charakterystyka: industry + fav_place. Sekcja nearest-cluster usunięta w wariancie minimalnym.")
