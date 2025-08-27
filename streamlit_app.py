
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
from pathlib import Path
from collections import Counter

st.set_page_config(page_title="👥 Znajdź znajomych z kursu", layout="wide")
st.title("👥 Znajdź znajomych z kursu")

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

# ---------- Mapowania / przygotowanie ----------
# płeć binarna na żądanie: kobieta=1.0, mężczyzna=0.0
if "gender" in df.columns:
    df["gender_num"] = df["gender"].map({
        "female": 1.0, "woman": 1.0, "kobieta": 1.0,
        "male": 0.0, "man": 0.0, "mężczyzna": 0.0
    })
    df["gender_num"] = df["gender_num"].fillna(pd.to_numeric(df["gender"], errors="coerce"))

def age_bucket(x):
    try:
        a = float(x)
    except:
        return "unknown"
    bins = [(0,17,"≤17"),(18,24,"18–24"),(25,34,"25–34"),(35,44,"35–44"),(45,54,"45–54"),(55,120,"55+")]
    for lo,hi,label in bins:
        if lo <= a <= hi: return label
    return "unknown"

if "age" in df.columns:
    df["age_group"] = df["age"].apply(age_bucket)

# ---------- Sidebar: filtry ----------
st.sidebar.header("⚙️ Ustawienia")
k_choice = st.sidebar.slider("Liczba klastrów", 1, 10, 5, 1)

# Gender filter
gender_opts = []
if "gender_num" in df.columns:
    g_map = {"Kobieta (1.0)":1.0, "Mężczyzna (0.0)":0.0}
    gender_opts = st.sidebar.multiselect("Płeć (numerycznie)", options=list(g_map.keys()))

# Industry / fav_place
industry_filter = []
place_filter = []
if "industry" in df.columns:
    industry_filter = st.sidebar.multiselect("Branża (industry)", options=sorted(df["industry"].dropna().unique().tolist()))
if "fav_place" in df.columns:
    place_filter = st.sidebar.multiselect("Ulubione miejsce (fav_place)", options=sorted(df["fav_place"].dropna().unique().tolist()))

# Wiek
age_range = None
if "age" in df.columns and pd.api.types.is_numeric_dtype(pd.to_numeric(df["age"], errors="coerce")):
    a_min = int(pd.to_numeric(df["age"], errors="coerce").min(skipna=True))
    a_max = int(pd.to_numeric(df["age"], errors="coerce").max(skipna=True))
    if a_min == a_max:
        a_min = max(0, a_min-1); a_max = a_max+1
    age_range = st.sidebar.slider("Wiek [min, max]", a_min, a_max, (a_min, a_max))

# Zastosuj filtry
df_f = df.copy()
if gender_opts and "gender_num" in df_f.columns:
    df_f = df_f[df_f["gender_num"].isin([g_map[x] for x in gender_opts])]
if industry_filter and "industry" in df_f.columns:
    df_f = df_f[df_f["industry"].isin(industry_filter)]
if place_filter and "fav_place" in df_f.columns:
    df_f = df_f[df_f["fav_place"].isin(place_filter)]
if age_range and "age" in df_f.columns:
    ages = pd.to_numeric(df_f["age"], errors="coerce")
    df_f = df_f[(ages >= age_range[0]) & (ages <= age_range[1])]

if df_f.empty:
    st.info("Brak danych po filtrach. Zmień filtry.")
    st.stop()

# ---------- Klastrowanie ----------
@st.cache_resource
def prepare_clustering(data: pd.DataFrame, n_clusters: int = 5):
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_remove = [c for c in ["id"]]  # usuń znane ID jeżeli istnieją
    features = [c for c in numeric_cols if c not in cols_to_remove]
    if not features:
        raise ValueError("Brak kolumn numerycznych do klastrowania.")
    clean = data.dropna(subset=features).copy()
    if clean.empty:
        return None
    scaler = StandardScaler()
    X = scaler.fit_transform(clean[features].astype(float))
    n_samples = X.shape[0]
    k = max(1, min(n_clusters, n_samples))
    if k == 1:
        clusters = np.zeros(n_samples, dtype=int)
        kmeans = None
        centroids = np.array([X.mean(axis=0)])
    else:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X)
        centroids = kmeans.cluster_centers_
    return {"X":X, "features":features, "scaler":scaler, "clusters":clusters,
            "kept_idx":clean.index.to_list(), "k":k, "centroids":centroids, "kmeans":kmeans}

res = prepare_clustering(df_f, k_choice)
if res is None:
    st.info("Po czyszczeniu brak danych. Zmień filtry.")
    st.stop()

X = res["X"]; features = res["features"]; clusters = res["clusters"]
kept_idx = res["kept_idx"]; k_used = res["k"]; centroids = res["centroids"]

df_view = df_f.copy()
df_view["cluster"] = np.nan
df_view.loc[kept_idx, "cluster"] = clusters
df_clust = df_view.dropna(subset=["cluster"]).copy()
df_clust["cluster"] = df_clust["cluster"].astype(int)

st.sidebar.write(f"k={k_used} | próbki={X.shape[0]}")

# ---------- Wybór użytkownika ----------
st.sidebar.header("🔍 Znajdź swoją grupę")
user_index_options = df_clust.index.tolist()
def user_label(idx):
    age = df_clust.loc[idx, "age"] if "age" in df_clust.columns and pd.notna(df_clust.loc[idx, "age"]) else "Profile"
    return f"User {int(idx)} - {age}"

user_choice = st.sidebar.selectbox("Wybierz swój profil:", options=[user_label(i) for i in user_index_options])
user_id = int(user_choice.split()[1])

if user_id not in df_clust.index:
    st.info("Wybrany użytkownik nie istnieje po filtrach.")
    st.stop()

user_cluster = int(df_clust.loc[user_id, "cluster"])
same_cluster = df_clust[df_clust["cluster"] == user_cluster]

st.sidebar.success(f"Jesteś w grupie {user_cluster}")
st.sidebar.metric("Osób w twojej grupie", len(same_cluster))

# ---------- Sekcja główna ----------
st.header("👋 Twoja grupa znajomych")
st.write(f"Znaleziono {len(same_cluster)} osób podobnych do Ciebie!")
st.dataframe(same_cluster)

# Lista userów na dole z grupą wiekową
st.subheader("👣 Uczestnicy z Twojej grupy (z przedziałem wieku)")
if "age_group" in df_clust.columns:
    listing = same_cluster.copy()
    listing["age_group"] = listing["age_group"].fillna("unknown")
    st.dataframe(listing[[c for c in ["age","age_group","industry","fav_place","cluster"] if c in listing.columns]])
else:
    st.dataframe(same_cluster[[c for c in ["age","industry","fav_place","cluster"] if c in same_cluster.columns]])

# ---------- Charakterystyka grup ----------
st.header("📊 Charakterystyka grup")
clusters_available = sorted(df_clust["cluster"].unique().tolist())
default_idx = clusters_available.index(user_cluster) if user_cluster in clusters_available else 0
cluster_desc = st.selectbox("Wybierz grupę do opisania:", options=clusters_available, index=default_idx)
cluster_data = df_clust[df_clust["cluster"] == cluster_desc]
st.write(f"**Grupa {int(cluster_desc)}** — {len(cluster_data)} osób")

col1, col2 = st.columns(2)
with col1:
    if "gender_num" in cluster_data.columns:
        g_counts = cluster_data["gender_num"].value_counts(dropna=True)
        men = int(g_counts.get(0.0, 0))
        women = int(g_counts.get(1.0, 0))
        fig, ax = plt.subplots()
        ax.pie([men, women], labels=["Mężczyzna (0.0)", "Kobieta (1.0)"], autopct="%1.1f%%")
        ax.set_title(f"Płeć numerycznie — Grupa {int(cluster_desc)}")
        st.pyplot(fig)
    else:
        st.info("Brak kolumny 'gender'/'gender_num'.")

with col2:
    if "age" in cluster_data.columns and cluster_data["age"].notna().any():
        age_counts = cluster_data["age"].value_counts().sort_index()
        fig, ax = plt.subplots()
        ax.bar(age_counts.index.astype(str), age_counts.values)
        ax.set_title(f"Rozkład wieku — Grupa {int(cluster_desc)}")
        ax.tick_params(axis="x", rotation=45)
        st.pyplot(fig)
    else:
        st.info("Brak danych o wieku.")

# ---------- Rekomendacja: do której grupy najbliżej ----------
st.header("🧭 Do której grupy jest Ci najbliżej?")
def nearest_cluster_row(row: pd.Series):
    # zbuduj wektor użytkownika w tej samej przestrzeni cech co X
    try:
        vec = row[features].astype(float).to_numpy()
    except:
        return None, None
    # braków nie dopuszczamy
    if np.isnan(vec).any() or np.isinf(vec).any():
        return None, None
    # Standaryzacja jak przy trenowaniu
    scaler = res["scaler"]
    vec_s = scaler.transform(vec.reshape(1, -1))
    # odległość euklidesowa do centroidów
    dists = np.linalg.norm(centroids - vec_s, axis=1)
    best = int(np.argmin(dists))
    return best, float(dists[best])

best_cluster, best_dist = nearest_cluster_row(df_clust.loc[user_id])
if best_cluster is None:
    st.info("Brak pełnych danych numerycznych dla porównania.")
else:
    st.write(f"🔎 Najbliżej Ci do **grupy {best_cluster}** (wg odległości do centroidu).")

# ---------- Śmieszne podsumowanie ----------
def funny_summary(df_subset: pd.DataFrame) -> str:
    bits = []
    # top branża / miejsce
    if "industry" in df_subset.columns and not df_subset["industry"].dropna().empty:
        top_ind = df_subset["industry"].mode().iloc[0]
        bits.append(f"klan {top_ind.lower()}")
    if "fav_place" in df_subset.columns and not df_subset["fav_place"].dropna().empty:
        top_pl = df_subset["fav_place"].mode().iloc[0]
        bits.append(f"wyznawcy miejscówki „{top_pl}”")
    # płeć
    if "gender_num" in df_subset.columns:
        g = df_subset["gender_num"].round(0).value_counts()
        if g.get(1.0,0) > g.get(0.0,0):
            bits.append("prym wiodą kobiety")
        elif g.get(0.0,0) > g.get(1.0,0):
            bits.append("męska frakcja na prowadzeniu")
    # wiek
    if "age_group" in df_subset.columns and not df_subset["age_group"].dropna().empty:
        top_age = df_subset["age_group"].mode().iloc[0]
        bits.append(f"dominanta wiekowa: {top_age}")
    # fallback
    if not bits:
        bits = ["zjadacze tokenów na ChatGPT", "fascynaci kotów łażących po górach"]
    # składanie
    core = ", ".join(bits[:3])
    punch = [
        "najbliżej Ci do ekipy zjadaczy tokenów 🧠",
        "wygląda, że to plemię prompt wizardów 🔮",
        "statystyka szepcze: to Twoje klimaty 😎"
    ]
    return f"➡️ Podsumowanie: {core}. {np.random.choice(punch)}"

st.subheader("😼 TL;DR Twojej grupy")
target_df = same_cluster if not same_cluster.empty else df_clust
st.write(funny_summary(target_df))

st.markdown("---")
st.caption("Filtrowanie rozszerzone + rekomendacja najbliższej grupy + podsumowanie. Jeśli coś pęknie, to znaczy, że test był skuteczny XD")
