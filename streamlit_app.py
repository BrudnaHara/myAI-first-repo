import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
from pathlib import Path

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
    st.warning("Nie znaleziono pliku `35__welcome_survey_cleaned.csv`. Wgraj CSV poniżej (separator ;).")
    up = st.file_uploader("Wgraj plik CSV", type=["csv"])
    if up:
        df = pd.read_csv(up, sep=";")
    else:
        st.stop()

# ---------- Mapowanie płci ----------
if "gender" in df.columns:
    df["gender_num"] = df["gender"].map({"female": 1.0, "woman": 1.0, "kobieta": 1.0,
                                         "male": 0.0, "man": 0.0, "mężczyzna": 0.0})
    # w razie gdyby kolumna gender miała już liczby
    df["gender_num"] = df["gender_num"].fillna(df["gender"])

# ---------- Funkcje ----------
@st.cache_resource
def prepare_clustering(data: pd.DataFrame, n_clusters: int = 5):
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_remove = [c for c in ["id"] if c in numeric_cols]
    features = [c for c in numeric_cols if c not in cols_to_remove]

    clean = data.copy().dropna(subset=features)
    if clean.empty:
        raise ValueError("Brak danych numerycznych po czyszczeniu.")

    scaler = StandardScaler()
    X = scaler.fit_transform(clean[features].astype(float))

    k = max(2, min(n_clusters, X.shape[0]))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)

    return clusters, features, scaler, kmeans, clean.index.to_list()

# ---------- Sidebar ----------
st.sidebar.header("⚙️ Ustawienia")
k_choice = st.sidebar.slider("Liczba klastrów", min_value=2, max_value=10, value=5, step=1)

# Filtry dodatkowe
if "industry" in df.columns:
    industry_filter = st.sidebar.multiselect("Filtruj po branży:", options=df["industry"].dropna().unique())
    if industry_filter:
        df = df[df["industry"].isin(industry_filter)]

if "fav_place" in df.columns:
    place_filter = st.sidebar.multiselect("Filtruj po ulubionym miejscu:", options=df["fav_place"].dropna().unique())
    if place_filter:
        df = df[df["fav_place"].isin(place_filter)]

# ---------- Klastrowanie ----------
clusters, features, scaler, kmeans, kept_idx = prepare_clustering(df, k_choice)
df = df.copy()
df["cluster"] = np.nan
df.loc[kept_idx, "cluster"] = clusters

df_clust = df.dropna(subset=["cluster"]).copy()
df_clust["cluster"] = df_clust["cluster"].astype(int)

# ---------- Wybór użytkownika ----------
st.sidebar.header("🔍 Znajdź swoją grupę")
user_options = [
    f"User {int(idx)} - {df.loc[idx, 'age'] if 'age' in df.columns and pd.notna(df.loc[idx, 'age']) else 'Profile'}"
    for idx in df_clust.index
]
if len(user_options) == 0:
    st.error("Brak rekordów po przygotowaniu danych.")
    st.stop()

user_choice = st.sidebar.selectbox("Wybierz swój profil:", options=user_options)
user_id = int(user_choice.split()[1])
user_cluster = int(df_clust.loc[user_id, "cluster"])
same_cluster = df_clust[df_clust["cluster"] == user_cluster]

st.sidebar.success(f"Jesteś w grupie {user_cluster}")
st.sidebar.metric("Osób w twojej grupie", len(same_cluster))

# ---------- Sekcja główna ----------
st.header("👋 Twoja grupa znajomych")
st.write(f"Znaleziono {len(same_cluster)} osób podobnych do Ciebie!")
st.dataframe(same_cluster)

st.header("📊 Charakterystyka grup")
clusters_available = sorted(df_clust["cluster"].unique().tolist())
cluster_desc = st.selectbox("Wybierz grupę do opisania:", options=clusters_available, index=clusters_available.index(user_cluster))

cluster_data = df_clust[df_clust["cluster"] == cluster_desc]
st.write(f"**Grupa {int(cluster_desc)}** — {len(cluster_data)} osób")

col1, col2 = st.columns(2)

with col1:
    if "gender_num" in cluster_data.columns:
        g_counts = cluster_data["gender_num"].value_counts()
        if not g_counts.empty:
            labels = ["Mężczyzna (0.0)", "Kobieta (1.0)"]
            values = [g_counts.get(0.0, 0), g_counts.get(1.0, 0)]
            fig, ax = plt.subplots()
            ax.pie(values, labels=labels, autopct="%1.1f%%")
            ax.set_title(f"Płeć numerycznie — Grupa {int(cluster_desc)}")
            st.pyplot(fig)

with col2:
    if "age" in cluster_data.columns:
        age_counts = cluster_data["age"].value_counts().sort_index()
        if len(age_counts) > 0:
            fig, ax = plt.subplots()
            ax.bar(age_counts.index.astype(str), age_counts.values)
            ax.set_title(f"Rozkład wieku — Grupa {int(cluster_desc)}")
            ax.tick_params(axis="x", rotation=45)
            st.pyplot(fig)

# ---------- Różnice cech ----------
st.header("🎯 Co łączy Twoją grupę?")
cluster_mean = same_cluster.mean(numeric_only=True)
overall_mean = df_clust.mean(numeric_only=True)
if not cluster_mean.empty and not overall_mean.empty:
    differences = (cluster_mean - overall_mean).abs().sort_values(ascending=False)
    top_features = differences.head(3).index.tolist()
    if len(top_features) > 0:
        st.write("**Twoja grupa wyróżnia się w:**")
        for feature in top_features:
            your_value = cluster_mean[feature]
            avg_value = overall_mean[feature]
            st.write(f"- **{feature}**: {your_value:.2f} (średnia ogólna: {avg_value:.2f})")

st.markdown("---")
st.caption("Znajdowanie znajomych | teraz z płcią numeryczną i filtrami branży i ulubionego miejsca 🚀")


