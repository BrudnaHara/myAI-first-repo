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
    st.warning("Nie znaleziono pliku `35__welcome_survey_cleaned.csv`. Wgraj CSV (separator ;).")
    up = st.file_uploader("Wgraj plik CSV", type=["csv"])
    if up:
        df = pd.read_csv(up, sep=";")
    else:
        st.stop()

# ---------- Mapowanie płci ----------
if "gender" in df.columns:
    df["gender_num"] = df["gender"].map({
        "female": 1.0, "woman": 1.0, "kobieta": 1.0,
        "male": 0.0, "man": 0.0, "mężczyzna": 0.0
    })
    # jeżeli już liczby – zostaw
    df["gender_num"] = df["gender_num"].fillna(pd.to_numeric(df["gender"], errors="coerce"))

# ---------- Sidebar: ustawienia i filtry ----------
st.sidebar.header("⚙️ Ustawienia")
k_choice = st.sidebar.slider("Liczba klastrów", min_value=1, max_value=10, value=5, step=1)

if "industry" in df.columns:
    ind_opts = sorted(df["industry"].dropna().unique().tolist())
    industry_filter = st.sidebar.multiselect("Filtr: branża (industry)", options=ind_opts)
else:
    industry_filter = []

if "fav_place" in df.columns:
    place_opts = sorted(df["fav_place"].dropna().unique().tolist())
    place_filter = st.sidebar.multiselect("Filtr: ulubione miejsce (fav_place)", options=place_opts)
else:
    place_filter = []

# zastosuj filtry
df_f = df.copy()
if industry_filter:
    df_f = df_f[df_f["industry"].isin(industry_filter)]
if place_filter:
    df_f = df_f[df_f["fav_place"].isin(place_filter)]

if df_f.empty:
    st.info("Brak danych po zastosowaniu filtrów. Zmień filtry.")
    st.stop()

# ---------- Klastrowanie ----------
@st.cache_resource
def prepare_clustering(data: pd.DataFrame, n_clusters: int = 5):
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_remove = [c for c in ["id"] if c in numeric_cols]
    features = [c for c in numeric_cols if c not in cols_to_remove]

    if not features:
        raise ValueError("Brak kolumn numerycznych do klastrowania.")

    clean = data.dropna(subset=features).copy()
    n_samples = len(clean)
    if n_samples == 0:
        return None  # sygnał: brak próbek po czyszczeniu

    scaler = StandardScaler()
    X = scaler.fit_transform(clean[features].astype(float))

    # dopasuj k do liczby próbek
    k = max(1, min(n_clusters, n_samples))
    if k == 1:
        clusters = np.zeros(n_samples, dtype=int)
        kmeans = None
    else:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X)

    return {
        "clusters": clusters,
        "features": features,
        "scaler": scaler,
        "kmeans": kmeans,
        "kept_idx": clean.index.to_list(),
        "k_used": k,
        "n_samples": n_samples
    }

res = prepare_clustering(df_f, k_choice)
if res is None:
    st.info("Po usunięciu braków w danych nie zostały żadne wiersze. Zmień filtry lub uzupełnij dane.")
    st.stop()

clusters = res["clusters"]
kept_idx = res["kept_idx"]
k_used = res["k_used"]

df_view = df_f.copy()
df_view["cluster"] = np.nan
df_view.loc[kept_idx, "cluster"] = clusters
df_clust = df_view.dropna(subset=["cluster"]).copy()
df_clust["cluster"] = df_clust["cluster"].astype(int)

st.sidebar.write(f"Użyto klastrów: **k={k_used}** | próbek: **{res['n_samples']}**")

# ---------- Wybór użytkownika ----------
st.sidebar.header("🔍 Znajdź swoją grupę")
user_index_options = df_clust.index.tolist()
if not user_index_options:
    st.info("Brak rekordów z przypisanym klastrem po filtrach.")
    st.stop()

def user_label(idx):
    age = df_clust.loc[idx, "age"] if "age" in df_clust.columns and pd.notna(df_clust.loc[idx, "age"]) else "Profile"
    return f"User {int(idx)} - {age}"

user_choice = st.sidebar.selectbox("Wybierz swój profil:", options=[user_label(i) for i in user_index_options])
user_id = int(user_choice.split()[1])
user_cluster = int(df_clust.loc[user_id, "cluster"])
same_cluster = df_clust[df_clust["cluster"] == user_cluster]

st.sidebar.success(f"Jesteś w grupie {user_cluster}")
st.sidebar.metric("Osób w twojej grupie", len(same_cluster))

# ---------- Sekcja główna ----------
st.header("👋 Twoja grupa znajomych")
st.write(f"Znaleziono {len(same_cluster)} osób podobnych do Ciebie!")
st.dataframe(same_cluster if not same_cluster.empty else pd.DataFrame({"info": ["Brak osób w tej grupie po filtrach."]}))

st.header("📊 Charakterystyka grup")
clusters_available = sorted(df_clust["cluster"].unique().tolist())
# wybór grupy do opisu: domyślnie grupa usera, ale bezpiecznie, gdy jej nie ma
default_idx = clusters_available.index(user_cluster) if user_cluster in clusters_available else 0
cluster_desc = st.selectbox("Wybierz grupę do opisania:", options=clusters_available, index=default_idx)

cluster_data = df_clust[df_clust["cluster"] == cluster_desc]
st.write(f"**Grupa {int(cluster_desc)}** — {len(cluster_data)} osób")

col1, col2 = st.columns(2)

with col1:
    if "gender_num" in cluster_data.columns:
        g_counts = cluster_data["gender_num"].value_counts(dropna=True)
        # bezpieczne zera
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

# ---------- Różnice cech ----------
st.header("🎯 Co łączy Twoją grupę?")
cluster_mean = same_cluster.mean(numeric_only=True)
overall_mean = df_clust.mean(numeric_only=True)
if not cluster_mean.empty and not overall_mean.empty:
    differences = (cluster_mean - overall_mean).abs().sort_values(ascending=False)
    top_features = differences.head(3).index.tolist()
    if top_features:
        st.write("**Twoja grupa wyróżnia się w:**")
        for feature in top_features:
            your_value = cluster_mean[feature]
            avg_value = overall_mean[feature]
            st.write(f"- **{feature}**: {your_value:.2f} (średnia ogólna: {avg_value:.2f})")
    else:
        st.info("Brak wyróżniających się cech numerycznych.")
else:
    st.info("Za mało danych numerycznych do porównań.")

st.markdown("---")
st.caption("Stabilna wersja: odporna na puste filtry, k ≤ liczba próbek. Jeśli 0 rekordów, to 0 — bez crasha. XD")
