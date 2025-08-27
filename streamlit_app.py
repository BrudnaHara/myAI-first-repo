
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
df_f = multiselect_filter(df_f, "gender", "Płeć")  # tekstowe, nie numeryczne
df_f = multiselect_filter(df_f, "fav_animals", "Ulubione zwierzę")
df_f = multiselect_filter(df_f, "city", "Miasto")  # jeśli masz

# binarne 0/1 → tak/nie
with st.sidebar.expander("Preferencje i motywacje"):
    binary_cols = ["hobby_movies", "hobby_sport", "learning_pref_chatgpt", "motivation_challenges"]
    for col in binary_cols:
        if col in df_f.columns:
            s = pd.to_numeric(df_f[col], errors="coerce")  # akceptuje 0/1 i "0"/"1"
            choice = st.sidebar.radio(col, ["Wszystko", "tak", "nie"], index=0, horizontal=True)
            if choice != "Wszystko":
                want = 1 if choice == "tak" else 0
                df_f = df_f[s == want]

# brak wyników po filtrach
if df_f.empty:
    st.write("you weirdo as fuck XD")
    st.stop()


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
# ---------- Preprocessing: age → liczba, kategorie → one-hot ----------
df_enc = df_f.copy()

# age: mapuj przedziały na środki (dopasuj do swoich etykiet)
if "age" in df_enc.columns:
    age_map = {
        "<18": 16, "18-24": 21, "18–24": 21,
        "25-34": 29.5, "35-44": 39.5, "45-54": 49.5,
        "55-64": 59.5, ">=65": 70, "unknown": np.nan
    }
    df_enc["age_num"] = df_enc["age"].map(age_map)

# kolumny kategoryczne do one-hot (użyj tylko tych, które masz)
cat_cols_cfg = ["edu_level", "fav_place", "gender", "industry", "city", "fav_animals"]
cat_cols = [c for c in cat_cols_cfg if c in df_enc.columns]

dummies = pd.get_dummies(df_enc[cat_cols], dummy_na=False) if cat_cols else pd.DataFrame(index=df_enc.index)


# numeryczne bazowe: wszystkie 0/1 (hobby_*, learning_*, motivation_*, + age_num)
df_enc = df_enc.apply(lambda s: pd.to_numeric(s, errors="ignore"))
num_base = df_enc.select_dtypes(include=[np.number])
# dołącz age_num, jeśli jeszcze nie weszło
if "age_num" in df_enc.columns and "age_num" not in num_base.columns:
    num_base = pd.concat([num_base, df_enc[["age_num"]]], axis=1)

# finalny numeric dataframe do klastrowania
df_num = pd.concat([num_base, dummies], axis=1)
df_num = df_num.dropna(axis=1, how="all")
df_num = df_num.loc[:, df_num.nunique() > 1]

res = prepare_clustering(df_num, k_choice)
if res is None:
    st.header("👋 Nerdy jak Ty XD")
    st.write("you weirdo as fuck XD")
    st.stop()

# wstrzyknij etykiety do tabeli
df_view = df_f.copy()
df_view["cluster"] = np.nan
df_view.loc[res["kept_idx"], "cluster"] = res["clusters"]
df_clust = df_view.dropna(subset=["cluster"]).copy()
df_clust["cluster"] = df_clust["cluster"].astype(int)
if df_clust.empty:
    st.header("👋 Nerdy jak Ty XD")
    st.write("you weirdo as fuck XD")
    st.stop()


# ---------- Auto-wybór grupy ----------
mode_series = df_clust["cluster"].mode()
if mode_series.empty:
    st.header("👋 Nerdy jak Ty XD")
    st.write("you weirdo as fuck XD")
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

# KOL_1: kategorie tekstowe (więcej pól)
cat_cols_cfg = ["industry", "fav_place", "edu_level", "gender", "fav_animals", "city"]
present_cat = [c for c in cat_cols_cfg if c in cluster_data.columns and cluster_data[c].notna().any()]

if not present_cat:
    st.info("Brak danych kategorycznych do pokazania.")
else:
    cols = st.columns(2)
    for i, c in enumerate(present_cat):
        with cols[i % 2]:
            vc = cluster_data[c].value_counts()
            if len(vc) == 0:
                continue
            fig, ax = plt.subplots()
            if len(vc) <= 8:
                ax.pie(vc.values, labels=vc.index.astype(str), autopct="%1.1f%%")
            else:
                top = vc.head(10)
                ax.bar(top.index.astype(str), top.values)
                ax.tick_params(axis="x", rotation=45)
            ax.set_title(f"{c} — Grupa {int(cluster_desc)}")
            st.pyplot(fig)

# KOL_2: podsumowanie pól binarnych 0/1
st.subheader("🔧 Preferencje i motywacje (udział TAK)")
bin_cols = ["hobby_movies", "hobby_sport", "learning_pref_chatgpt", "motivation_challenges"]
bin_present = [c for c in bin_cols if c in cluster_data.columns]

if bin_present:
    summary = []
    for c in bin_present:
        s = pd.to_numeric(cluster_data[c], errors="coerce")
        p = float((s == 1).mean()*100) if s.notna().any() else 0.0
        summary.append((c, f"{p:.0f}%"))
    st.table(pd.DataFrame(summary, columns=["cecha", "TAK"]))
else:
    st.info("Brak pól binarnych do podsumowania.")

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
st.caption("Kto to czyta ten mieszka w piwnicy XD")
