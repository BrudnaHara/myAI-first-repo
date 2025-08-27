import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

# Konfiguracja strony
st.set_page_config(page_title="Znajdź znajomych z kursu", layout="wide")
st.title("👥 Znajdź znajomych z kursu")

# Wczytanie danych
@st.cache_data
def load_data():
    return pd.read_csv("35__welcome_survey_cleaned.csv", sep=';')

df = load_data()

# Przygotowanie danych do klastrowania
@st.cache_resource
def prepare_clustering(data):
    # Wybierz kolumny numeryczne do klastrowania
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    # Usuń kolumny które nie powinny być używane
    cols_to_remove = ['id']  # dostosuj jeśli masz kolumnę id
    features = [col for col in numeric_cols if col not in cols_to_remove]
    
    # Skalowanie danych
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[features])
    
    # Klastrowanie
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_data)
    
    return clusters, features, scaler, kmeans

# Dodaj klastry do danych
clusters, features, scaler, kmeans = prepare_clustering(df)
df['cluster'] = clusters

# Sidebar - wyszukiwanie znajomych
st.sidebar.header("🔍 Znajdź swoją grupę")

# Wybór użytkownika
user_options = [f"User {idx} - {df.loc[idx, 'age'] if 'age' in df.columns else 'Profile'}" 
                for idx in df.index]
user_choice = st.sidebar.selectbox("Wybierz swój profil:", options=user_options)

if user_choice:
    user_id = int(user_choice.split(" ")[1])  # Extract user index
    user_cluster = df.loc[user_id, 'cluster']
    same_cluster = df[df['cluster'] == user_cluster]
    
    st.sidebar.success(f"Jesteś w grupie {int(user_cluster)}")
    st.sidebar.metric("Osób w twojej grupie", len(same_cluster))

    # Podstawowe statystyki
    st.header("👋 Twoja grupa znajomych")
    st.write(f"Znaleziono {len(same_cluster)} osób podobnych do Ciebie!")
    st.dataframe(same_cluster)

# Opis grup klastrowych
st.header("📊 Charakterystyka grup")
cluster_desc = st.selectbox(
    "Wybierz grupę do opisania:",
    options=sorted(df['cluster'].unique())
)

if cluster_desc is not None:
    cluster_data = df[df['cluster'] == cluster_desc]
    st.write(f"**Grupa {int(cluster_desc)}** - {len(cluster_data)} osób")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'gender' in df.columns:
            gender_counts = cluster_data['gender'].value_counts()
            fig, ax = plt.subplots()
            ax.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%')
            ax.set_title(f"Rozkład płci - Grupa {int(cluster_desc)}")
            st.pyplot(fig)
    
    with col2:
        if 'age' in df.columns:
            age_counts = cluster_data['age'].value_counts()
            fig, ax = plt.subplots()
            ax.bar(age_counts.index, age_counts.values)
            ax.set_title(f"Rozkład wieku - Grupa {int(cluster_desc)}")
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig)

# Dodatkowe informacje o grupach
st.header("🎯 Co łączy Twoją grupę?")
if user_choice:
    st.write("**Twoja grupa charakteryzuje się:**")
    cluster_mean = same_cluster.mean(numeric_only=True)
    overall_mean = df.mean(numeric_only=True)
    
    # Znajdź cechy które najbardziej różnią grupę od średniej
    differences = (cluster_mean - overall_mean).abs().sort_values(ascending=False)
    top_features = differences.head(3).index.tolist()
    
    for feature in top_features:
        your_value = cluster_mean[feature]
        avg_value = overall_mean[feature]
        st.write(f"- **{feature}**: {your_value:.2f} (średnia: {avg_value:.2f})")

# Footer
st.markdown("---")
st.caption("Aplikacja do znajdowania znajomych na kursie data science | Moduł 7")
