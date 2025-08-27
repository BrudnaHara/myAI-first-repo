import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Konfiguracja strony
st.set_page_config(page_title="Ankieta Welcome", layout="wide")
st.title("🎯 Eksploracja danych z ankiety powitalnej")

# Wczytanie danych
@st.cache_data
def load_data():
    return pd.read_csv("35__welcome_survey_cleaned.csv", sep=';')

df = load_data()

# Sidebar z filtrami
st.sidebar.header("Filtry")

# Filtry (POPRAWIONE - usuwamy nan)
gender_filter = st.sidebar.multiselect(
    "Płeć:",
    options=df['gender'].dropna().unique(),
    default=df['gender'].dropna().unique()
)

age_filter = st.sidebar.multiselect(
    "Przedział wiekowy:",
    options=df['age'].dropna().unique(),
    default=df['age'].dropna().unique()
)

industry_filter = st.sidebar.multiselect(
    "Branża:",
    options=df['industry'].dropna().unique(),
    default=df['industry'].dropna().unique()
)

# Filtrowanie danych
filtered_df = df[
    (df['gender'].isin(gender_filter)) &
    (df['age'].isin(age_filter)) &
    (df['industry'].isin(industry_filter))
]

# Podstawowe statystyki
st.header("Podsumowanie")
col1, col2, col3 = st.columns(3)
col1.metric("Liczba respondentów", len(filtered_df))
col2.metric("Liczba kobiet", len(filtered_df[filtered_df['gender'] == 1.0]))
col3.metric("Liczba mężczyzn", len(filtered_df[filtered_df['gender'] == 0.0]))

# Tabela z danymi
st.header("Dane")
st.dataframe(filtered_df.head(10))

# Wykresy
st.header("Wizualizacje")

col1, col2 = st.columns(2)

with col1:
    # Rozkład płci (POPRAWIONE)
    gender_map = {0.0: 'Mężczyźni', 1.0: 'Kobiety'}
    gender_counts = filtered_df['gender'].value_counts()
    gender_counts.index = gender_counts.index.map(gender_map)
    fig, ax = plt.subplots()
    ax.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%')
    ax.set_title("Rozkład płci")
    st.pyplot(fig)

with col2:
    # Rozkład hobby
    hobby_cols = [col for col in df.columns if 'hobby_' in col]
    hobby_counts = filtered_df[hobby_cols].sum().sort_values(ascending=False)
    fig, ax = plt.subplots()
    ax.bar(hobby_counts.index, hobby_counts.values)
    ax.set_title("Popularność hobby")
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

# Podgląd surowych danych
if st.checkbox("Pokaż surowe dane"):
    st.subheader("Surowe dane")
    st.write(filtered_df)
