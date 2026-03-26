import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
movies = pd.read_csv("Movies_dataset.csv")
movies = movies[['title', 'overview']]
movies.dropna(inplace=True)

# TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['overview'])

# Similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Index
movies['title'] = movies['title'].str.strip()
movies['title_clean'] = movies['title'].str.lower().str.strip()
indices = pd.Series(movies.index, index=movies['title_clean']).drop_duplicates()

# Function
def recommend(movie_title, n=5):
    movie_title = movie_title.lower().strip()   # FIXED
    
    if movie_title not in indices:
        return ["Movie not found"]
    
    idx = indices[movie_title]

    if isinstance(idx, pd.Series):
        idx = idx.iloc[0]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:n+1]
    movie_indices = [i[0] for i in sim_scores]

    return movies['title'].iloc[movie_indices].tolist()

# UI
st.title("🎬 Movie Recommendation System")

movie = st.text_input("Enter a movie name")

if st.button("Recommend"):
    result = recommend(movie)
    for m in result:
        st.write(m)