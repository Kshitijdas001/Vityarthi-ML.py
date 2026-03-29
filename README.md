
# Movie Recommendation System 🎥

This is a simple Movie Recommendation System built using Python, Machine Learning, and Streamlit. It suggests movies similar to the one entered by the user.

## Features
- Recommends movies based on content similarity  
- Uses TF-IDF vectorization and cosine similarity  
- Combines overview and genres for better accuracy  
- Simple and interactive UI using Streamlit

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Streamlit

##  Dataset
The project uses a movie dataset (`Movies_dataset.csv`) containing:
- Title
- Overview

## How It Works
- Combines movie overview and genres
- Converts text into numerical vectors using TF-IDF
- Calculates similarity using cosine similarity
- Recommends top similar movies


## How to Run

1. Install dependencies:
```bash
pip install pandas
pip install scikit-learn
pip install streamlit

# What to write in terminal after running the code
streamlit run main.py



