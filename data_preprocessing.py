import pandas as pd
import numpy as np


def preprocess(movies_path: str, raings_path: str) -> [pd.DataFrame, pd.DataFrame]:
    """Preprocesses data and returns data frames"""
    ratings = pd.read_csv(raings_path).drop('timestamp', axis=1)
    movies = pd.read_csv(movies_path)
    movies['id'] = movies['id'].apply(clear_id).astype(int)
    movies = movies[movies['id'] != -1]

    return movies, ratings

def clear_id(movie_id):
    try:
        movie_id = int(movie_id)
        return movie_id
    except:
        return - 1
