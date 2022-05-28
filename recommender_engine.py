import pandas as pd
import numpy as np
import json
import random


class MovieRecommender:
    def __init__(self, movies: pd.DataFrame, ratings: pd.DataFrame):
        self.movies = movies
        self.ratings = ratings
        self.ratings_matrix = self.get_ratings_matrix(self.ratings)
        self.VT = self.SVD(self.ratings_matrix)[2][:300, :]
        self.users_N, self.movies_N = self.ratings_matrix.shape
        self.available = set(movies['id'])
        self.slope = 1
        self.intercept = 0.8
        self.movieId_map, self.movieId_reversed_map = self._get_movie_id_mappers(movies, ratings)

    def get_ratings_matrix(self, ratings: pd.DataFrame):
        """Returns a matrix representation of users' ratings"""
        ratings_matrix = ratings.pivot(index = 'userId',
                                       columns ='movieId',
                                       values = 'rating').fillna(0)
        ratings_matrix = np.matrix(ratings_matrix)
    
        return ratings_matrix

    def already_seen(self, user_id: int) -> list:
        """Returns list with movies id which current user already seen"""
        result_df = self.ratings.loc[self.ratings['userId'] == user_id]
        lst = result_df['movieId'].tolist()
        return list(map(lambda movie: self.movieId_reversed_map[movie], lst))

    def highest_rated(self, n: int, user_id: int) -> list:
        """Returns list with n movies which user hasn't seen and
        for which predicted ratings are the highest"""
        already_seen = self.already_seen(user_id)
        similar = []
    
        for movie in already_seen:
            similar.append(self.cosine_similarity(movie))

        similar = [x for sublist in similar for x in sublist]
        similar = filter(lambda el: el[1] not in already_seen, similar)
        rated = []

        for movie in similar: 
            predicted = self.predict_rating(movie[2], user_id, movie[0])
            rated.append((movie[1], predicted))
    
        result = list(dict(sorted(rated, key=lambda v: v[1])).items())
        return sorted(result, key=lambda el: el[1], reverse=True)

    def SVD(self, A: np.matrix):
        """Returns U, S, V decomposition for matrix A"""
        return np.linalg.svd(A, full_matrices=False)

    def cosine_similarity(self, movie_id: int) -> list:
        """Returns list of tuples (index, similarity) in descending
        order by similarity for given movie"""
        movie = self.VT[:, movie_id].T
    
        similarities = (np.dot(movie, self.VT) / (np.linalg.norm(movie) *
                        np.apply_along_axis(np.linalg.norm, 0, self.VT)))
        similarities = np.array(similarities).flatten()
        similarities = [(movie_id, i, score) for i, score in enumerate(similarities)]
    
        return sorted(similarities, key=lambda el: el[2], reverse=True)

    def _two_movie_similarity(self, movie_id1: int, movie_id2: int) -> float:
        """Calculates the cosine similarity between two film vectors"""
        
        mov1_data = self.VT[:, movie_id1].T
        mov2_data = self.VT[:, movie_id2]
        
        dot_product = np.dot(mov1_data, mov2_data)[0,0]
        
        norm_1 = np.linalg.norm(mov1_data)
        norm_2 = np.linalg.norm(mov2_data)

        similarity = dot_product/(norm_1 * norm_2)

        return abs(similarity)

    def top_recommendations(self, user_id: int, n: int) -> list:
        """Returns n most recommended movies for the given user"""
        recommended_movies = list(map(lambda el: el[0], self.highest_rated(n, user_id)))
        description = f'Top {n} recommended movies for user {user_id}\n\n'
        i = 1

        for movie in recommended_movies:
            movie = self.movieId_map[movie]
            if movie in self.available:
                desc = self.movie_description(movie)
                description += f'{i}. ' + desc + '\n'
                i += 1
            
            if i == n + 1:
                break

        return description

    def movie_description(self, movie_id: int) -> str:
        """Returns full description about the movie"""
        infos = [('Title', self._get_title),
                ('Genre', self._get_genre),
                ('Overview', self._get_overview),
                ('Average Rate', self._get_avg_rate)]
        description = ''

        for title, info in infos:
            description += title + ': ' + info(movie_id) + '\n'

        return description

    def _choose_people(self, people_num: int) -> list:
      """Returns n randomnly choosen people from the user list"""
      random.seed(10)
      return random.choices([i for i in range(self.ratings_matrix.shape[0])],
                            k=people_num)

    def _get_title(self, movie_id: int) -> str:
        """Returns the title of the movie"""
        return self.movies[self.movies['id'] == movie_id]['title'].iloc[0]

    def _get_genre(self, movie_id: int) -> str:
        """Returns the genre of the movie"""
        genres = self.movies[self.movies['id'] == movie_id]['genres'].iloc[0].replace("'", "\"")
        genres = json.loads(genres)
        genre_desc = ''

        for genre in genres:
            genre_desc += genre['name'] + ', '

        return genre_desc[:-2]

    def _get_overview(self, movie_id: int) -> str:
        """Returns the overview of the movie"""
        return str(self.movies[self.movies['id'] == movie_id]['overview'].iloc[0])

    def _get_avg_rate(self, movie_id: int) -> str:
        """Returns the average rate of the movie"""
        return str(self.movies[self.movies['id'] == movie_id]['vote_average'].iloc[0])

    def _get_movie_id_mappers(self, movies: pd.DataFrame, ratings: pd.DataFrame):
        """Returns maps matrix id -> movie id and vice-versa"""
        movieId_map = {i: el for i, el in enumerate(sorted(ratings['movieId'].unique()))}
        movieId_reversed_map = {el: i for i, el in enumerate(sorted(ratings['movieId'].unique()))}

        return movieId_map, movieId_reversed_map

    def evaluate(self, n: int) -> float:
        """Returns root mean square error of the algorithm"""
        y_hat, y = self._get_test_data(n)

        return np.sqrt(np.sum((y_hat - y)**2) / len(y))

    def _get_test_data(self, n: int):
        """Returns matrix where some known movies are deleted
        and matrix with scores for this movie"""
        random.seed(10)

        y, y_hat = [], []
        users = self._choose_people(n)
        
        for user in users:
            seen_movies = self.already_seen(user)
            unseen_movies = random.choices(seen_movies, k=10)
            seen_movies = list(filter(lambda movie: movie not in unseen_movies, seen_movies))

            for movie in unseen_movies:
                max_similarity, most_similar_movie = float('-inf'), None

                for sim_movie in seen_movies:
                    similarity = self._two_movie_similarity(movie, sim_movie)

                    if similarity > max_similarity:
                        most_similar_movie = sim_movie
                        max_similarity = similarity

                y.append(self.ratings_matrix[user-1, movie])
                y_hat.append(self.predict_rating(max_similarity, user, movie))

        return np.array(y), np.array(y_hat)

    def predict_rating(self, similarity: float, user_id: int, movie_id: int) -> float:
        """Returns predicted rating for current user"""
        rate = min(self.ratings_matrix[user_id-1, movie_id] * (self.slope*similarity + self.intercept), 5)

        return max(round(rate * 2) / 2, 1)
