from data_preprocessing import preprocess
from recommender_engine import MovieRecommender
from system_testing import plot_rmse


if __name__ == '__main__':
    mr = MovieRecommender(*preprocess('data/movies_metadata.csv', 'data/ratings_small.csv'))
    print(mr.top_recommendations(10, 5))
    plot_rmse(mr)
