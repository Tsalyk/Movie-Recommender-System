from recommender_engine import MovieRecommender
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_rmse(mr: MovieRecommender):
    """Plots RMSE for the different range of parameters"""
    intercepts = np.arange(0, 1.01, 0.1)
    slopes = [1, 2]
    rmse = [[] for _ in range(len(slopes))]

    for slope in slopes:
        for intercept in intercepts:
            mr.slope, mr.intercept = slope, intercept
            rmse[slope-1].append(mr.evaluate(100))

    plt.figure(figsize=(7, 5))
    plt.plot(intercepts, rmse[0], label='slope = 1')
    plt.plot(intercepts, rmse[1], label='slope = 2')
    plt.xlabel('intercept')
    plt.ylabel('RMSE')
    plt.legend()
    plt.title('Root Mean Square Error for 100 people and 1000 films')
    plt.show()
