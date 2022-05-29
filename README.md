# Movie-Recommender-System
***
## Aim
The main goal of our project is an effective movie recommender system based on collaborative filtering and SVD.
***
## Data
Dataset is in public access:
https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?select=ratings.csv

There are two main files: ratings.csv and movies_metadata.csv. The first one contains *100,000* ratings from *700* users on *9,000* movies. The second has descriptions of all these movies: posters, backdrops, budget, revenue, release dates, languages, production countries, and companies.

***
## Brief explanation of the algoritm
The whole idea is to conduct SVD on matrix with rated films and choose principal components to be reduced SVD. After that we can compute cosine similarity between each film vector and based on rated films predict ratings for unseen films.
***
## Singular Value Decomposition
Assume that we have some matrix A, which is m x n  with a rank equal to r. Then its SVD decomposition will equal to the multiplication of several matrices:

$USV^T$ = (orthogonal)(diagonal)(orthogonal)

* **$S$** is diagonal, not necessarily square m-by-n matrix, that on the diagonal has square roots of eigenvalues of $A^TA$ arranged in decreasing order. Those numbers are called singular values of $A$, they fill the first r places on the main diagonal of $S$ (r corresponds to a rank of matrix $A$) and other entries are zeroes

* **$U$** is m x m orthogonal matrix, in which columns are orthonormal eigenvectors of $AA^T$

* **$V$** is  n x n orthogonal matrix, in which columns correspond to orthonormal eigenvectors of $A^TA$

***
## Reduced Singular Value Decomposition
As before we represent $A = USV^T$, but know our matrix $S$ is a square diagonal r-by-r matrix with r equal to a rank of $A$, which is often smaller than initial m or n. Then matrices $U$ became the m x r matrix and $V$ r x n matrix. They are called **semi-orthogonal** because they are not square anymore. This approach helps to reduce memory usage.
***
## Cosine similarity
$\Huge cos(\theta) = \frac {A \cdot B}{||A|| \cdot ||B||}$

To recommend a certain movie to a person, we have to find films that are as similar as possible to those that he likes a lot. This task is about finding the closest vectors to that, which represent the person's favorite movie. To accomplish this task we will use the cosine similarity technique, which measures the cosine of the angle between two vectors. The smaller angle, the higher the cosine similarity we will receive in the result.
***
## Pseudocode
```
A ← construct_ratings_matrix()
VT← SVD(A)
ratings ← (array[])
for each col VT
       similarity ← cosine_similarity(col)
       predicted_rating ← predict(similarity)
       ratings.add(predicted_rating)

return top(n) sorted(ratings, reverse=True)
```
***
## Example of the output

***
## Results
As a result, we got good-working algorithm, which can predict ratings of films for the user. We conducted expiriments with different range of parameters and got low enough **RMSE**.

<img width="421" alt="rmse" src="https://user-images.githubusercontent.com/73395389/170862266-25f07deb-f6da-4f59-ad4b-3b752a89f75d.png">

***
## Credits
[Markiian Tsalyk](https://www.linkedin.com/in/markiian-tsalyk-193758224/)

[Yuriy Sukhorskyy](https://ua.linkedin.com/in/yuriy-sukhorskyy)

[Severyn Peleshko](https://www.linkedin.com/in/severyn-peleshko-163a71225/)
***
## License
[MIT](https://github.com/Tsalyk/Movie-Recommender-System/blob/main/LICENSE)
