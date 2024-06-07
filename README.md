# movie-recommendation-model
Movie Recommendation System
This project is a movie recommendation system that uses TF-IDF vectorization and cosine similarity to recommend movies based on the user's favorite movie. The system suggests movies that are similar in genres, keywords, tagline, cast, and director.

Table of Contents
Introduction
Dataset
Requirements
Usage
Detailed Steps
Data Preparation
Feature Engineering
Model Training
Movie Recommendation
Results
License
Introduction
This project demonstrates how to build a content-based movie recommendation system. The system takes a user's favorite movie as input and recommends movies that are similar based on several features extracted from the dataset.

Dataset
The dataset used in this project is movies.csv, which contains information about movies, including genres, keywords, tagline, cast, and director.

Requirements
Python 3.x
NumPy
Pandas
scikit-learn
difflib
You can install the required packages using the following command:

bash
Copy code
pip install numpy pandas scikit-learn
Usage
Clone the repository:
bash
Copy code
git clone https://github.com/yourusername/movierecommendation.git
cd movierecommendation
Ensure the movies.csv file is in the project directory.

Run the script:

bash
Copy code
python recommend.py
Follow the prompts to enter your favorite movie name.
Detailed Steps
Data Preparation
First, the necessary libraries are imported, and the current working directory is set to the location of the movies.csv file. The dataset is loaded using Pandas.

python
Copy code
import numpy as np
import pandas as pd
import os

# Load dataset
movies_data = pd.read_csv('movies.csv')
Feature Engineering
We select relevant features for the recommendation: genres, keywords, tagline, cast, and director. Missing values are filled with empty strings.

python
Copy code
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')
The selected features are then combined into a single string for each movie.

python
Copy code
combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + movies_data['tagline'] + ' ' + movies_data['cast'] + ' ' + movies_data['director']
Model Training
TF-IDF Vectorizer is used to convert the text data into numerical values, which are then used to compute cosine similarity between movies.

python
Copy code
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

similarity = cosine_similarity(feature_vectors)
Movie Recommendation
The user is prompted to enter their favorite movie. The system finds the closest match in the dataset and recommends similar movies.

python
Copy code
import difflib

# Get user input
movie_name = input('Enter your favourite movie name: ')

# Find close match
list_of_all_titles = movies_data['title'].tolist()
find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
close_match = find_close_match[0]

# Get index of the movie
index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]

# Get similarity scores
similarity_score = list(enumerate(similarity[index_of_the_movie]))

# Sort movies based on similarity scores
sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

# Display recommended movies
print('Movies suggested for you:\n')
i = 1
for movie in sorted_similar_movies:
    index = movie[0]
    title_from_index = movies_data[movies_data.index == index]['title'].values[0]
    if i < 30:
        print(i, '.', title_from_index)
        i += 1
Results
When you run the script and enter your favorite movie name, the system will output a list of movies similar to your input based on the combined features.
