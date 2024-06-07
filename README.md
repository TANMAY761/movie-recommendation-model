### Movie Recommendation System
This project is a movie recommendation system that uses TF-IDF vectorization and cosine similarity to recommend movies based on the user's favorite movie. The system suggests movies that are similar in genres, keywords, tagline, cast, and director.

### Introduction
This project demonstrates how to build a content-based movie recommendation system. The system takes a user's favorite movie as input and recommends movies that are similar based on several features extracted from the dataset.

### Dataset
The dataset used in this project is movies.csv, which contains information about movies, including genres, keywords, tagline, cast, and director.

### Requirements
Python 3.x
NumPy
Pandas
scikit-learn
difflib

### Detailed Steps:
Data Preparation
First, the necessary libraries are imported, and the current working directory is set to the location of the movies.csv file. The dataset is loaded using Pandas.

### Feature Engineering
We select relevant features for the recommendation: genres, keywords, tagline, cast, and director. Missing values are filled with empty strings

### Model Training
TF-IDF Vectorizer is used to convert the text data into numerical values, which are then used to compute cosine similarity between movies.
### Movie Recommendation
The user is prompted to enter their favorite movie. The system finds the closest match in the dataset and recommends similar movies.
### Results
When you run the script and enter your favorite movie name, the system will output a list of movies similar to your input based on the combined features.

