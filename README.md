# Football Transfer Market Mimic : 

## Overview : 
  - This project was made since I have a great passion in football and understadning the market dynamics for player trade, player to player comparisons are all key aspects that had to be considered and taken account of.

## Features :

This project utilizes the following techniques/frameworks/tech : 

- ***KNN (K- Nearest Neighbors)*** -> machine learning algorithm that classifies data points based on the majority class of their k nearest neighbors. In this project, KNN helps identify similar players by finding those with the closest attribute profiles
- ***FAISS (Facebook AI Similarity Search)*** -> enables efficient similarity search and clustering of dense vectors. Its primary purpose is to perform rapid searching through large datasets to find vectors similar to a given query vector. It significantly outperforms traditional KNN when dealing with high-dimensional data
- ***tSNE (t-distributed Stochastic Neighbor Embedding)*** -> non-linear dimensionality reduction technique that's well-suited for visualizing high-dimensional data. tSNE focuses on preserving local similarities, which is great for identifying clusters of similar players. It helps visualize player groupings based on their attributes.
- ***PCA (Principal Component Analysis)*** -> linear dimensionality reduction technique that identifies the directions (principal components) along which data varies the most. PCA helps reduce the number of player attributes while retaining the most important information, making similarity searches more efficient.

## Process : 

- Initially the Dataset was taken from Kaggle and the dataset was analyzed thoroughly to see if it had the requirements of players attributes, etc from which Similarity Search and Market analysis of players could be done.
- Next mapping of categorical data to numerical values for easier processing was done.
- Once, the categorical data was mapped to numerical values, data preprocessing was ready to be done.
- Data preprocessing was done in order to drop data without enough data/repeating rows and Scale values so that all values are on a certain scale.
- Next using KNN, the closest players for a  given player were extracted and displayed

## Issues and Rectifications
- If I wanted to scale this, I identified that KNN isn't the most optimnal solution and came around FAISS to be the best alternative.
- FAISS is a scalable, fast library which allows us to quick search and retrieve (near-optimal) values.
- In the code, the parameters k(number of neighbors), m, d and n_bits were varied to identify the optimal solutions -> HyperParameters

## Further Optimization 
- Feature Selection was done to extract the 20 best features then map, since I wanted to understand the properties of features and how choosing/leaving out features afftects the quality of results
- PCA and tSNE were incorporated to understand and implement dimensionality reduction and understand how dimensionality reduction affect quality of output
- the difference between the core undwerstanding of PCA and tSNE along with the logic was implemented and learnt.
