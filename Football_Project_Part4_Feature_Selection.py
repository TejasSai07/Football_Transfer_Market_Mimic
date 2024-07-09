import pandas as pd
import faiss
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression

# Load the data from the CSV file
df = pd.read_csv('Dataset2.csv', encoding='ISO-8859-1', delimiter=';')

# Dropping any non-numeric columns assuming they're not part of features
numeric_columns = df.select_dtypes(include=np.number).columns
selected_data = df[numeric_columns]

# Drop rows with missing values
selected_data.dropna(inplace=True)

# Separating player names and the rest of the data
player_names = selected_data['Player']
selected_data.drop(columns=['Player'], inplace=True)

# Feature selection using SelectKBest
X = selected_data.values
y = player_names.values

# Perform feature selection (Select top 20 features based on f_regression)
selector = SelectKBest(score_func=f_regression, k=20)  # Adjust 'k' as needed
X_selected = selector.fit_transform(X, y)

# Get indices of selected features
selected_feature_indices = selector.get_support(indices=True)

# Use only selected features for faiss indexing
vectors_selected = X[:, selected_feature_indices]

# Initialize and train the faiss index
d = 20  # Assuming 'd' is the number of dimensions after feature selection
m = 5
n_bits = 5

pq = faiss.IndexPQ(d, m, n_bits)
pq.train(vectors_selected)
pq.add(vectors_selected)

# Example query
query_player = 'Dani Alves'
query_vector = selected_data[player_names == query_player].values.astype('float32')

k = 6
distances, indices = pq.search(query_vector, k)

query_index = np.where(player_names == query_player)[0][0]  
nearest_neighbors_indices = indices[0][indices[0] != query_index][:k]  
nearest_neighbor_names = player_names.iloc[nearest_neighbors_indices].values.tolist()

print("Query Player:", query_player)
print("Nearest neighbor indices:", nearest_neighbors_indices)
print("Nearest neighbor names:", nearest_neighbor_names)
