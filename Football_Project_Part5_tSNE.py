import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import faiss

data = pd.read_csv('Dataset2.csv', encoding='ISO-8859-1', delimiter=';')
column = 'Pos'
one_hot_encoded = pd.get_dummies(data['Pos'],prefix = 'Pos')
data_enc = pd.concat([data,one_hot_encoded],axis=1)
data_enc.drop(columns = [column],inplace=True)
mapping_dict_comp = {'Premier League': 1, 'La Liga': 2, 'Serie A': 3, 'Bundesliga': 4, 'Ligue 1': 5}
data_enc.loc[:, 'Comp'] = data_enc['Comp'].map(mapping_dict_comp)
data_enc.drop(columns = ['Nation','Squad','Comp'],inplace=True)

data_enc.dropna(inplace=True)
player_names = data_enc['Player']
data_enc.drop(columns = ['Player'],inplace=True)

x = data_enc.copy()

tsne = TSNE(n_components=2,perplexity=5,random_state=34)
x_tsne = tsne.fit_transform(x)
'''
plt.figure(figsize=(8, 6))
plt.scatter(x_tsne[:, 0], x_tsne[:, 1], marker='.')
plt.title('t-SNE Visualization')
plt.xlabel('Player')
plt.ylabel('Attributes after Dimensionality Reduction')
plt.show()
'''
vectors = x_tsne.astype('float32')

d = 2
m = 2
n_bits = 5

pq = faiss.IndexPQ(d,m,n_bits)
pq.train(vectors)

pq.add(vectors)

query_player = 'Sofyan Amrabat'
query_vector = x_tsne[player_names == query_player].astype('float32')

k = 6

distances, indices = pq.search(query_vector, k)

query_index = np.where(player_names == query_player)[0][0]  
nearest_neighbors_indices = indices[0][indices[0] != query_index][:k]  
nearest_neighbor_names = player_names.iloc[nearest_neighbors_indices].values.tolist()

print("Query Player:", query_player)
print("Nearest neighbor indices:", nearest_neighbors_indices)
print("Nearest neighbor names:", nearest_neighbor_names)
print("Distances:", distances)
