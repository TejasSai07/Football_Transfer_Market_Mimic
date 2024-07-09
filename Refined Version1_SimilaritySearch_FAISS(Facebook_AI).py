import pandas as pd
import faiss
import numpy as np

df = pd.read_csv('Dataset2.csv', encoding='ISO-8859-1', delimiter=';')

selected_attributes = ['Player', 'Comp', 'Pos', 'Goals', 'Shots', 'PasTotCmp', 'Assists', 'PasCrs', 'SCA', 'ScaDrib', 'GCA', 'TklWon', 'TklDriPast', 'PresSucc', 'Int', 'TouDefPen', 'Carries', 'CrdY', 'Recov', 'AerWon%','PKwon']
selected_data = df[selected_attributes].copy()  

mapping_dict_comp = {'Premier League': 1, 'La Liga': 2, 'Serie A': 3, 'Bundesliga': 4, 'Ligue 1': 5}
selected_data.loc[:, 'Comp'] = selected_data['Comp'].map(mapping_dict_comp)

mapping_dict_pos = {'DF': 1, 'MF': 2, 'FW': 3, 'MFFW': 4, 'FWMF': 5, 'GK': 6, 'DFMF': 7, 'MFDF': 8, 'FWDF': 9, 'DFFW': 10}
selected_data.loc[:, 'Pos'] = selected_data['Pos'].map(mapping_dict_pos)

selected_data.dropna(inplace=True)  

player_names = selected_data['Player']
selected_data.drop(columns=['Player'], inplace=True)  

vectors = selected_data.astype('float32').values

d = 20
m = 5
n_bits = 5

pq = faiss.IndexPQ(d,m,n_bits)
pq.train(vectors)

pq.add(vectors)

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
print("Distances:", distances)