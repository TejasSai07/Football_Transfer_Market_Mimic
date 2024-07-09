import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

df = pd.read_csv('Dataset2.csv', encoding='ISO-8859-1', delimiter=';')

selected_attributes = ['Player', 'Comp', 'Pos', 'Goals', 'Shots', 'PasTotCmp', 'Assists', 'PasCrs', 'SCA', 'ScaDrib', 'GCA', 'TklWon', 'TklDriPast', 'PresSucc', 'Int', 'TouDefPen', 'Carries', 'CrdY', 'Recov', 'AerWon%']
selected_data = df[selected_attributes].copy()  

mapping_dict_comp = {'Premier League': 1, 'La Liga': 2, 'Serie A': 3, 'Bundesliga': 4, 'Ligue 1': 5}
selected_data.loc[:, 'Comp'] = selected_data['Comp'].map(mapping_dict_comp)

mapping_dict_pos = {'DF': 1, 'MF': 2, 'FW': 3, 'MFFW': 4, 'FWMF': 5, 'GK': 6, 'DFMF': 7, 'MFDF': 8, 'FWDF': 9, 'DFFW': 10}
selected_data.loc[:, 'Pos'] = selected_data['Pos'].map(mapping_dict_pos)

selected_data.dropna(inplace=True)  

player_names = selected_data['Player']
selected_data.drop(columns=['Player'], inplace=True)  

scaler = StandardScaler()
scaled_data = scaler.fit_transform(selected_data)

knn = NearestNeighbors(n_neighbors=55, algorithm='auto')
knn.fit(scaled_data)

def find_similar_player(player_name):
    if player_name in player_names.values:
        player_index = player_names[player_names == player_name].index[0]
        player_data = scaled_data[player_index].reshape(1, -1)
        distances, indices = knn.kneighbors(player_data)
        closest_player_index = indices.flatten()[1]  
        closest_player = player_names[closest_player_index]
        distance_to_player = distances.flatten()[1]
        return closest_player, distance_to_player
    else:
        return None, None

player_name_to_find = 'Dani Alves' 
closest_player, distance_to_player = find_similar_player(player_name_to_find)
if closest_player and distance_to_player:
    print(f"Closest player to {player_name_to_find}: {closest_player}")
else:
    print(f"{player_name_to_find} not found in the dataset or has insufficient data.")
