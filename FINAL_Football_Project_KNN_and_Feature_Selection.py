import pandas as pd
import numpy as np
import faiss
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler

def Analysis(Player,data_enc) : 
    mapping_dict_comp = {'Premier League': 1, 'La Liga': 2, 'Serie A': 3, 'Bundesliga': 4, 'Ligue 1': 5}
    data_enc.loc[:, 'Comp'] = data_enc['Comp'].map(mapping_dict_comp)
    data_enc.drop(columns = ['Nation','Squad','Comp'],inplace=True)

    data_enc.dropna(inplace=True)
    player_names = data_enc['Player']
    data_enc.drop(columns = ['Player'],inplace=True)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(data_enc)

    pca = decomposition.PCA(n_components=30)
    x = pca.fit_transform(x_scaled)


    feature_names = data_enc.columns
    loadings = pd.DataFrame(pca.components_.T,columns=['PC{}'.format(i) for i in range(1, 31)],index = feature_names)
    loadings = loadings.abs()
    df_loadings = loadings.sort_values(by = 'PC1',ascending=False)
    first_30_indexes = df_loadings.index[:30].tolist()#the 20 most relevant attributes

    vectors = x.astype('float32')

    d = 30
    m = 5
    n_bits = 5

    pq = faiss.IndexPQ(d,m,n_bits)
    pq.train(vectors)
    pq.add(vectors)

    query_player = Player
    query_data = data_enc[player_names == query_player].astype('float32')
    query_scaled = scaler.transform(query_data)
    query_pca = pca.transform(query_scaled).astype('float32')

    k = 25

    distances, indices = pq.search(query_pca, k)

    query_index = np.where(player_names == query_player)[0][0]  
    nearest_neighbors_indices = indices[0][indices[0] != query_index][:k]  
    nearest_neighbor_names = player_names.iloc[nearest_neighbors_indices].values.tolist()
    return nearest_neighbor_names,distances

Player = input("Enter the Player Name : \n")
Choice = int(input(f"Enter 1 to Find the 5 closest Players to {Player}\nEnter 2 to Find the Closest Under 23 Player to {Player}"))

data = pd.read_csv('Dataset2.csv', encoding='ISO-8859-1', delimiter=';')
column = 'Pos'
one_hot_encoded = pd.get_dummies(data['Pos'],prefix = 'Pos')
data_enc = pd.concat([data,one_hot_encoded],axis=1)
data_enc.drop(columns = [column],inplace=True)


print(f"The Query Player is : {Player}")

if Choice == 1:
    nearest_neighbors, distances = Analysis(Player, data_enc)
    for i in range(5):
        print(f"Player {i + 1}: {nearest_neighbors[i]}")

if Choice == 2:
    players, distances = Analysis(Player, data_enc)
    for i, player in enumerate(players):
        player_age = data_enc.loc[data['Player'] == player, 'Age'].values[0]
        if player_age <= 23:
            print(f"Closest player under 23: {player}")
