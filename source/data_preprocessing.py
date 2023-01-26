import pandas as pd
import pyproj
import networkx as nx
import pickle

# Constancts 
multi_search_cutoff = 5
recover_p = 0.5

#load edge data
df_e=pd.read_pickle('/data/cip19ql/rail_psgr_tasmax/data_input/edge_nc_cells.pkl')

#load OD matrix 
OD_pkl_path='/data/cip19ql/rail_psgr_tasmax/data_input/df_OD.pkl'
OD=pd.read_pickle(OD_pkl_path)
OD_reduced=OD[(OD.journeys>15) & (OD.distance>30000)]

od_e_path=OD.edge_path
od_idx=OD.index
s_journeys=OD.journeys
idx_reduced_OD=OD_reduced.index.tolist()


#Calculate the length of every edge
def geo2len(line_string):
    geod = pyproj.Geod(ellps="WGS84")
    edge_len = geod.geometry_length(line_string)
    #edge_len=int(edge_len)
    return edge_len

df_e['distance'] = df_e['geometry'].map(geo2len).tolist()


#Create a undirected graph 
G_asset = nx.Graph()
for i in range(df_e.shape[0]):
    G_asset.add_edge(df_e.iloc[i]['from_node'], df_e.iloc[i]['to_node'],
                     edge_id=df_e.iloc[i]['edge_id'],
                     weight=df_e.iloc[i]['journeys'],
                     distance=df_e.iloc[i]['distance'],
                     capacity=df_e.iloc[i]['journeys']*0.5)
#remove self loops
G_asset.remove_edges_from(nx.selfloop_edges(G_asset))

