# standad imports 
import numpy as np
import pandas as pd
import networkx as nx

# local inports 
import source.data_preprocessing as dp


# Try to find a path in G between origin node and destination node
def find_a_path(o,d,dist,G):
    if nx.has_path(G,o,d):
        try: 
            path=nx.single_source_dijkstra(G,o,target=d, cutoff=dist*2, weight='distance')
            return path[1]
        except:
            return np.nan
    else:
        return np.nan




#input: R, s_flow (series - OD flows thast are still not delivered (2282270 rows) )
#return: R (reduced capacity), rerouted flow
def what_can_be_rerouted(R,s_flow, idx_reduced_OD=dp.idx_reduced_OD, OD_reduced=dp.OD_reduced ):
    #global: OD_reduced, idx_reduced_OD
    idx_list=s_flow[(s_flow>0) & (s_flow.index.isin(idx_reduced_OD))].index.tolist()
    
    #remove zero capacity edges 
    zero_capacity=[]
    for u,v,attr in R.edges(data=True):
        if attr['capacity']==0:
            zero_capacity.append(tuple([u,v]))
        elif attr['capacity']<0:
            print('Error: Negative Capacity')
        else:
            pass
    R.remove_edges_from(zero_capacity)
    
    #list out isolated nodes in R  
    iso_nodes=list(nx.isolates(R))

    iso_od_idx=OD_reduced[(OD_reduced.origin_id.isin(iso_nodes)) | 
                      (OD_reduced.destination_id.isin(iso_nodes)) ].index.tolist()
    #and excludes those from path search
    idx_list=set(idx_list) - set(iso_od_idx)
    
    """BREAK_POINT - idx_list is empty"""
    if len(idx_list)==0:
        return "break","break"
    
    else:
        #temporay dataframe df
        df=OD_reduced[OD_reduced.index.isin(idx_list)].copy()
        df=df.drop(columns=['node_path','edge_path'])

        #map function (find_a_path) to df
        df['new_path']=df.apply(lambda x: find_a_path(x['origin_id'],x['destination_id'],x['distance'],R), axis=1)
        #4min for 85k solid path search (used G_asset, not G)
        #2min for 22k solid search with 63k no_path

        #select rows that with a path found
        df_path=df[df.new_path.notnull()]
        
        """BREAK POINT - cant find any path with capacity"""
        if df_path.shape[0]==0:
            return "break", "break"
        
        else:
            # note: there is no priority here, as we assume all spare capacity are shared across all od
            # therefore, there will be a residual network, where every edge, as long as such edges exist, 
            # will have a capacity of 0.5*journey

            # Initialize/reset the residual network.
            for u in R:
                for e in R[u].values():
                    e["flow"] = 0
                    e["status"]=0

            #Augment flow along a path from o to d
            for idx in df_path.index:
                path = df_path.loc[idx,'new_path']
                flow = s_flow.loc[idx]
                for i in range(len(path)-1):
                    u=path[i]
                    v=path[i+1]
                    R[u][v]['flow']+=flow

            #Calculate status of each asset: within/over its capacity 
            for u,v,attr in R.edges(data=True):
                if attr['capacity']>0 and attr['flow']>0:
                    if attr['flow']>attr['capacity']:           #over its spare capcity
                        R[u][v]["status"]= attr['capacity'] / attr['flow']
                        R[u][v]['capacity']=0
                    else:                                       #within its spare capacity
                        R[u][v]["status"]= 1
                        R[u][v]['capacity']-=attr['flow']
                elif attr['capacity']>0 and attr['flow']==0:     #have capacity but no flow, 
                    R[u][v]["status"]= 1
                    #do nothing to R[u][v]['capacity']
                else:
                    pass 

            #find the limiting asset / bottle neck along the path
            bottle_neck=pd.Series(data=[0]*df_path.shape[0],index=df_path.index, dtype='float32')

            for idx in df_path.index:
                path=df_path.loc[idx,'new_path']
                factor=[]

                for i in range(len(path)-1):
                    u=path[i]
                    v=path[i+1]
                    factor.append(R[u][v]['status'])
                bottle_neck.loc[idx]=min(factor)
            if any(bottle_neck>1):
                print('bottle_neck > 1')
            rerouted_flow=bottle_neck.multiply(s_flow[bottle_neck.index], fill_value=0)
            if any(rerouted_flow>s_flow.loc[rerouted_flow.index]):
                print('rerouted_flow>s_flow - check1')
            return R, rerouted_flow
