# standard imports
import numpy as np
import pandas as pd
import networkx as nx

# local imports 
from source.reroute import what_can_be_rerouted
import source.data_preprocessing as dp


# Decide the edge path for an OD pair is success or fail 
def decide_path_status(edge_path,set_fail_edge_id):
    if (set(edge_path)&set_fail_edge_id):
        return 0
    else:
        return 1
    


#input: s_demand, fail_edge_idx
#output: s_delivered (original routes & rerouted)
#output: s_reouted (those get delivered through rerouting)
def calculate_delivered_flows(s_demand,fail_edge_idx, 
                              G_asset=dp.G_asset,
                              df_e=dp.df_e,
                              od_e_path=dp.od_e_path,
                              s_journeys=dp.s_journeys,
                              od_idx=dp.od_idx,
                              multi_search_cutoff=dp.multi_search_cutoff):
    #return whateve get delivered
    
    #Copy original G_asset and remove edges that 1)failed 2)with zero journey:
    R=G_asset.copy()
    df_failed_e=df_e.loc[fail_edge_idx]
    edge_tuples=list(zip(df_failed_e.from_node,df_failed_e.to_node))
    R.remove_edges_from(edge_tuples)

    df_zero=df_e[df_e.journeys==0]
    edge_tuples=list(zip(df_zero.from_node,df_zero.to_node))
    R.remove_edges_from(edge_tuples)

    #decide status of od flows
    fail_edge_id=df_e.loc[fail_edge_idx].edge_id.tolist()
    set_fail_edge_id=set(fail_edge_id)

    #Decide if the original path is on or off:
    s_mask=od_e_path.map(lambda x: decide_path_status(x,set_fail_edge_id))

    #s_od_state=s_mask*s_journeys/s_demand
    #

    s_overflow=s_demand.sub(s_journeys*s_mask)

    s_rerouted=pd.Series(data=[0]*len(od_idx), index=od_idx, dtype='float32')
    
    s_flow=s_overflow.copy()
    j=0
    while j<multi_search_cutoff:
        #start=time.time()
        R,rerouted_flow=what_can_be_rerouted(R,s_flow) #len(s_flow=2m)


        if R=="break":
            break
        else:
            check=rerouted_flow-s_flow.loc[rerouted_flow.index]
            if any(check>0.000001):
                print(j, 'rerouted_flow > s_flow - check2')
            j+=1
            s_flow.loc[rerouted_flow.index] = s_flow.loc[rerouted_flow.index].sub(rerouted_flow)
            s_rerouted.loc[rerouted_flow.index] = s_rerouted.loc[rerouted_flow.index].add(rerouted_flow) 
            '''
            if any(s_rerouted >s_overflow):
                bug = s_rerouted - s_overflow
                lenn=bug[bug>0].shape
                summ = bug[bug>0].sum()
                print(j,'s_rerouted>s_overflow', summ, lenn)'''

        #stop=time.time()
        #print(s_rerouted.sum(),stop-start)
    s_delivered=(s_journeys*s_mask).add(s_rerouted, fill_value=0)
    return s_delivered,s_rerouted
