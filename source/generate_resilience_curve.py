'''
    what does this script do 
'''

# Standard imports
import numpy as np
import pandas as pd
import networkx as nx
import itertools

# Local application/library specific imports
import source.data_preprocessing as dp
from source.calculate_delivered_flows import calculate_delivered_flows



# Variables from data preprocessing 
# df_e = dp.df_e
# od_e_path = dp.od_e_path
# od_idx = dp.od_idx
# idx_reduced_OD = dp.idx_reduced_OD
# OD_reduced = dp.OD_reduced
# s_journeys = dp.s_journeys
# G_asset = dp.G_asset



####################
# DEFINE FUNCTIONS #
####################
def curve_generator(a_scenario, t_step_cutoff,
                    df_e=dp.df_e,
                    s_journeys=dp.s_journeys,
                    recover_p=dp.recover_p):

    if sum(a_scenario)==0:
        return np.nan, np.nan, np.nan, np.nan
    else:
        #give the list of index of failed edges: 
        mask=[bool(item) for item in a_scenario] #0-functioning, 1-fail
        fail_edge_idx=df_e[mask].index.tolist()

        w_d = []
        
        demand=s_journeys
        i=0
        while i <t_step_cutoff:
            #start=time.time()
            N=len(fail_edge_idx)
            if N>0:
                delivered, rerouted=calculate_delivered_flows(demand,fail_edge_idx) #pd series object 
                
                # calculate percentage unsatisfied demand 
                w_d.append(delivered.sum() / demand.sum())

                #try to recover
                mask=(np.random.uniform(size=N) < (1-recover_p)) * 1
                fail_edge_idx_i=list(itertools.compress(fail_edge_idx, mask))

                # calculate the new demand
                demand_i=s_journeys.add(demand,fill_value=0).sub(delivered, fill_value=0)

                i=i+1
                fail_edge_idx=fail_edge_idx_i
                demand=demand_i
                
            else:
                delivered=np.minimum(s_journeys*1.5,demand)

                w_d.append(delivered.sum() / demand.sum())

                demand_i=s_journeys.add(demand,fill_value=0).sub(delivered, fill_value=0)

                i=i+1
                demand=demand_i

            #stop = time.time()
            #print(stop-start)

        arr_curve = np.array(w_d)
        return arr_curve



'''
#input: sample_idx 
#output - 4*dataframe: array - demand, delivered,rerouted, points for the curves 
#v2 - STOP rerouting even the original path is recovered. 
# v3 - w_d only
def curve_generator(a_scenario, t_step_cutoff,
                    df_e=dp.df_e,
                    s_journeys=dp.s_journeys,
                    recover_p=dp.recover_p):

    if sum(a_scenario)==0:
        return np.nan, np.nan, np.nan, np.nan
    else:
        #give the list of index of failed edges: 
        mask=[bool(item) for item in a_scenario] #0-functioning, 1-fail
        fail_edge_idx=df_e[mask].index.tolist()

        # df_demand=pd.DataFrame(s_journeys, dtype='float32')
        # df_delivered=pd.DataFrame(s_journeys, dtype='float32')
        # df_rerouted=pd.DataFrame([0]*s_journeys.shape[0],index=s_journeys.index, dtype='float32')

        w_d = []
        
        demand=s_journeys
        i=0
        while i <t_step_cutoff:
            #start=time.time()
            N=len(fail_edge_idx)
            if N>0:
                #print(i,day_idx,sample_idx)
                delivered, rerouted=calculate_delivered_flows(demand,fail_edge_idx) #pd series object 
                # df_delivered.loc[:,'t{}'.format(i+1)]=delivered
                # df_rerouted.loc[:,'t{}'.format(i+1)]=rerouted

                w_d.append(delivered.sum() / demand.sum())

                #try to recover
                mask=(np.random.uniform(size=N) < (1-recover_p)) * 1
                fail_edge_idx_i=list(itertools.compress(fail_edge_idx, mask))

                demand_i=s_journeys.add(demand,fill_value=0).sub(delivered, fill_value=0)
                #For those not get rerouted, their demand will not be added up, 
                #cz we asusme they either give up the trip, or used an alternative type of transport
                #they will recover when the original path get recovered. 
                # reset_demand=s_journeys.loc[~s_journeys.index.isin(idx_reduced_OD)]
                # demand_i.loc[~demand_i.index.isin(idx_reduced_OD)]=reset_demand
                # df_demand.loc[:,'t{}'.format(i+1)]=demand_i

                i=i+1
                fail_edge_idx=fail_edge_idx_i
                demand=demand_i
            else:

                #print(i)
                #delivered = 0.5* journeys (extra capacity) + 1*journeys
                delivered=np.minimum(s_journeys*1.5,demand)
                # df_delivered.loc[:,'t{}'.format(i+1)]=delivered
                # df_rerouted.loc[:,'t{}'.format(i+1)]=0

                w_d.append(delivered.sum() / demand.sum())

                demand_i=s_journeys.add(demand,fill_value=0).sub(delivered, fill_value=0)
                # reset_demand=s_journeys.loc[~s_journeys.index.isin(idx_reduced_OD)]
                # demand_i.loc[~demand_i.index.isin(idx_reduced_OD)]=reset_demand
                # df_demand.loc[:,'t{}'.format(i+1)]=demand_i

                i=i+1
                demand=demand_i

            #stop = time.time()
            #print(stop-start)

        # arr_demand=df_demand.to_numpy()
        # arr_delivered=df_delivered.to_numpy()
        # arr_rerouted=df_rerouted.to_numpy()

        arr_curve = np.array(w_d)
        return arr_curve

'''
    


