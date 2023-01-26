'''
    simulate the resilience curve for random failure scenarios  
'''

# Standard library imports
import sys
import math
import scipy
import pickle
import argparse
import numpy as np
from scipy import stats 

# Related third party imports
import h5py



# Local application/library specific imports

# sys.path.append('/data/cip19ql/rail_psgr_tasmax/source')
import source.data_preprocessing as dp

from source.generate_resilience_curve import curve_generator

import source.exp_edge_failure as exp_edge_failure


# Constants
N = 250 # number of trials

# Output filenames
RESULTS_FOLDER = '/data/cip19ql/rail_psgr_tasmax/data_output/targeted/'



#############################

def main(day, exp):

    df_e = dp.df_e
    recover_p = dp.recover_p
    
    sort_j_idx = df_e.sort_values(by='journeys', ascending=False, na_position='last').index.tolist()
    
    #fragile function 
    mean = 35 + 273
    sd = 2.5  #about 95% of the values lie within two standard deviations, so 2*std=5deg,

    t_step_cutoff = math.floor(4*( 1- math.log(exp)/math.log(1-recover_p) ))
    
    # all edges are subjected to teh same probability of failure of NO_removal/NO_edges 
    p_fail=[exp/df_e.shape[0]]*df_e.shape[0]
    N_sample=[(np.random.uniform(size=N) < p) * 1 for p in p_fail] #0-functioning, 1-fail
    
    # #0-functioning, 1-fail

    ################################################################################################

    path1 = RESULTS_FOLDER +  'targ_' + day + '.hdf5'
    f1 = h5py.File(path1,'a')

    dset_curve = f1.create_dataset('curve', shape=tuple([N,t_step_cutoff]), data=None, compression="gzip", compression_opts=9)

    f1.close()
    
    for i in range(N):
        a_scenario=np.array([item[i] for item in N_sample])
        NO_removal = sum(a_scenario)
        the_scenario = np.zeros(df_e.shape[0])
        the_scenario[sort_j_idx[:NO_removal]]=1
        if sum(the_scenario) != 0:
            arr_curve = curve_generator(the_scenario,t_step_cutoff)
            f1 = h5py.File(path1,'a')
            f1['curve'][i,:] = arr_curve
            f1.close()
        else:
            pass
    






    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--index', type=int, help='index')
    
    parser.add_argument(
        '-d', '--day', type=str, help='e.g. 2016_d1679')
    
    parser.add_argument(
        '-e', '--exp', type=float, help='this is the exp_e_failure, float')

    args = parser.parse_args()

    if args.index is None:
        if args.day is None:
            args.day = ''
        
        if args.exp is None:
            args,exp = 99999
    else:
        args.day, args.exp = exp_edge_failure.exp_set[args.index]

    main(args.day, args.exp)
