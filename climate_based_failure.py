'''
    simulate the resilience curve for climate-based failure scenarios  
'''

# Standard library imports
import math
import scipy
import pickle
import argparse
import numpy as np
from scipy import stats 

# Related third party imports
import h5py
import netCDF4




# Local application/library specific imports

# sys.path.append('/data/cip19ql/rail_psgr_tasmax/source')
import source.data_preprocessing as dp

from source.generate_resilience_curve import curve_generator

import source.example_days as days


# Constants
# date = 940
N = 250 # number of trials
NC_FOLDER = '/data/cip19ql/input_tasmax/'
NC_DATASET_NAME = 'tasmax_EUR-11_CNRM-CERFACS-CNRM-CM5_rcp85_r1i1p1_MOHC-HadREM3-GA7-05_v2_day_'

# Output filenames
RESULTS_FOLDER = '/data/cip19ql/rail_psgr_tasmax/data_output/climate/curves/'

EXP_E_FAILURE = '/data/cip19ql/rail_psgr_tasmax/data_output/climate/exp_e_fail.txt'
LOG  = '/data/cip19ql/rail_psgr_tasmax/data_output/climate/simulation_log.txt'

####################
# DEFINE FUNCTIONS #
####################

def main(date, date_range):
    
    print(date_range)
    df_e = dp.df_e
    recover_p = dp.recover_p

    #fragile function 
    mean = 35 + 273
    sd = 2.5  #about 95% of the values lie within two standard deviations, so 2*std=5deg,

    #load climate data 
    #date_range = '20960101-21001230'
    nc_path = NC_FOLDER + NC_DATASET_NAME + date_range + '.nc'
    ncfile = netCDF4.Dataset(nc_path, 'r')
    y0=date_range[:4]



    #Random sample N times 
    tasmax=[ncfile['tasmax'][date][cell[0]][cell[1]] for cell in df_e.nc_cells]
    p_fail=[stats.norm.cdf(temp,mean,sd) for temp in tasmax]
    N_sample=[(np.random.uniform(size=N) < p) * 1 for p in p_fail] #0-functioning, 1-fail

    if sum(N_sample).sum()==0:
        file_log= open(LOG, "a")
        content = y0  + '_d{}'.format(date)+ ', Number of failure = 0' + '\n'
        file_log.write(content)
        file_log.close()
        exit()

    t_step_cutoff = math.floor(4*( 1- math.log(sum(p_fail))/math.log(1-recover_p) ))

    if t_step_cutoff<1:
        file_log= open(LOG, "a")
        content = y0  + '_d{}'.format(date)+ 't_step_cutoff = {}'.format(t_step_cutoff) + '\n'
        file_log.write(content)
        file_log.close()
        
    else: 
        exp=sum(p_fail)
        file_exp_e_fail = open(EXP_E_FAILURE, "a")
        content = y0  + '_d{}'.format(date) + ', {}'.format(exp) + '\n'
        file_exp_e_fail.write(content)
        file_exp_e_fail.close()
        
        
        path1 = RESULTS_FOLDER + y0  + '_d{}'.format(date) + '.hdf5'

        f1 = h5py.File(path1,'a')
        
        #path1: to be stored: tasmax, p_fail, N_samples, 
        tasmax = np.array(tasmax)
        p_fail = np.array(p_fail)
        N_sample = np.array(N_sample)

        dset_tasmax = f1.create_dataset('tasmax',shape=tasmax.shape, data=tasmax)
        dset_p = f1.create_dataset('p_fail', shape=p_fail.shape, data=p_fail)
        dset_N_sample = f1.create_dataset('N_sample', shape=N_sample.shape, data=N_sample, compression="gzip", compression_opts=9)
        dset_curve = f1.create_dataset('curve', shape=tuple([N,t_step_cutoff]), data=None, compression="gzip", compression_opts=9)

        f1.close()

        for i in range(N):
            a_scenario=[item[i] for item in N_sample]
            if sum(a_scenario) != 0:
                arr_curve = curve_generator(a_scenario,t_step_cutoff)
                f1 = h5py.File(path1,'a')
                f1['curve'][i,:] = arr_curve
                f1.close()
            else:
                pass
        

 


################
#     Main     #
################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-i', '--index', type=int, help='index (bypasses manual date year setting)')

    parser.add_argument(
        '-d', '--date', type=int, help='this is the date parameter')
    
    parser.add_argument(
        '-y', '--date_range', type=str, help='year parameter in the format of 20960101-21001230/1')
    args = parser.parse_args()


    if args.index is None:
        if args.date is None:
            args.date = 9999
        
        if args.date_range is None:
            args.date_range='2000'
    else:
        args.date, args.date_range = days.combi[args.index]

    main(args.date, args.date_range)


