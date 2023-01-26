'''
    Clustering analysis - Hierarchical
'''

# Standard library imports
import time
import numpy as np
from scipy.spatial.distance import euclidean 

# Related third party imports
import pickle
import netCDF4
import multiprocessing
from joblib import Parallel, delayed
from scipy.spatial import KDTree
from sklearn.cluster import KMeans
from sklearn.cluster import BisectingKMeans
from sklearn.cluster import AgglomerativeClustering




# Input filenames and directory names.
NC_GRID_RAIL = '/data/cip19ql/rail_psgr_tasmax/data_input/nc_grid_rail.pkl'
NC_FOLDER = '/data/cip19ql/input_tasmax/'

# Output filenames
CLUSTERING_CEN_DAYS = '/data/cip19ql/rail_psgr_tasmax/data_output/clustering/clustering_cen_days_HC.pkl'
CLUSTERING_VIS = '/data/cip19ql/rail_psgr_tasmax/data_output/clustering/clustering_vis_HC.pkl'

# Constancts 
DATE_RANGES=['{}'.format(i) + '0101-' + '{}'.format(i+4) + '1231' for i in range(2006,2100,5)]
DATE_RANGES[-1]='20960101-21001230'
NC_DATASET_NAME = 'tasmax_EUR-11_CNRM-CERFACS-CNRM-CM5_rcp85_r1i1p1_MOHC-HadREM3-GA7-05_v2_day_'
N_CLUSTERS = 10


######################
# DEFINE FUNCTION(S) #
######################

def cluster_summer_days(date_range):
    # t0 = time.time()
    
    # 1) READ TEH CLIMATE DATA FILE (NetCDF)
    nc_file_path = NC_FOLDER + NC_DATASET_NAME + date_range + '.nc'
    ncfile = netCDF4.Dataset(nc_file_path, 'r')

    # 2) FIND SUMMER DAYS (MAY-SEPT)
    # numb(index) to dates
    converted_dates = netCDF4.num2date(ncfile['time'],ncfile['time'].units, calendar=ncfile['time'].calendar,
                                    only_use_cftime_datetimes=True, only_use_python_datetimes=False)
    year_0 = converted_dates[0].year
    # dates to num(index)
    num_may2sep = []
    for i in range(5):
        all_dates_of_year_i = [date_ for date_ in converted_dates if date_.year == year_0+i ]
        dates_may2sep_year_i = [date_ for date_ in all_dates_of_year_i if date_.month in [5,6,7,8,9]]
        num_may2sep_year_i = netCDF4.date2index(dates_may2sep_year_i,ncfile['time'],calendar=ncfile['time'].calendar)
        num_may2sep.extend(list(num_may2sep_year_i))

    # t1 = time.time()
    
    # 3) CONSTRUCT THE DATA
    nc_grids_rail = pickle.load(open(NC_GRID_RAIL, 'rb'))
    X = np.zeros((len(num_may2sep),len(nc_grids_rail)))

    for i in range(len(num_may2sep)):
        date_ = num_may2sep[i]
        for j in range(len(nc_grids_rail)):
            grid_lon_lat = nc_grids_rail[j]
            grid_tasmax = ncfile['tasmax'][date_][grid_lon_lat[0]][grid_lon_lat[1]]
            X[i,j] = grid_tasmax
    num_may2sep = np.array(num_may2sep)    
        
    # t2  = time.time()
    
    # 4) CLUSTERING ANALYSIS
    HCModel = AgglomerativeClustering(n_clusters=N_CLUSTERS,  linkage='ward')
    HCModel.fit(X)
    # Labels of each point
    labels_x = HCModel.labels_
    # Coordinates of cluster centers / algebraic mean
    # c_centers = HCModel.cluster_centers_

    # t3 = time.time()
    
    # FIND TEH CENTROID DAY IN EACH CLUSTER
    cen_days = {}
    vis_info = {}
    for i in range(N_CLUSTERS):
        # get the indices of the points belongs to cluster_i (out of the 5*153 days)
        c_indices = np.where(labels_x==i)[0]
        # get the num(date) of teh points belongs to cluster_i (out of the 5*365 days)
        c_num = num_may2sep[c_indices]

        # get the weather data for the points belongs to cluster_i
        c_points = X[c_indices]
        
        # get the cluster center / algrbraic mean
        c_center = np.mean(c_points, axis=0)
        # build the KDTree
        kdt=KDTree(c_points)
        # find the centroid day - the point closest to the cluster center 
        nearest_point_distance, nearest_point_index = kdt.query(c_center)
        
        # inertia 
        inertia_ = np.sum([euclidean(item, c_center) for item in c_points])
        
        # get the combination of date_range and num (date) 
        cen_day_idx = tuple([date_range, c_num[nearest_point_index]])

        # get other information for the visualisation of the clustering analysis 
        # 1) national avg_temp of the centroid day  
        # 2) number of days in the the cluster 
        vis_ = {'cen_avg_temp': np.average(X[c_indices[nearest_point_index]]), 
                'num_days': len(c_indices),
                'inertia': inertia_}

        cen_days[i] = cen_day_idx
        vis_info[i] = vis_

    # t4 = time.time()
    # print(t1-t0, t2-t1, t3-t2, t4-t3)
    print(date_range)
    return [cen_days, vis_info]



num_cores = multiprocessing.cpu_count()
print(num_cores)
results = Parallel(n_jobs=num_cores, backend='loky',
                               verbose=0)(delayed(cluster_summer_days)(date_range) for date_range in DATE_RANGES)

results_cen_days = [item[0] for item in results]
results_vis_info = [item[1] for item in results]

with open(CLUSTERING_CEN_DAYS, 'wb') as f:
    pickle.dump(results_cen_days,f)

with open(CLUSTERING_VIS, 'wb') as f:
    pickle.dump(results_vis_info,f)


