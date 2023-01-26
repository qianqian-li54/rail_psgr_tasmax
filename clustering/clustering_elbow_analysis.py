'''
    Elbow analysis for the k-mean method 
'''

# Standard library imports
import time
import numpy as np

# Related third party imports
import pickle
import netCDF4
import multiprocessing
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.cluster import BisectingKMeans
from sklearn.metrics import silhouette_score
from sklearn import metrics
from scipy.spatial.distance import cdist
from kneed import KneeLocator


# Local application/library specific imports


# Input filenames and directory names.
NC_GRID_RAIL = '/data/cip19ql/rail_psgr_tasmax/data_input/nc_grid_rail.pkl'
NC_FOLDER = '/data/cip19ql/input_tasmax/'

# Output filenames
SLICED_WEATHER_DATA = '/data/cip19ql/rail_psgr_tasmax/data_output/clustering/sliced_weather_data.pkl'
CLUSTERED_SUMMER_DAYS = '/data/cip19ql/rail_psgr_tasmax/data_output/clustering/clustered_summer_days.pkl'

ELBOW_ANALYSIS = '/data/cip19ql/rail_psgr_tasmax/data_output/clustering/elbow_analysis.pkl'
ELBOW_ANALYSIS_TXT = '/data/cip19ql/rail_psgr_tasmax/data_output/clustering/elbow_analysis.txt'


# Constancts 
DATE_RANGES=['{}'.format(i) + '0101-' + '{}'.format(i+4) + '1231' for i in range(2006,2100,5)]
DATE_RANGES[-1]='20960101-21001230'
NC_DATASET_NAME = 'tasmax_EUR-11_CNRM-CERFACS-CNRM-CM5_rcp85_r1i1p1_MOHC-HadREM3-GA7-05_v2_day_'



'''
######################
# DEFINE FUNCTION(S) #
######################

def slice_weather_data(date_range):
    # Read the climate data file (NetCDF)
    nc_file_path = NC_FOLDER + NC_DATASET_NAME + date_range + '.nc'
    ncfile = netCDF4.Dataset(nc_file_path, 'r')

    # Find summer days: May-Sept
    converted_dates = netCDF4.num2date(ncfile['time'],ncfile['time'].units, calendar=ncfile['time'].calendar,
                                    only_use_cftime_datetimes=True, only_use_python_datetimes=False)
    year_0 = converted_dates[0].year
    num_may2sep = []
    for i in range(5):
        all_dates_of_year_i = [date_ for date_ in converted_dates if date_.year == year_0+i ]
        dates_may2sep_year_i = [date_ for date_ in all_dates_of_year_i if date_.month in [5,6,7,8,9]]
        num_may2sep_year_i = netCDF4.date2index(dates_may2sep_year_i,ncfile['time'],calendar=ncfile['time'].calendar)
        num_may2sep.extend(list(num_may2sep_year_i))


    # Construct the data
    nc_grids_rail = pickle.load(open(NC_GRID_RAIL, 'rb'))
    X = np.zeros((len(num_may2sep),len(nc_grids_rail)))

    for i in range(len(num_may2sep)):
        date_ = num_may2sep[i]
        for j in range(len(nc_grids_rail)):
            grid_lon_lat = nc_grids_rail[j]
            grid_tasmax = ncfile['tasmax'][date_][grid_lon_lat[0]][grid_lon_lat[1]]
            X[i,j] = grid_tasmax
            
    return X

num_cores = multiprocessing.cpu_count()
# t0 = time.time()
sliced_weather_data = Parallel(n_jobs=num_cores, backend='loky',
                               verbose=0)(delayed(slice_weather_data)(date_range) for date_range in DATE_RANGES)
# t1 = time.time()
# print('time_taken={}'.format(t1-t0))

with open(SLICED_WEATHER_DATA, 'wb') as f:
    pickle.dump(sliced_weather_data,f)
'''

with open(SLICED_WEATHER_DATA, 'rb') as f:
    sliced_weather_data = pickle.load(f)

K = list(range(2,21))
K.extend([25,30,35,40,50,60])
elbow_analysis = {}

for i in range(len(sliced_weather_data)):
    X = sliced_weather_data[i]
    
    # Clustering analysis
    distortions = {}
    inertias = {}
    sil_coeff = {}
    

    for k in K:
        
        # Building and fitting the model
        kmeanModel = BisectingKMeans(n_clusters=k)
        kmeanModel.fit(X)
        
        distortions[k] = sum(np.min(cdist(X, kmeanModel.cluster_centers_,
                                          'euclidean'), axis=1)) / X.shape[0]
        inertias[k] = kmeanModel.inertia_
        
        label_ = kmeanModel.labels_
        sil_coeff[k] = silhouette_score(X, label_, metric='euclidean')
        
        
    x1 = list(distortions.keys())
    y1 = list(distortions.values())
    kneedle1 = KneeLocator(x1, y1, S=1, curve="convex", direction="decreasing")
    opti_k_1 = tuple([kneedle1.knee, kneedle1.elbow])
    
    x2 = list(inertias.keys())
    y2 = list(inertias.values())
    kneedle2 = KneeLocator(x2,y2, S=1, curve="convex", direction="decreasing")
    opti_k_2 = tuple([kneedle2.knee, kneedle2.elbow])
    
    elbow_analysis[i] = {'distortions': distortions, 
                         'inertias':inertias, 
                         'sil_coeff':sil_coeff, 
                         'opti_k_1': opti_k_1, 
                         'opti_k_2':opti_k_2}

print([elbow_analysis[item]['opti_k_2'] for item in range(19)])

with open(ELBOW_ANALYSIS, 'wb') as f:
    pickle.dump(elbow_analysis, f)
    
    
# try:
#     txt_file = open(ELBOW_ANALYSIS_TXT, 'a')
#     txt_file.write(distortions+ '\n')
#     txt_file.write(inertias+ '\n')
#     txt_file.write(sil_coeff+ '\n')
    
#     txt_file.close()
  
# except:
#     print("Unable to append to file")

################
#     Main     #
################
