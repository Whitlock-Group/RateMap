import scipy.io
import pickle
import numpy as np
from Archived.RateMap.ratemaps import *


file1 = '/Users/jingyig/Work/Kavli/Data/TriTask/Leia/gui_processed/leia_social1_rec_reheaded.mat'
file2 = '/Users/jingyig/Work/Kavli/Data/TriTask/Leia/gui_processed/leia_social1_prop_reheaded.mat'

pmat = data_loader(file2)
rmat = data_loader(file1)


pdata = data_generator(pmat, head_angle_thresh=None)
rdata = data_generator(rmat, head_angle_thresh=None)

pre_data = merge_comparing_data(rdata, pdata)


# infile = open('test_combine_animal.pkl', 'wb')
# pickle.dump(pre_data, infile, protocol=pickle.HIGHEST_PROTOCOL)
# infile.close()

# --------------------------------------------------

infile = open('test_combine_animal.pkl', 'rb')
pre_data = pickle.load(infile)
infile.close()

vec2 = pre_data['matrix_data'][1]['allo_head_rotm'][:, 0, :2]
vec1 = pre_data['matrix_data'][0]['allo_head_rotm'][:, 0, :2]

neck1 = pre_data['matrix_data'][0]['sorted_point_data'][:, 4, :2]
neck2 = pre_data['matrix_data'][1]['sorted_point_data'][:, 4, :2]

head_vec = pre_data['matrix_data'][0]['allo_head_rotm'][:, 0, :2]
neck_vec = neck2 - neck1


angs = np.zeros(len(vec1))
angs[:] = np.nan
dists = np.zeros(len(vec1))
dists[:] = np.nan
looking_angs = np.zeros(len(vec1))
looking_angs[:] = np.nan

for t in range(len(vec1)):
    angs[t] = get_related_head_angle(vec1[t], vec2[t])
    dists[t] = get_related_dist(neck1[t], neck2[t])
    looking_angs[t] = get_related_head_angle(head_vec[t], neck_vec[t])

angs = angs * 180 / np.pi
dists = dists * 100
looking_angs = looking_angs * 180 / np.pi


pelev = relative_neck_elevation(pre_data['matrix_data'][1]['sorted_point_data'])
relev = relative_neck_elevation(pre_data['matrix_data'][0]['sorted_point_data'])

pelev = pelev * 100
relev = relev * 100


include_factor = {'factor_name': ('S Relative_head_angle', 'T Relative distance', 'U Adj_neck_elev', 'U Adj_neck_elev'),
                  'factor': (angs, dists, relev, pelev),
                  'bounds': [[-180, 180, True], [0, 0, False], [0, 0, False], [0, 0, False]],
                  'x_axis': ('angles', 'cm', 'cm', 'cm'),
                  'animal_id': (1, 1, 1, 2)}

mat_data = get_rm_pre_data(pre_data, use_even_odd_minutes=True, speed_type='jump', window_size=250,
                           include_factor=include_factor, derivatives_param=(10, 10), boundary=None,
                           filter_by_speed=None, filter_by_spatial=None, filter_by_factor=None,
                           num_bins_1d=36, occupancy_thresh_1d=0.4, save_data=True)

infile = open('rm_pre_data_leia_social1_reheaded_XYZeuler_notricks_eo.pkl', 'rb')
mat_data = pickle.load(infile)
infile.close()

extra2d = {'W R_H_and_R_Dist': ['S Relative_head_angle', 'T Relative distance']}


ratemap_generator(mat_data, cell_index=(10,), temp_offsets=0, n_bins_1d=36, smoothing_par_1d=1,
                  occupancy_thresh_1d=0.4, occupancy_thresh_2d=0.4, smoothing_par_2d=(1.15, 1.15),
                  n_shuffles=10, shuffle_range=(15, 60), use_time_bins=True, use_quantile=True, periodic=True,
                  velocity_par=(60, 18, 20), spatial_par=30, self_motion_par=(0, 80, -5, 80, 3),
                  include_derivatives=True, extra_2d_tasks=extra2d, comparing=False, pie_style=True,
                  color_map='jet', pl_subplot=(14, 10), pl_size=(70, 70), seeds=None, debug_mode=False)