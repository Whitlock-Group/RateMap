import scipy.io
import pickle
import numpy as np
from Archived.RateMap.ratemaps import *


file1 = '/Users/jingyig/Work/Kavli/Data/TriTask/Leia/gui_processed/leia_chasing1_reheaded.mat'
file2 = '/Users/jingyig/Work/Kavli/Data/TriTask/Leia/gui_processed/leia_chasing2_reheaded.mat'
file3 = '/Users/jingyig/Work/Kavli/Data/TriTask/Leia/gui_processed/leia_chasing3_notreheaded.mat'


# data1 = scipy.io.loadmat('/Users/jingyig/Work/Kavli/Data/TriTask/Leia/gui_processed/leia_social1_rec_reheaded.mat')
# data2 = scipy.io.loadmat('/Users/jingyig/Work/Kavli/Data/TriTask/Leia/gui_processed/leia_social2_rec_notreheaded.mat')
# data3 = scipy.io.loadmat('/Users/jingyig/Work/Kavli/Data/TriTask/Leia/gui_processed/leia_social3_rec_reheaded.mat')
#
# data1.keys()
# ts = np.ravel(data1['trackingTS'])
# data1['sessionTS']
# data1['pointdatadimensions'][0][2]
# frame_rate = data1['pointdatadimensions'][0][2] / (ts[1] - ts[0])
#
# data2['trackingTS']
# data2['sessionTS']
#
# data3['trackingTS']
# data3['sessionTS']

file_list = [file1, file2, file3]


m_data = merge_sessions(file_list, file_info='leia_chasing_merge')

pre_data = data_generator(m_data)

reward_marker = pre_data['matrix_data'][0]['sorted_point_data'][:, 10, :]
neck_marker = pre_data['matrix_data'][0]['sorted_point_data'][:, 4, :]
vec2 = reward_marker - neck_marker
vec2 = vec2[:, :2]
vec1 = pre_data['matrix_data'][0]['allo_head_rotm'][:, 0, :2]


marker1 = pre_data['matrix_data'][0]['sorted_point_data'][:, 10, :]
marker2 = pre_data['matrix_data'][0]['sorted_point_data'][:, 4, :]

angs = np.zeros(len(vec1))
dists = np.zeros(len(vec1))

for t in range(len(vec1)):
    angs[t] = get_related_head_angle(vec1[t], vec2[t])
    dists[t] = get_related_dist(marker1[t], marker2[t])

angs = angs * 180 / np.pi
dists = dists * 100


include_factor = {'factor_name': ['S Relative_head_angle', 'T Relative distance'],
                  'factor': [angs, dists],
                  'bounds': [[-180, 180, True], [0, 0, False]],
                  'x_axis': ['angles', 'cm'],
                  'animal_id': [1, 1]}

pre_rm_data = get_rm_pre_data(pre_data, use_even_odd_minutes=True, speed_type='jump', window_size=250,
                              include_factor=include_factor, derivatives_param=(10, 10), boundary=None,
                              filter_by_speed=None, filter_by_spatial=None, filter_by_factor=None,
                              num_bins_1d=36, occupancy_thresh_1d=0.4, save_data=True)

infile = open('rm_pre_data_leia_chasing_merge_XYZeuler_notrickseo.pkl', 'rb')
pre_rm_data = pickle.load(infile)
infile.close()

extra2d = {'W R_H_and_R_Dist': ['S Relative_head_angle', 'T Relative distance']}


ratemap_generator(pre_rm_data, cell_index=(100,), temp_offsets=0, n_bins_1d=36, smoothing_par_1d=1,
                  occupancy_thresh_1d=0.4, occupancy_thresh_2d=0.4, smoothing_par_2d=(1.15, 1.15),
                  n_shuffles=10, shuffle_range=(15, 60), use_time_bins=True, use_quantile=True, periodic=True,
                  velocity_par=(60, 18, 20), spatial_par=30, self_motion_par=(0, 80, -5, 80, 3),
                  include_derivatives=True, extra_2d_tasks=extra2d, comparing=False, pie_style=True,
                  color_map='jet', pl_subplot=(14, 10), pl_size=(70, 70), seeds=None, debug_mode=False)



