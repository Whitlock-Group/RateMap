import scipy.io
import pickle
import numpy as np
from Archived.RateMap.ratemaps import *

boundaries = {'neck_elevation': (0., 0.36),
              'back_ang': ((-60, 60), (-60,60)),
              'opt_back_ang': ((-60, 60), (-60,60)),
              'speeds':((0,120),(0,120),(0,120),(0,120)),
              'selfmotion':((-104.02846857,83.66376106),
                            (-10.54837855,175.69859119),
                            (-119.35408181, 96.78457956),
                            (-30.42151711, 161.00896119),
                            (-104.41416926, 101.5806638),
                            (-6.96034067, 403.66958929),
                            (-115.14593404, 93.70091392),
                            (-30.26921306, 242.64589367)),
              'speeds_1st_der': ((-150, 150), (-150, 150), (-150, 150), (-150, 150)),
              'neck_1st_der': (-0.1, 0.1),
              'neck_2nd_der': (-0.8, 0.8),
              'allo_head_1st_der': ((-400, 400), (-300, 300), (-400, 400)),
              'allo_head_2nd_der': ((-4000, 4000), (-3000, 3000), (-4000, 4000)),
              'bodydir_1st_der':(-300, 300),
              'bodydir_2nd_der': (-1000, 1000),
              'ego3_head_1st_der': ((-400, 400), (-300, 300), (-400, 400)),
              'ego3_head_2nd_der': ((-4000, 4000), (-3000, 3000), (-4000, 4000)),
              'ego2_head_1st_der':((-400, 400), (-300, 300), (-400, 400)),
              'ego2_head_2nd_der': ((-4000, 4000), (-3000, 3000), (-4000, 4000)),
              'back_1st_der':((-100, 100), (-100, 100)),
              'back_2nd_der':((-1000, 1000), (-1000, 1000)),
              'opt_back_1st_der':((-100, 100), (-100, 100)),
              'opt_back_2nd_der':((-1000, 1000), (-1000, 1000)),
              'imu_allo_head_1st_der': ((-400, 400), (-300, 300), (-400, 400)),
              'imu_allo_head_2nd_der': ((-4000, 4000), (-3000, 3000), (-4000, 4000)),
              'imu_ego3_head_1st_der': ((-400, 400), (-300, 300), (-400, 400)),
              'imu_ego3_head_2nd_der': ((-4000, 4000), (-3000, 3000), (-4000, 4000)),
              'imu_ego2_head_1st_der': ((-400, 400), (-300, 300), (-400, 400)),
              'imu_ego2_head_2nd_der':  ((-4000, 4000), (-3000, 3000), (-4000, 4000))}

file = '/Users/jingyig/Work/Kavli/Data/Bartuce_Data/johnjohn/26471_johnjohn_220520_2130_intermediate_s2_dark/johnjohn_220520_s2_intermediate_dark_reheaded.mat'

mat = data_loader(file)

pre_data = data_generator(mat)

data = get_rm_pre_data(pre_data, use_even_odd_minutes=True, speed_type='jump', window_size=250,
                       include_factor=None, derivatives_param=(10, 10), boundary=boundaries, avoid_2nd=False,
                       filter_by_speed=None, filter_by_spatial=None, filter_by_factor=None,
                       num_bins_1d=36, occupancy_thresh_1d=0.4, save_data=True)


infile = open('rm_pre_data_johnjohn_220520_s2_intermediate_dark_reheaded_XYZeuler_notricks_eo.pkl', 'rb')
data = pickle.load(infile)
infile.close()


ratemap_generator(data, cell_index=(375,), temp_offsets=0, n_bins_1d=36, smoothing_par_1d=1,
                  occupancy_thresh_1d=0.4, occupancy_thresh_2d=0.4, smoothing_par_2d=(1.15, 1.15),
                  n_shuffles=20, shuffle_range=(15, 60), use_time_bins=False, use_quantile=True, periodic=True,
                  velocity_par=None, spatial_par=30, self_motion_par=(0, 80, -5, 80, 3),
                  include_derivatives=True, extra_2d_tasks=None, comparing=False, pie_style=True,
                  color_map='jet', pl_subplot=(14, 10), pl_size=(70, 70), limit_y=False, seeds=None, debug_mode=False)
