# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 11:43:54 2019

@author: storm
"""

import os
import gc
import numpy as np
import pandas as pd

import json
import math
import sys
import time
from datetime import datetime
from typing import Tuple, List

import cv2
import matplotlib.pyplot as plt
import sklearn.metrics
from PIL import Image

from matplotlib.axes import Axes
from matplotlib import animation, rc
import plotly.graph_objs as go
import plotly.tools as tls
from plotly.offline import plot, init_notebook_mode
import plotly.figure_factory as ff

init_notebook_mode(connected=True)

import seaborn as sns
from pyquaternion import Quaternion
from tqdm import tqdm

from lyft_dataset_sdk.utils.map_mask import MapMask
from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.geometry_utils import view_points, box_in_image, BoxVisibility
from lyft_dataset_sdk.utils.geometry_utils import view_points, transform_matrix
from pathlib import Path

import struct
from abc import ABC, abstractmethod
from functools import reduce
from typing import Tuple, List, Dict
import copy

def render_scene(index):
    my_scene = lyft_dataset.scene[index]
    my_sample_token = my_scene["first_sample_token"]
    lyft_dataset.render_sample(my_sample_token)


DATA_PATH =r'C:\Users\storm\Documents\3d-object-detection-for-autonomous-vehicles'+os.sep
#train = pd.read_csv(DATA_PATH + 'train.csv')
#sample_submission = pd.read_csv(DATA_PATH + 'sample_submission.csv')
#lyft_dataset = LyftDataset(data_path=DATA_PATH, json_path=DATA_PATH+'train_data')
#
#object_columns = ['sample_id', 'object_id', 'center_x', 'center_y', 'center_z',
#                  'width', 'length', 'height', 'yaw', 'class_name']
#objects = []
#for sample_id, ps in tqdm(train.values[:]):
#    object_params = ps.split()
#    n_objects = len(object_params)
#    for i in range(n_objects // 8):
#        x, y, z, w, l, h, yaw, c = tuple(object_params[i * 8: (i + 1) * 8])
#        objects.append([sample_id, i, x, y, z, w, l, h, yaw, c])
#        
#train_objects = pd.DataFrame(objects,columns = object_columns)
#
#numerical_cols = ['object_id', 'center_x', 'center_y', 'center_z', 'width', 'length', 'height', 'yaw']
#train_objects[numerical_cols] = np.float32(train_objects[numerical_cols].values)
#
#train_objects.head()

my_scene=lyft_dataset.scene[0]

my_sample_token = my_scene["first_sample_token"]
my_sample = lyft_dataset.get('sample', my_sample_token)

lyft_dataset.render_pointcloud_in_image(sample_token = my_sample["token"],
                                        dot_size = 1,
                                        camera_channel = 'CAM_FRONT')








































































































































