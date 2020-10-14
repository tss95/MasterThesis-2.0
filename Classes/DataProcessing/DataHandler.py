import numpy as np
import pandas as pd
import json
import h5py
import matplotlib.pyplot as plt
from obspy import Stream, Trace, UTCDateTime
import os
import csv
import seaborn as sns
import time
import tables
import random

import tensorflow as tf
from LoadData import LoadData

class DataHandler(LoadData):
    
    def __init__(self, csv_root):
        super().__init__(csv_root)
        self.label_dict = {'explosion':0, 'earthquake':1, 'noise':2, 'induced':3}
        
    def get_trace_shape_no_cast(self, ds):
        num_ds = len(ds)
        with h5py.File(ds[0][0], 'r') as dp:
            trace_shape = dp.get('traces').shape
        return num_ds, trace_shape[0], trace_shape[1]

    def path_to_trace(self, path):
        trace_array = np.empty((3,6001))
        with h5py.File(path, 'r') as dp:
            trace_array[:3] = dp.get('traces')
            info = np.array(dp.get('event_info'))
            info = json.loads(str(info))
        return trace_array, info
    
    def convert_to_tensor(self, value, dtype_hint = None, name = None):
        tensor = tf.convert_to_tensor(value, dtype_hint, name)
        return tensor
    
    def detrend_trace(self, trace):
        trace_BHE = Trace(data=trace[0])
        trace_BHN = Trace(data=trace[1])
        trace_BHZ = Trace(data=trace[2])
        stream = Stream([trace_BHE, trace_BHN, trace_BHZ])
        stream.detrend('demean')
        return np.array(stream)
    
    def highpass_filter(self, trace, highpass_freq):
        trace_BHE = Trace(data=trace[0])
        trace_BHN = Trace(data=trace[1])
        trace_BHZ = Trace(data=trace[2])
        stream = Stream([trace_BHE, trace_BHN, trace_BHZ])
        stream.taper(max_percentage=0.05, type='cosine')
        stream.filter('highpass', freq = highpass_freq)
        return np.array(stream)
    
    def get_class_array(self, ds, num_classes = 3):
        class_array = np.zeros((len(path),num_classes))
        for idx, path, label in enumerate(ds):
            if label == "explosion":
                class_array[idx][0] = 1
            if label == "earthquake":
                class_array[idx][1] = 1
            if label == "noise":
                class_array[idx][2] = 1
        return class_array