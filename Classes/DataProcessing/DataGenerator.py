import numpy as np
from keras.utils import np_utils

from DataProcessing.LoadData import LoadData
from DataProcessing.DataHandler import DataHandler

class DataGenerator(DataHandler):
    
    def __init__(self, csv_root):
        super().__init__(csv_root)
   
   
    def data_generator(self, ds, batch_size, test = False, detrend = False, num_classes = 3, useScaler = False, scaler = None, use_highpass = False, highpass_freq = 0.49):
        num_samples, channels, timesteps = self.get_trace_shape_no_cast(ds)
        num_samples = len(ds)
        if test:
            num_samples = int(num_samples * 0.1)
        while True:
            for offset in range(0, num_samples, batch_size):
                # Get the samples you'll use in this batch
                self.batch_samples = np.empty((batch_size,2), dtype = np.ndarray)
                
                if offset+batch_size > num_samples and not test:
                    overflow = offset + batch_size - num_samples
                    self.batch_samples[0:batch_size-overflow] = ds[offset:offset+batch_size]
                    i_start = random.randint(0, num_samples-overflow)
                    self.batch_samples[batch_size-overflow:batch_size] = ds[i_start:i_start+overflow]           
                else:
                    self.batch_samples = ds[offset:offset+batch_size]
                    
                # Initialize X and y arrays for this batch
                X = np.empty((batch_size, channels, timesteps))
                y = np.empty((batch_size))
                for idx, batch_sample in enumerate(self.batch_samples):
                    # Load trace
                    if detrend:
                        X[idx] = self.detrend_trace(self.path_to_trace(batch_sample[0])[0])
                    else:
                        X[idx] = self.path_to_trace(batch_sample[0])[0]
                        
                    # Read label:
                    y[idx] = self.label_dict.get(batch_sample[1])
                    
                    # Scale data
                    if useScaler:
                        X[idx] = scaler.transform(X[idx])
                    if use_highpass:
                        X[idx] = self.highpass_filter(X[idx], highpass_freq)
                
                try:
                    y = np_utils.to_categorical(y, num_classes, dtype=np.int64)
                except:
                    raise Exception(f'Error when doing to_categorical. Inputs are y: {y} and num_classes: {num_classes}')               
                yield X, y

    
            