import numpy as np
from keras.utils import np_utils
import random

from .LoadData import LoadData
from .DataHandler import DataHandler


class NoiseAugmentor(DataHandler):
    # TODO: Consider this: https://stackoverflow.com/questions/47324756/attributeerror-module-matplotlib-has-no-attribute-plot
    # How does SNR impact the use of this class???
    def __init__(self, ds):
        super(self).__init__()
        self.ds = ds
        self.noise_ds = get_noise(self.ds)
        self.noise_std = get_noise_ds(self.noise_ds)
        
    def create_noise(self, mean, std, sample_shape):
        noise = np.random.normal(mean, std, (sample_shape))
        return noise
    
    def batch_augment_noise(self, X, mean, std):
        noise = create_noise(mean, std, X.shape)
        X = X + noise
        return X
    
    def get_noise(self, ds):
        noise_ds = []
        for path, label in ds:
            if label == "noise":
                noise_ds.append([path,label])
        return np.array(noise_ds)

    def get_mean_noise(self, noise_ds):
        noise_total = 0
        for path, label in noise_ds:
            noise_total += np.mean(self.path_to_trace(path)[0])
        return noise_total / len(noise_ds)

    def get_noise_std(self, noise_ds):
        total_std = 0
        for path, label in noise_ds:
            total_std += np.std(self.path_to_trace(path)[0])
        return total_std / len(noise_ds)
