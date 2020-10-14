import numpy as np
import pandas as pd
import json
import h5py
import sklearn as sk
import matplotlib.pyplot as plt
from obspy import Stream, Trace, UTCDateTime
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os
import csv
import pylab as pl
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import keras
from keras.layers import Activation, Conv1D, Dense, Dropout, Flatten, MaxPooling3D, BatchNormalization, InputLayer, LSTM
from keras.layers import Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.utils import Sequence
from keras.optimizers import Adam
from tensorflow.keras import regularizers
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
import datetime
import re
from sklearn.metrics import confusion_matrix
from livelossplot import PlotLossesKeras
from CustomCallback import CustomCallback

class BaselineHelperFunctions():
    
    def plot_confusion_matrix(self, model, test_gen, test_ds, batch_size, train_ds = None, train_testing = False):
        if not train_testing:
            steps = len(test_ds)/batch_size
            predictions = model.predict_generator(test_gen, steps)
            predicted_classes = self.convert_to_class(predictions)[0:(len(test_ds))]
            true_classes = self.get_class_array(test_ds, 3)
            labels = ['explosion', 'earthquake', 'noise']
            cm = confusion_matrix(true_classes.argmax(axis=1), predicted_classes.argmax(axis=1))
            print(cm)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            cax = ax.matshow(cm)
            plt.title('Confusion matrix of the classifier')
            fig.colorbar(cax)
            ax.set_xticklabels([''] + labels)
            ax.set_yticklabels([''] + labels)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.show()
            return
        else:
            steps = len(train_ds)/batch_size
            predictions = model.predict_generator(train_gen, steps)
            predicted_classes = self.convert_to_class(predictions)[0:(len(train_ds))]
            true_classes = self.get_class_array(train_ds, 3)
            labels = ['explosion', 'earthquake', 'noise']
            cm = confusion_matrix(true_classes.argmax(axis=1), predicted_classes.argmax(axis=1))
            print(cm)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            cax = ax.matshow(cm)
            plt.title('Confusion matrix of the classifier')
            fig.colorbar(cax)
            ax.set_xticklabels([''] + labels)
            ax.set_yticklabels([''] + labels)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.show()
            return
        
    def get_steps_per_epoch(self, gen_set, batch_size, test):
        if test:
            if (int(0.05*len(gen_set)/batch_size) == 0):
                print("oof")
                return 1
            return int(0.05*len(gen_set)/batch_size)
        return len(gen_set)/batch_size
    
    def get_class_array(self, ds, num_classes = 3):
        class_array = np.zeros((len(ds),num_classes))
        for idx, path_and_label in enumerate(ds):
            if path_and_label[1] == "explosion":
                class_array[idx][0] = 1
            elif path_and_label[1] == "earthquake":
                class_array[idx][1] = 1
            elif path_and_label[1] == "noise":
                class_array[idx][2] = 1
            else:
                print(f"No class available: {path_and_label[1]}")
                break
        return class_array

    def convert_to_class(self, predictions):
        predicted_classes = np.zeros((predictions.shape))
        for idx, prediction in enumerate(predictions):
            highest_pred = max(prediction)
            highest_pred_index = np.where(prediction == highest_pred)
            predicted_classes[idx][highest_pred_index] = 1
        return predicted_classes
    
    def get_class_distribution_from_csv(self,data_csv):
        with open(data_csv) as file:
            nr_earthquakes = 0
            nr_explosions = 0
            nr_noise = 0
            nr_total = 0
            for row in file:
                event_type = row.split(',')[1].rstrip()
                if event_type == "earthquake":
                    nr_earthquakes += 1
                elif event_type == "explosion":
                    nr_explosions += 1
                elif event_type == "noise":
                    nr_noise += 1
                nr_total += 1

            return nr_earthquakes, nr_explosions, nr_noise, nr_total
        
    def batch_class_distribution(self, batch):
        batch_size, nr_classes = batch[1].shape
        class_distribution = np.zeros((1,nr_classes))[0]
        print(class_distribution)
        for sample in batch[1]:
            for idx, i in enumerate(sample):
                if i == 1:
                    class_distribution[idx] += 1
        return class_distribution
    
    def get_trace_shape_no_cast(self, ds):
        num_ds = len(ds)
        with h5py.File(ds[0][0], 'r') as dp:
            trace_shape = dp.get('traces').shape
        return num_ds, trace_shape[0], trace_shape[1]
    
    def generate_build_model_args(self, model_nr, batch_size, dropout_rate, activation, output_layer_activation, l2_r, l1_r, 
                                  start_neurons, filters, kernel_size, padding, 
                                  num_classes = 3, channels = 3, timesteps = 6001):
        return {"model_nr" : model_nr,
                "input_shape" : (batch_size, channels, timesteps),
                "num_classes" : num_classes,
                "dropout_rate" : dropout_rate,
                "activation" : activation,
                "output_layer_activation" : output_layer_activation,
                "l2_r" : l2_r,
                "l1_r" : l1_r,
                "full_regularizer" : True,
                "start_neurons" : start_neurons,
                "filters" : filters,
                "kernel_size" : kernel_size,
                "padding" : padding}
    
    def generate_model_compile_args(self, opt):
         return {"loss" : "categorical_crossentropy",
                      "optimizer" : opt,
                      "metrics" : ["accuracy",
                                   tf.keras.metrics.Precision(thresholds=None, top_k=None, class_id=None, name=None, dtype=None),
                                   tf.keras.metrics.Recall(thresholds=None, top_k=None, class_id=None, name=None, dtype=None)]}
    def generate_gen_args(self, batch_size, test, detrend, useScaler = False, scaler = None, num_classes = 3):
        return {"batch_size" : batch_size,
                    "shuffle" : True,
                    "test" : test,
                    "detrend" : detrend,
                    "useScaler" : useScaler,
                    "scaler" : scaler,
                    "num_classes" : num_classes}
    
    def generate_fit_args(self, train_ds, val_ds, batch_size, test, epoch, val_gen, use_tensorboard = False):
        callbacks = [PlotLossesKeras()]
        if use_tensorboard:
            log_dir = "tensorboard_dir/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
            callbacks = [tensorboard_callback]
        
        return {"steps_per_epoch" : self.get_steps_per_epoch(train_ds, batch_size, test),
                        "epochs" : epoch,
                        "validation_data" : val_gen,
                        "validation_steps" : self.get_steps_per_epoch(val_ds, batch_size, test),
                        "verbose" : 1,
                        "use_multiprocessing" : False, 
                        "workers" : 1,
                        "callbacks" : callbacks
                       }
    
    def get_n_points_with_highest_training_loss(self, train_ds, n, full_logs):
        train_ds_dict = {}
        for path, label in train_ds:
            train_ds_dict[path] = {'label' : label,
                                   'loss': 0,
                                   'average_loss' : 0,
                                   'occurances' : 0}
        counter = 0
        for batch in full_logs:
            loss = batch['loss']
            for path_class in batch['batch_samples']:
                train_ds_dict[path_class[0]]['loss'] += loss
                train_ds_dict[path_class[0]]['occurances'] += 1

        train_ds_list = []
        for sample in np.array(train_ds[:,0]):
            if train_ds_dict[sample]['occurances'] == 0:
                continue
            train_ds_dict[sample]['average_loss'] = train_ds_dict[sample]['loss'] / train_ds_dict[sample]['occurances']
            train_ds_list.append((sample, train_ds_dict[sample]['label'],train_ds_dict[sample]['average_loss']))

        sorted_train_ds_list = sorted(train_ds_list, key=lambda x: x[2], reverse = True)


        return sorted_train_ds_list[0:n]