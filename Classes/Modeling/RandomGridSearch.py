import numpy as np
import pandas as pd
import h5py

import sklearn as sk
from sklearn.metrics import classification_report, confusion_matrix
from keras.callbacks import ModelCheckpoint

import keras

#from keras.layers import Activation, Conv1D, Dense, Dropout, Flatten, MaxPooling3D, BatchNormalization, InputLayer, LSTM
#from keras.layers import Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.losses import categorical_crossentropy
from keras.models import Sequential
#from keras.utils import Sequence
#from keras.optimizers import Adam
from tensorflow.keras import regularizers
from keras.utils import np_utils
#from keras.utils.vis_utils import plot_model
#from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import ParameterGrid
#import re
#from sklearn.metrics import confusion_matrix
#from sklearn.preprocessing import StandardScaler



import tensorflow as tf

from Modeling.Models import Models

from DataProcessing.BaselineHelperFunctions import BaselineHelperFunctions
from DataProcessing.LoadData import LoadData
from DataProcessing.DataGenerator import DataGenerator

from Scaling.MinMaxScalerFitter import MinMaxScalerFitter
from Scaling.StandardScalerFitter import StandardScalerFitter

from livelossplot import PlotLossesKeras
import random
import pprint
import re
import json

"""
Best so far:
{'batch_size': 16, 'epochs': 95, 'learning_rate': 0.1, 'optimizer': 'adam'},
 {'activation': 'tanh',
  'dropout_rate': 0.5,
  'l1_r': 0.1,
  'l2_r': 0.0001,
  'start_neurons': 128},
 {'loss': 358.6396789550781,
  'accuracy': 0.51602566242218018,
  'mse': 0.21034729480743408,
  'precision': 0.33800473809242249,
  'recall': 0.22719238698482513},
 {'train_loss': 358.9141845703125,
  'train_accuracy': 0.37990197539329529,
  'train_mse': 0.23922176659107208,
  'train_precision': 0.33883824944496155,
  'train_recall': 0.22711670398712158}
  
  Best training results:
  {'batch_size': 128,
  'epochs': 100,
  'learning_rate': 0.0001,
  'optimizer': 'rmsprop'},
 {'activation': 'tanh',
  'dropout_rate': 0.2,
  'l1_r': 0.2,
  'l2_r': 0.001,
  'start_neurons': 512},
 {'loss': 7.242753028869629,
  'accuracy': 0.3671875,
  'mse': 0.31585189700126648,
  'precision': 0.91042166948318481,
  'recall': 0.87738406658172607},
 {'train_loss': 4.252453804016113,
  'train_accuracy': 1.0,
  'train_mse': 0.00082400580868124962,
  'train_precision': 0.90749055147171021,
  'train_recall': 0.87465983629226685}
  """

class RandomGridSearch():
    hyper_grid = {
            "batch_size" : [8, 16, 32, 64, 128, 256],
            "epochs" : [50, 65, 70, 75, 80],
            "learning_rate" : [0.1, 0.01, 0.001, 0.0001, 0.00001],
            "optimizer" : ["adam", "rmsprop", "sgd"]
        }
    model_grid = {
        "start_neurons" : [2,4,8,16, 32, 64, 128, 256, 512],
        "dropout_rate" : [0.5, 0.4, 0.3, 0.2, 0.1, 0.01, 0],
        "filters" : [3, 9, 15, 21],
        "kernel_size" : [3],
        "padding" : ["same", "valid"],
        "l2_r" : [0.3, 0.2, 0.1, 0.01, 0.001, 0.0001],
        "l1_r" : [0.3, 0.2, 0.1, 0.01, 0.001, 0.0001],
        "activation" : ["relu", "sigmoid", "softmax", "tanh"],
        "output_layer_activation" : ["relu", "sigmoid", "softmax", "tanh"]
    }
    

    def __init__(self, train_ds, val_ds, test_ds, model_nr, test, detrend, useScaler, useMinMax, n_picks, 
                 hyper_grid=hyper_grid, model_grid=model_grid, num_classes = 3, use_tensorboard = False):
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.model_nr = model_nr
        self.test = test
        self.detrend = detrend
        self.useScaler = useScaler
        self.useMinMax = useMinMax
        self.n_picks = n_picks
        self.hyper_grid = hyper_grid
        self.model_grid = model_grid
        self.num_classes = num_classes
        self.use_tensorboard = use_tensorboard
        self.helper = BaselineHelperFunctions()
        self.csv_root = 'csv_folder_3_class'
        self.full_data_csv, self.train_csv, self.val_csv, self.test_csv = LoadData(self.csv_root).getData()
        self.data_gen = DataGenerator(self.csv_root, self.train_csv, self.val_csv, self.test_csv)
        

    def fit(self):
        #num_ds, channels, timesteps = data_gen.get_trace_shape_no_cast(train_ds)
        if self.useScaler:
            if self.useMinMax:
                self.scaler = MinMaxScalerFitter(self.train_ds).fit_scaler(shuffle = True, test = self.test, detrend = self.detrend)
            else:
                self.scaler = StandardScalerFitter(self.train_ds).fit_scaler(shuffle = True, test = self.test, detrend = self.detrend)
        self.delete_progress()
        self.hyper_picks = self.get_n_params_from_list(list(ParameterGrid(self.hyper_grid)), self.n_picks)
        self.model_picks = self.get_n_params_from_list(list(ParameterGrid(self.model_grid)), self.n_picks)
        self.results = []
        pp = pprint.PrettyPrinter(indent=4)
        for i in range(self.n_picks):
            model_info = {"model_nr" : self.model_nr, "index" : i}
            current_picks = [model_info, self.hyper_picks[i], self.model_picks[i]]
            self.save_params(current_picks)
            epoch = self.hyper_picks[i]["epochs"]
            batch_size = self.hyper_picks[i]["batch_size"]
            dropout_rate = self.model_picks[i]["dropout_rate"]
            activation = self.model_picks[i]["activation"]
            output_layer_activation = self.model_picks[i]["output_layer_activation"]
            l2_r = self.model_picks[i]["l2_r"]
            l1_r = self.model_picks[i]["l1_r"]
            start_neurons = self.model_picks[i]["start_neurons"]
            filters = self.model_picks[i]["filters"]
            kernel_size = self.model_picks[i]["kernel_size"]
            padding = self.model_picks[i]["padding"]
            opt = self.getOptimizer(self.hyper_picks[i]["optimizer"], self.hyper_picks[i]["learning_rate"])
            
            build_model_args = self.helper.generate_build_model_args(self.model_nr, batch_size, dropout_rate, 
                                                                     activation, output_layer_activation,
                                                                     l2_r, l1_r, start_neurons, filters, kernel_size, padding)
            model = Models(**build_model_args).model
            gen_args = self.helper.generate_gen_args(batch_size, self.test, self.detrend, useScaler = self.useScaler, scaler = self.scaler, num_classes = self.num_classes)
            train_gen = self.data_gen.data_generator(self.train_ds, **gen_args)
            val_gen = self.data_gen.data_generator(self.val_ds, **gen_args)
            test_gen = self.data_gen.data_generator(self.test_ds, **gen_args)
            
            model_compile_args = self.helper.generate_model_compile_args(opt)
            model.compile(**model_compile_args)
            
            print("Starting: ")
            pp.pprint(self.hyper_picks[i])
            print("---------------------------------------------------------------------------------")
            pp.pprint(self.model_picks[i])

            

            fit_args = self.helper.generate_fit_args(self.train_ds, self.val_ds, batch_size, self.test, 
                                                     epoch, val_gen, use_tensorboard = self.use_tensorboard)
            
            model_fit = model.fit(train_gen, **fit_args)
            loss, accuracy, precision, recall = model.evaluate_generator(
                generator=test_gen, steps=self.helper.get_steps_per_epoch(self.test_ds, batch_size, False))
            metrics = []
            metrics_test = {"test_loss" : loss,
                            "test_accuracy" : accuracy,
                            "test_precision": precision,
                            "test_recall" : recall}
            metrics.append(metrics_test)
            current_picks.append(metrics_test)
            train_loss, train_accuracy, train_precision, train_recall = model.evaluate_generator(
                generator=train_gen, steps=self.helper.get_steps_per_epoch(self.train_ds, batch_size, True))
            metrics_train = {"train_loss" : train_loss,
                             "train_accuracy" : train_accuracy,
                             "train_precision": train_precision,
                             "train_recall" : train_recall}
            metrics.append(metrics_train)
            current_picks.append(metrics_train)
            self.save_metrics(metrics)
            self.results.append(current_picks)
        highest_test_accuracy_index, highest_train_accuracy_index, highest_test_precision_index, highest_train_recall_index = self.find_best_performers(self.results)
        return self.results, highest_test_accuracy_index, highest_train_accuracy_index, highest_test_precision_index, highest_test_recall_index

    def find_best_performers(self, results):
        highest_test_accuracy = 0
        highest_test_accuracy_index = 0
        highest_train_accuracy = 0
        highest_train_accuracy_index = 0
        highest_test_precision = 0
        highest_test_precision_index = 0
        highest_train_precision = 0
        highest_train_precision_index = 0
        highest_test_recall = 0
        highest_test_recall_index = 0
        highest_train_recall = 0
        highest_train_recall_index = 0
        for idx, result in enumerate(results):
            if result[3]["test_accuracy"] > highest_test_accuracy:
                highest_test_accuracy = result[3]["test_accuracy"]
                highest_test_accuracy_index = idx
            if result[4]["train_accuracy"] > highest_train_accuracy:
                highest_train_accuracy = result[4]["train_accuracy"]
                highest_train_accuracy_index = idx
            if result[3]["test_precision"] > highest_test_precision:
                highest_test_precision = result[3]["test_precision"]
                highest_test_precision_index = idx
            if result[4]["train_precision"] > highest_train_precision:
                highest_train_precision = result[4]["train_precision"]
                highest_train_precision_index = idx
            if result[3]["test_recall"] > highest_test_recall:
                highest_test_recall = result[3]["test_recall"]
                highest_test_recall_index = idx
            if result[4]["train_recall"] > highest_train_recall:
                highest_train_recall = result[4]["train_recall"]
                highest_train_recall_index = idx
        print("----------------------------------------------------ACCURACY------------------------------------------------------")
        print(f'Highest test accuracy: {highest_test_accuracy}, at index: {highest_test_accuracy_index}')
        print(f'Highest training accuracy: {highest_train_accuracy}, at index: {highest_train_accuracy_index}')
        print("----------------------------------------------------PRECISION-----------------------------------------------------")
        print(f'Highest test precision: {highest_test_precision}, at index: {highest_test_precision_index}')
        print(f'Highest training precision: {highest_train_precision}, at index: {highest_train_precision_index}') 
        print("-----------------------------------------------------RECALL-------------------------------------------------------")
        print(f'Highest test recall: {highest_test_recall}, at index: {highest_test_recall_index}')
        print(f'Highest training recall: {highest_train_recall}, at index: {highest_train_recall_index}')
        print("------------------------------------------------------------------------------------------------------------------")
        return highest_test_accuracy_index, highest_train_accuracy_index, highest_test_precision_index, highest_test_recall_index
       
    
    def read_results(self):
        text_file = f'results_{self.model_nr}.txt'
        dictionaries = []
        with open(text_file, 'r') as file:
            for idx, line in enumerate(file):
                line = re.sub("\'", "\"", line.rstrip())
                if idx % 6 != 0: 
                    dictionaries.append(json.loads(line))
        dictionaries_by_model = []
        one_model = []
        for idx, dictionary in enumerate(dictionaries):
            if idx % 5 != 0 or idx == 0:
                one_model.append(dictionary)
            else:
                dictionaries_by_model.append(one_model)
                one_model = []
                one_model.append(dictionary)           
        return dictionaries_by_model
    
    def fit_from_result(self, dictionaries, index, train_channels = 3, timesteps = 6001, test = False, use_tensorboard = False):
        
        build_model_args = self.helper.generate_build_model_args(dictionaries[index][0]['model_nr'], dictionaries[index][1]['batch_size'], 
                                                     dictionaries[index][2]['dropout_rate'], dictionaries[index][2]['activation'], 
                                                     dictionaries[index][2]['l2_r'], dictionaries[index][2]['l1_r'],
                                                     dictionaries[index][2]['start_neurons'], dictionaries[index][2]['filters'],
                                                     dictionaries[index][2]['kernel_size'], dictionaries[index][2]['padding'])
        model = Models(**build_model_args).model

        model_compile_args = self.helper.generate_model_compile_args(dictionaries[index][1]['optimizer'])
        model.compile(**model_compile_args)
       
        gen_args = self.helper.generate_gen_args(dictionaries[index][1]['batch_size'], False, self.detrend, self.num_classes)
        
        train_gen = self.data_gen.data_generator(self.train_ds, **gen_args)
        val_gen = self.data_gen.data_generator(self.val_ds, **gen_args)
        test_gen = self.data_gen.data_generator(self.test_ds, **gen_args)

        fit_args = self.helper.generate_fit_args(self.train_ds, self.val_ds, dictionaries[index][1]['batch_size'], False, dictionaries[index][1]['epochs'], val_gen, use_tensorboard)
        model.fit(train_gen, **fit_args)
        return model
    
    def get_n_params_from_list(self, grid_list, n_picks):
        picks = []
        while (n_picks != 0):
            grid_length = len(grid_list)
            rand_int = random.randint(0,grid_length-1)
            picks.append(grid_list[rand_int])
            del grid_list[rand_int]
            n_picks -= 1
        return picks

    def getOptimizer(self, optimizer, learning_rate):
        if optimizer == "adam":
            return keras.optimizers.Adam(learning_rate=learning_rate)
        if optimizer == "rmsprop":
            return tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        if optimizer == "sgd":
            return tf.keras.optimizers.SGD(learning_rate=learning_rate)
        else:
            raise Exception(f"{optimizer} not implemented into getOptimizer")
    
    
    def save_params(self, current_params):
        with open(f'results_{self.model_nr}.txt', 'a', newline='') as file:
            file.writelines("-------------------------------------------------------------------------------------------------------------------------------------------\n")
            for part in current_params:
                file.writelines(str(part) + '\n')
            file.close()
            
    def save_metrics(self, metrics):
        with open(f'results_{self.model_nr}.txt', 'a', newline='') as file:
            for part in metrics:
                file.writelines(str(part) + '\n')
            file.close()
    
    def delete_progress(self):
        with open(f'results_{self.model_nr}.txt', 'w+', newline='') as file:
            file.truncate(0)

    """    
    def generate_build_model_args(self, model_nr, batch_size, dropout_rate, activation, l2_r, start_neurons, filters, kernel_size, padding):
        _, self.channels, self.timesteps = self.get_trace_shape_no_cast(self.test_ds)
        return {"model_nr" : model_nr,
                "input_shape" : (batch_size, self.channels, self.timesteps),
                "num_classes" : self.num_classes,
                "dropout_rate" : dropout_rate,
                "activation" : activation,
                "l2_r" : l2_r,
                "full_regularizer" : True,
                "start_neurons" : start_neurons,
                "filters" : filters,
                "kernel_size" : kernel_size,
                "padding" : padding}
    
    def generate_model_compile_args(self, opt):
         return {"loss" : "categorical_crossentropy",
                      "optimizer" : opt,
                      "metrics" : ["accuracy","MSE",
                                   tf.keras.metrics.Precision(thresholds=None, top_k=None, class_id=None, name=None, dtype=None),
                                   tf.keras.metrics.Recall(thresholds=None, top_k=None, class_id=None, name=None, dtype=None)]}
    def generate_gen_args(self, batch_size, test, detrend):
        return {"batch_size" : batch_size,
                    "shuffle" : True,
                    "test" : test,
                    "detrend" : detrend,
                    "num_classes" : self.num_classes}
    
    def generate_fit_args(self, batch_size, test, epoch, val_gen):
        return {"steps_per_epoch" : self.helper.get_steps_per_epoch(self.train_ds, batch_size, test),
                        "epochs" : epoch,
                        "validation_data" : val_gen,
                        "validation_steps" : self.helper.get_steps_per_epoch(self.val_ds, batch_size, test),
                        "verbose" : 1,
                        "use_multiprocessing" : False, 
                        "workers" : 1,
                        "callbacks" : [PlotLossesKeras()] 
                       }
"""
    
    
    
    
    
    
    
    
    
    
    
    
    