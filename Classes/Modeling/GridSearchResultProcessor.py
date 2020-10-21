from Classes.DataProcessing.LoadData import LoadData
import os
import sys
import numpy as np
import pandas as pd

class GridSearchResultProcessor():

    def __init__(self):
        super().__init__()
    
    def create_results_df(self):
        hyper_keys = list(self.hyper_grid.keys())
        model_keys = list(self.model_grid.keys())
        metrics_train_keys = ["train_loss", "train_accuracy", "train_precision", "train_recall"]
        metrics_val_keys = ["val_loss", "val_accuracy", "val_precision", "val_recall"]
        header = np.concatenate((hyper_keys, model_keys, metrics_train_keys, metrics_val_keys))
        results_df = pd.DataFrame(columns = header)
        return results_df
    
    def save_results_df(self, results_df, file_name):
        results_df.to_csv(file_name, mode = 'w')
    
    def clear_results_df(self, file_name):
        path = self.get_results_file_path()
        file = f"{path}/{file_name}"
        if os.path.isfile(file):
            f = open(file, "w+")
            f.close()
        
    
    def get_results_file_name(self):
        file_name = f"{self.get_results_file_path()}/results_{self.model_nr}"
        if self.test:
            file_name = f"{file_name}_test"
        if self.detrend:
            file_name = f"{file_name}_detrend"
        if self.use_scaler:
            if self.use_minmax:
                file_name = f"{file_name}_mmscale"
            else: 
                file_name = f"{file_name}_sscale"
        if self.use_noise_augmentor:
            file_name = f"{file_name}_noiseAug"
        if self.use_early_stopping:
            file_name = f"{file_name}_earlyS"
        if self.use_highpass:
            file_name = f"{file_name}_highpass{self.highpass_freq}"
        file_name = file_name + ".csv"
        return file_name
    
    def get_results_file_path(self):
        file_path = f'C:\Documents/Thesis_ssd/MasterThesis/GridSearchResults/{self.num_classes}_classes'
        return file_path
    
    def store_params_before_fit(self, current_picks, results_df, file_name):
        
        hyper_params = current_picks[1]
        model_params = current_picks[2]
        print(model_params)
        picks = []
        for key in list(hyper_params.keys()):
            picks.append(hyper_params[key])
        for key in list(model_params.keys()):
            picks.append(model_params[key])
        nr_fillers = len(results_df.columns) - len(picks)
        for i in range(nr_fillers):
            picks.append(np.nan)
        temp_df = pd.DataFrame(np.array(picks).reshape(1,21), columns = results_df.columns)
        results_df = results_df.append(temp_df, ignore_index = True)
        for idx, column in enumerate(results_df.columns):
            if idx >= 13:
                results_df[column] = results_df[column].astype('float')
        self.save_results_df(results_df, file_name)
        return results_df


    def store_metrics_after_fit(self, metrics, results_df, file_name):
        metrics_train = metrics[1]
        metrics_val = metrics[0]
        print(metrics_train, metrics_val)
        finished_train = False
        # Get list of columns containing nan values
        unfinished_columns = results_df.columns[results_df.isnull().any()].tolist()
        # Iterate through every unfinished column and change values
        for idx, column in enumerate(unfinished_columns):
            if not finished_train:
                # Change value at column using the metrics
                results_df.iloc[-1, results_df.columns.get_loc(column)] = metrics_train[column]
                if idx == len(metrics_train)-1:
                    finished_train = True
            else:
                # Change value at column using the metrics
                results_df.iloc[-1, results_df.columns.get_loc(column)] = metrics_val[column]
        self.save_results_df(results_df, file_name)
        return results_df