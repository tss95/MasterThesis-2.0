
import numpy as np
import pandas as pd
import json
import h5py
import seaborn as sns
import os
import csv

class LoadData():
    
    def __init__(self, csv_root, isBalanced = True):
        self.isBalanced = isBalanced
        self.data_path = 'data_tord_may2020'
        self.full_data_csv = 'balanced_csv_3_class.csv'
        self.csv_root = csv_root
        if isBalanced:
            self.root_sub = 'balanced'
        else:
            self.root_sub = 'raw'
        self.train_csv = f'{self.csv_root}/{self.root_sub}/train_set.csv'
        self.val_csv = f'{self.csv_root}/{self.root_sub}/validation_set.csv'
        self.test_csv = f'{self.csv_root}/{self.root_sub}/test_set.csv'
    
    def getCsvs(self):
        return self.full_data_csv, self.train_csv, self.val_csv, self.test_csv
    
    def getDatasets(self, shuffle = False):
        self.full_ds = self.load_dataset(self.full_data_csv, shuffle) 
        self.train_ds = self.load_dataset(self.train_csv, shuffle)
        self.val_ds = self.load_dataset(self.val_csv, shuffle)
        self.test_ds = self.load_dataset(self.test_csv, shuffle)
        return self.full_ds, self.train_ds, self.val_ds, self.test_ds
    
    def load_dataset(self, data_csv, shuffle = False):
        columns = ["path", "label"]
        df = pd.read_csv(data_csv, names = columns)
        if shuffle: 
            df = df.sample(frac = 1)
        return df.values
        
        