import glob, os, sys, pdb, time
import pandas as pd
import numpy as np
import cv2
import pickle
from torch.utils.data import Dataset
from tqdm import tqdm
import torch

import config 

def q(text = ''): # easy way to exiting the script. useful while debugging
    print('> ', text)
    sys.exit()

class XRaysTrainDataset(Dataset):
    def __init__(self, data_dir, transform = None):
        self.data_dir = data_dir

        self.transform = transform
        # print('self.data_dir: ', self.data_dir)

        # full dataframe including train_val and test set
        self.df = self.get_df()
        print('self.df.shape: {}'.format(self.df.shape))

        self.make_pkl_dir(config.pkl_dir_path)

        # get train_val_df
        if not os.path.exists(os.path.join(config.pkl_dir_path, config.train_val_df_pkl_path)):

            self.train_val_df = self.get_train_val_df()
            print('\nself.train_val_df.shape: {}'.format(self.train_val_df.shape))

            # pickle dump the train_val_df
            with open(os.path.join(config.pkl_dir_path, config.train_val_df_pkl_path), 'wb') as handle:
                pickle.dump(self.train_val_df, handle, protocol = pickle.HIGHEST_PROTOCOL)
            print('{}: dumped'.format(config.train_val_df_pkl_path))
            
        else:
            # pickle load the train_val_df
            with open(os.path.join(config.pkl_dir_path, config.train_val_df_pkl_path), 'rb') as handle:
                self.train_val_df = pickle.load(handle)
            print('\n{}: loaded'.format(config.train_val_df_pkl_path))
            print('self.train_val_df.shape: {}'.format(self.train_val_df.shape))

        self.the_chosen, self.all_classes, self.all_classes_dict = self.choose_the_indices()
    
        if not os.path.exists(os.path.join(config.pkl_dir_path, config.disease_classes_pkl_path)):
            # pickle dump the classes list
            with open(os.path.join(config.pkl_dir_path, config.disease_classes_pkl_path), 'wb') as handle:
                pickle.dump(self.all_classes, handle, protocol = pickle.HIGHEST_PROTOCOL)
                print('\n{}: dumped'.format(config.disease_classes_pkl_path))
        else:
            print('\n{}: already exists'.format(config.disease_classes_pkl_path))

        self.new_df = self.train_val_df.iloc[self.the_chosen, :] # this is the sampled train_val data
        print('\nself.all_classes_dict: {}'.format(self.all_classes_dict))
            
    def resample(self):
        self.the_chosen, self.all_classes, self.all_classes_dict = self.choose_the_indices()
        self.new_df = self.train_val_df.iloc[self.the_chosen, :]
        print('\nself.all_classes_dict: {}'.format(self.all_classes_dict))

    def make_pkl_dir(self, pkl_dir_path):
        if not os.path.exists(pkl_dir_path):
            os.mkdir(pkl_dir_path)

    def get_train_val_df(self):

        # get the list of train_val data 
        train_val_list = self.get_train_val_list()

        train_val_df = pd.DataFrame()
        print('\nbuilding train_val_df...')
        for i in tqdm(range(self.df.shape[0])):
            filename  = os.path.basename(self.df.iloc[i,0])
            # print('filename: ', filename)
            if filename in train_val_list:
                train_val_df = train_val_df.append(self.df.iloc[i:i+1, :])

        # print('train_val_df.shape: {}'.format(train_val_df.shape))

        return train_val_df

    def __getitem__(self, index):
        row = self.new_df.iloc[index, :]

        img = cv2.imread(row['image_links'])
        labels = str.split(row['Finding Labels'], '|')
        
        target = torch.zeros(len(self.all_classes))
        for lab in labels:
            lab_idx = self.all_classes.index(lab)
            target[lab_idx] = 1            
    
        if self.transform is not None:
            img = self.transform(img)
    
        return img, target
        
    def choose_the_indices(self):
        
        max_examples_per_class = 10000 # its the maximum number of examples that would be sampled in the training set for any class
        the_chosen = []
        all_classes = {}
        length = len(self.train_val_df)
        # for i in tqdm(range(len(merged_df))):
        print('\nSampling the huuuge training dataset')
        for i in tqdm(list(np.random.choice(range(length),length, replace = False))):
            
            temp = str.split(self.train_val_df.iloc[i, :]['Finding Labels'], '|')

            # special case of ultra minority hernia. we will use all the images with 'Hernia' tagged in them.
            if 'Hernia' in temp:
                the_chosen.append(i)
                for t in temp:
                    if t not in all_classes:
                        all_classes[t] = 1
                    else:
                        all_classes[t] += 1
                continue

            # choose if multiple labels
            if len(temp) > 1:
                bool_lis = [False]*len(temp)
                # check if any label crosses the upper limit
                for idx, t in enumerate(temp):
                    if t in all_classes:
                        if all_classes[t]< max_examples_per_class: # 500
                            bool_lis[idx] = True
                    else:
                        bool_lis[idx] = True
                # if all lables under upper limit, append
                if sum(bool_lis) == len(temp):                    
                    the_chosen.append(i)
                    # maintain count
                    for t in temp:
                        if t not in all_classes:
                            all_classes[t] = 1
                        else:
                            all_classes[t] += 1
            else:        # these are single label images
                for t in temp:
                    if t not in all_classes:
                        all_classes[t] = 1
                    else:
                        if all_classes[t] < max_examples_per_class: # 500
                            all_classes[t] += 1
                            the_chosen.append(i)

        # print('len(all_classes): ', len(all_classes))
        # print('all_classes: ', all_classes)
        # print('len(the_chosen): ', len(the_chosen))
        
        '''
        if len(the_chosen) != len(set(the_chosen)):
            print('\nGadbad !!!')
            print('and the difference is: ', len(the_chosen) - len(set(the_chosen)))
        else:
            print('\nGood')
        '''

        return the_chosen, sorted(list(all_classes)), all_classes
    
    def get_df(self):
        csv_path = os.path.join(self.data_dir, 'Data_Entry_2017.csv')
        print('\n{} found: {}'.format(csv_path, os.path.exists(csv_path)))
        
        all_xray_df = pd.read_csv(csv_path)

        df = pd.DataFrame()        
        df['image_links'] = [x for x in glob.glob(os.path.join(self.data_dir, 'images*', '*', '*.png'))]

        df['Image Index'] = df['image_links'].apply(lambda x : x[len(x)-16:len(x)])
        merged_df = df.merge(all_xray_df, how = 'inner', on = ['Image Index'])
        merged_df = merged_df[['image_links','Finding Labels']]
        return merged_df
    
    def get_train_val_list(self):
        f = open(os.path.join('data', 'NIH Chest X-rays', 'train_val_list.txt'), 'r')
        train_val_list = str.split(f.read(), '\n')
        return train_val_list

    def __len__(self):
        return len(self.new_df)


# prepare the test dataset
class XRaysTestDataset(Dataset):
    def __init__(self, data_dir, transform = None):
        self.data_dir = data_dir
        self.transform = transform
        # print('self.data_dir: ', self.data_dir)

        # full dataframe including train_val and test set
        self.df = self.get_df()
        print('\nself.df.shape: {}'.format(self.df.shape))

        self.make_pkl_dir(config.pkl_dir_path)

        # loading the classes list
        with open(os.path.join(config.pkl_dir_path, config.disease_classes_pkl_path), 'rb') as handle:
            self.all_classes = pickle.load(handle)

        # get test_df
        if not os.path.exists(os.path.join(config.pkl_dir_path, config.test_df_pkl_path)):

            self.test_df = self.get_test_df()
            print('self.test_df.shape: ', self.test_df.shape)
            
            # pickle dump the test_df
            with open(os.path.join(config.pkl_dir_path, config.test_df_pkl_path), 'wb') as handle:
                pickle.dump(self.test_df, handle, protocol = pickle.HIGHEST_PROTOCOL)
            print('\n{}: dumped'.format(config.test_df_pkl_path))
        else:
            # pickle load the test_df
            with open(os.path.join(config.pkl_dir_path, config.test_df_pkl_path), 'rb') as handle:
                self.test_df = pickle.load(handle)
            print('\n{}: loaded'.format(config.test_df_pkl_path))
            print('self.test_df.shape: {}'.format(self.test_df.shape))

    def __getitem__(self, index):
        row = self.test_df.iloc[index, :]
        
        img = cv2.imread(row['image_links'])
        labels = str.split(row['Finding Labels'], '|')
        
        target = torch.zeros(len(self.all_classes))
        for lab in labels:
            lab_idx = self.all_classes.index(lab)
            target[lab_idx] = 1            
    
        if self.transform is not None:
            img = self.transform(img)
    
        return img, target

    def make_pkl_dir(self, pkl_dir_path):
        if not os.path.exists(pkl_dir_path):
            os.mkdir(pkl_dir_path)

    def get_df(self):
        csv_path = os.path.join(self.data_dir, 'Data_Entry_2017.csv')
        
        all_xray_df = pd.read_csv(csv_path)

        df = pd.DataFrame()        
        df['image_links'] = [x for x in glob.glob(os.path.join(self.data_dir, 'images*', '*', '*.png'))]

        df['Image Index'] = df['image_links'].apply(lambda x : x[len(x)-16:len(x)])
        merged_df = df.merge(all_xray_df, how = 'inner', on = ['Image Index'])
        merged_df = merged_df[['image_links','Finding Labels']]
        return merged_df

    def get_test_df(self):

        # get the list of test data 
        test_list = self.get_test_list()

        test_df = pd.DataFrame()
        print('\nbuilding test_df...')
        for i in tqdm(range(self.df.shape[0])):
            filename  = os.path.basename(self.df.iloc[i,0])
            # print('filename: ', filename)
            if filename in test_list:
                test_df = test_df.append(self.df.iloc[i:i+1, :])
         
        print('test_df.shape: ', test_df.shape)

        return test_df

    def get_test_list(self):
        f = open( os.path.join('data', 'NIH Chest X-rays', 'test_list.txt'), 'r')
        test_list = str.split(f.read(), '\n')
        return test_list

    def __len__(self):
        return len(self.test_df)






'''
# prepare the test dataset
import random
class XRaysTestDataset2(Dataset):
    def __init__(self, test_data_dir, transform = None):
        self.test_data_dir = test_data_dir
        self.transform = transform
        self.data_list = self.get_data_list(self.test_data_dir)
        
        self.subset = self.data_list[:1000]

    def __getitem__(self, index):
        img_path = self.data_list[index]
        img = cv2.imread(img_path)
        
        if self.transform is not None:
            img = self.transform(img)
            
        return img_path
    
    def sample(self):

        random.shuffle(self.data_list)

        self.subset = self.data_list[:np.random.randint(500,700)]

    def __len__(self):
        return len(self.subset)
        
    def get_data_list(self, data_dir):
        data_list = []
        for path in glob.glob(data_dir + os.sep + '*'):
            data_list.append(path)
        return data_list
'''
