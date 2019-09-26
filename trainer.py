import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys, os, time, random, pdb
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch
import pickle
import tqdm, pdb
from sklearn.metrics import roc_auc_score

import config

def get_roc_auc_score(y_true, y_probs):
    '''
    Uses roc_auc_score function from sklearn.metrics to calculate the micro ROC AUC score for a given y_true and y_probs.
    '''

    with open(os.path.join(config.pkl_dir_path, config.disease_classes_pkl_path), 'rb') as handle:
        all_classes = pickle.load(handle)
    
    NoFindingIndex = all_classes.index('No Finding')

    if True:
        print('\nNoFindingIndex: ', NoFindingIndex)
        print('y_true.shape, y_probs.shape ', y_true.shape, y_probs.shape)
        GT_and_probs = {'y_true': y_true, 'y_probs': y_probs}
        with open('GT_and_probs', 'wb') as handle:
            pickle.dump(GT_and_probs, handle, protocol = pickle.HIGHEST_PROTOCOL)

    class_roc_auc_list = []    
    useful_classes_roc_auc_list = []
    
    for i in range(y_true.shape[1]):
        class_roc_auc = roc_auc_score(y_true[:, i], y_probs[:, i])
        class_roc_auc_list.append(class_roc_auc)
        if i != NoFindingIndex:
            useful_classes_roc_auc_list.append(class_roc_auc)
    if True:
        print('\nclass_roc_auc_list: ', class_roc_auc_list)
        print('\nuseful_classes_roc_auc_list', useful_classes_roc_auc_list)

    return np.mean(np.array(useful_classes_roc_auc_list))

def make_plot(epoch_train_loss, epoch_val_loss, total_train_loss_list, total_val_loss_list, save_name):
    '''
    This function makes the following 4 different plots-
    1. mean train loss VS number of epochs
    2. mean val   loss VS number of epochs
    3. batch train loss for all the training   batches VS number of batches
    4. batch val   loss for all the validation batches VS number of batches
    '''
    fig = plt.figure(figsize=(16,16))
    fig.suptitle('loss trends', fontsize=20)
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    ax1.title.set_text('epoch train loss VS #epochs')
    ax1.set_xlabel('#epochs')
    ax1.set_ylabel('epoch train loss')
    ax1.plot(epoch_train_loss)

    ax2.title.set_text('epoch val loss VS #epochs')
    ax2.set_xlabel('#epochs')
    ax2.set_ylabel('epoch val loss')
    ax2.plot(epoch_val_loss)

    ax3.title.set_text('batch train loss VS #batches')
    ax3.set_xlabel('#batches')
    ax3.set_ylabel('batch train loss')
    ax3.plot(total_train_loss_list)

    ax4.title.set_text('batch val loss VS #batches')
    ax4.set_xlabel('#batches')
    ax4.set_ylabel('batch val loss')
    ax4.plot(total_val_loss_list)
    
    plt.savefig(os.path.join(config.models_dir,'losses_{}.png'.format(save_name)))

def get_resampled_train_val_dataloaders(XRayTrain_dataset, transform, bs):
    '''
    Resamples the XRaysTrainDataset class object and returns a training and a validation dataloaders, by splitting the sampled dataset in 80-20 ratio.
    '''
    XRayTrain_dataset.resample()

    train_percentage = 0.8
    train_dataset, val_dataset = torch.utils.data.random_split(XRayTrain_dataset, [int(len(XRayTrain_dataset)*train_percentage), len(XRayTrain_dataset)-int(len(XRayTrain_dataset)*train_percentage)])

    print('\n-----Resampled Dataset Information-----')
    print('num images in train_dataset   : {}'.format(len(train_dataset)))
    print('num images in val_dataset     : {}'.format(len(val_dataset)))
    print('---------------------------------------')

    # make dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = bs, shuffle = True)
    val_loader   = torch.utils.data.DataLoader(val_dataset,   batch_size = bs, shuffle = not True)

    print('\n-----Resampled Batchloaders Information -----')
    print('num batches in train_loader: {}'.format(len(train_loader)))
    print('num batches in val_loader  : {}'.format(len(val_loader)))
    print('---------------------------------------------\n')

    return train_loader, val_loader
    
def train_epoch(device, train_loader, model, loss_fn, optimizer, epochs_till_now, final_epoch, log_interval):
    '''
    Takes in the data from the 'train_loader', calculates the loss over it using the 'loss_fn' 
    and optimizes the 'model' using the 'optimizer'  
    
    Also prints the loss and the ROC AUC score for the batches, after every 'log_interval' batches. 
    '''
    model.train()
    
    running_train_loss = 0
    train_loss_list = []

    start_time = time.time()
    for batch_idx, (img, target) in enumerate(train_loader):
        # print(type(img), img.shape) # , np.unique(img))

        img = img.to(device)
        target = target.to(device)
        
        optimizer.zero_grad()    
        out = model(img)        
        loss = loss_fn(out, target)
        running_train_loss += loss.item()*img.shape[0]
        train_loss_list.append(loss.item())

        loss.backward()
        optimizer.step()
        
        if (batch_idx+1)%log_interval == 0:
            # batch metric evaluation
# #             out_detached = out.detach()
# #             batch_roc_auc_score = get_roc_auc_score(target, out_detached.numpy())
            # 'out' is a torch.Tensor and 'roc_auc_score' function first tries to convert it into a numpy array, but since 'out' has requires_grad = True, it throws an error
            # RuntimeError: Can't call numpy() on Variable that requires grad. Use var.detach().numpy() instead. 
            # so we have to 'detach' the 'out' tensor and then convert it into a numpy array to avoid the error !  

            batch_time = time.time() - start_time
            m, s = divmod(batch_time, 60)
            print('Train Loss for batch {}/{} @epoch{}/{}: {} in {} mins {} secs'.format(str(batch_idx+1).zfill(3), str(len(train_loader)).zfill(3), epochs_till_now, final_epoch, round(loss.item(), 5), int(m), round(s, 2)))
        
        start_time = time.time()
            
    return train_loss_list, running_train_loss/float(len(train_loader.dataset))

def val_epoch(device, val_loader, model, loss_fn, epochs_till_now = None, final_epoch = None, log_interval = 1, test_only = False):
    '''
    It essentially takes in the val_loader/test_loader, the model and the loss function and evaluates
    the loss and the ROC AUC score for all the data in the dataloader.
    
    It also prints the loss and the ROC AUC score for every 'log_interval'th batch, only when 'test_only' is False
    '''
    model.eval()

    running_val_loss = 0
    val_loss_list = []
    val_loader_examples_num = len(val_loader.dataset)

    probs = np.zeros((val_loader_examples_num, 15), dtype = np.float32)
    gt    = np.zeros((val_loader_examples_num, 15), dtype = np.float32)
    k=0

    with torch.no_grad():
        batch_start_time = time.time()    
        for batch_idx, (img, target) in enumerate(val_loader):
            if test_only:
                per = ((batch_idx+1)/len(val_loader))*100
                a_, b_ = divmod(per, 1)
                print(f'{str(batch_idx+1).zfill(len(str(len(val_loader))))}/{str(len(val_loader)).zfill(len(str(len(val_loader))))} ({str(int(a_)).zfill(2)}.{str(int(100*b_)).zfill(2)} %)', end = '\r')
    #         print(type(img), img.shape) # , np.unique(img))

            img = img.to(device)
            target = target.to(device)    
    
            out = model(img)        
            loss = loss_fn(out, target)    
            running_val_loss += loss.item()*img.shape[0]
            val_loss_list.append(loss.item())

            # storing model predictions for metric evaluat`ion 
            probs[k: k + out.shape[0], :] = out.cpu()
            gt[   k: k + out.shape[0], :] = target.cpu()
            k += out.shape[0]

            if ((batch_idx+1)%log_interval == 0) and (not test_only): # only when ((batch_idx + 1) is divisible by log_interval) and (when test_only = False)
                # batch metric evaluation
#                 batch_roc_auc_score = get_roc_auc_score(target, out)

                batch_time = time.time() - batch_start_time
                m, s = divmod(batch_time, 60)
                print('Val Loss   for batch {}/{} @epoch{}/{}: {} in {} mins {} secs'.format(str(batch_idx+1).zfill(3), str(len(val_loader)).zfill(3), epochs_till_now, final_epoch, round(loss.item(), 5), int(m), round(s, 2)))
            
            batch_start_time = time.time()    
            
    # metric scenes
    roc_auc = get_roc_auc_score(gt, probs)

    return val_loss_list, running_val_loss/float(len(val_loader.dataset)), roc_auc

def fit(device, XRayTrain_dataset, train_loader, val_loader, test_loader, model,
                                         loss_fn, optimizer, losses_dict,
                                         epochs_till_now, epochs, 
                                         log_interval, save_interval, 
                                         lr, bs, stage, test_only = False):
    '''
    Trains or Tests the 'model' on the given 'train_loader', 'val_loader', 'test_loader' for 'epochs' number of epochs.
    If training ('test_only' = False), it saves the optimized 'model' and  the loss plots ,after every 'save_interval'th epoch.
    '''
    epoch_train_loss, epoch_val_loss, total_train_loss_list, total_val_loss_list = losses_dict['epoch_train_loss'], losses_dict['epoch_val_loss'], losses_dict['total_train_loss_list'], losses_dict['total_val_loss_list']

    final_epoch = epochs_till_now + epochs

    if test_only:
        print('\n======= Testing... =======\n')
        test_start_time = time.time()
        test_loss, mean_running_test_loss, test_roc_auc = val_epoch(device, test_loader, model, loss_fn, log_interval, test_only = test_only)
        total_test_time = time.time() - test_start_time
        m, s = divmod(total_test_time, 60)
        print('test_roc_auc: {} in {} mins {} secs'.format(test_roc_auc, int(m), int(s)))
        sys.exit()

    starting_epoch  = epochs_till_now
    print('\n======= Training after epoch #{}... =======\n'.format(epochs_till_now))

    # epoch_train_loss = []
    # epoch_val_loss = []
    
    # total_train_loss_list = []
    # total_val_loss_list = []

    for epoch in range(epochs):

        if starting_epoch != epochs_till_now:
            # resample the train_loader and val_loader
            train_loader, val_loader = get_resampled_train_val_dataloaders(XRayTrain_dataset, config.transform, bs = bs)

        epochs_till_now += 1
        print('============ EPOCH {}/{} ============'.format(epochs_till_now, final_epoch))
        epoch_start_time = time.time()
        
        print('TRAINING')
        train_loss, mean_running_train_loss          =  train_epoch(device, train_loader, model, loss_fn, optimizer, epochs_till_now, final_epoch, log_interval)
        print('VALIDATION')
        val_loss, mean_running_val_loss, roc_auc     =  val_epoch(device, val_loader, model, loss_fn                             , epochs_till_now, final_epoch, log_interval)
        
        epoch_train_loss.append(mean_running_train_loss)
        epoch_val_loss.append(mean_running_val_loss)

        total_train_loss_list.extend(train_loss)
        total_val_loss_list.extend(val_loss)

        save_name = 'stage{}_{}_{}'.format(stage, str.split(str(lr), '.')[-1], str(epochs_till_now).zfill(2))

        # the follwoing piece of codw needs to be worked on !!! LATEST DEVELOPMENT TILL HERE
        if ((epoch+1)%save_interval == 0) or test_only:
            save_path = os.path.join(config.models_dir, '{}.pth'.format(save_name))
            
            torch.save({
            'epochs': epochs_till_now,
            'model': model, # it saves the whole model
            'losses_dict': {'epoch_train_loss': epoch_train_loss, 'epoch_val_loss': epoch_val_loss, 'total_train_loss_list': total_train_loss_list, 'total_val_loss_list': total_val_loss_list}
            }, save_path)
            
            print('\ncheckpoint {} saved'.format(save_path))

            make_plot(epoch_train_loss, epoch_val_loss, total_train_loss_list, total_val_loss_list, save_name)
            print('loss plots saved !!!')

        print('\nTRAIN LOSS : {}'.format(mean_running_train_loss))
        print('VAL   LOSS : {}'.format(mean_running_val_loss))
        print('VAL ROC_AUC: {}'.format(roc_auc))

        total_epoch_time = time.time() - epoch_start_time
        m, s = divmod(total_epoch_time, 60)
        h, m = divmod(m, 60)
        print('\nEpoch {}/{} took {} h {} m'.format(epochs_till_now, final_epoch, int(h), int(m)))



'''   
def pred_n_write(test_loader, model, save_name):
    res = np.zeros((3000, 15), dtype = np.float32)
    k=0
    for batch_idx, img in tqdm.tqdm(enumerate(test_loader)):
        model.eval()
        with torch.no_grad():
            pred = torch.sigmoid(model(img))
            # print(k)
            res[k: k + pred.shape[0], :] = pred
            k += pred.shape[0]
            
    # write csv
    print('populating the csv')
    submit = pd.DataFrame()
    submit['ImageID'] = [str.split(i, os.sep)[-1] for i in test_loader.dataset.data_list]
    with open('disease_classes.pickle', 'rb') as handle:
        disease_classes = pickle.load(handle)
    
    for idx, col in enumerate(disease_classes):
        if col == 'Hernia':
            submit['Hern'] = res[:, idx]
        elif col == 'Pleural_Thickening':
            submit['Pleural_thickening'] = res[:, idx]
        elif col == 'No Finding':
            submit['No_findings'] = res[:, idx]
        else:
            submit[col] = res[:, idx]
    rand_num = str(random.randint(1000, 9999))
    csv_name = '{}___{}.csv'.format(save_name, rand_num)
    submit.to_csv('res/' + csv_name, index = False)
    print('{} saved !'.format(csv_name))
'''
