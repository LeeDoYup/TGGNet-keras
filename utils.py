
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import keras
import keras.backend as K
import random

def data_loader(dir='/data/project/rw/kakao_dataset', train=True, valid=False,\
                scale='log', valid_ratio=0.2, coord=25.0):
    """
    After load train(test) data in dir, preprocess and return the inputs and ouput of model.
    :param dir: directory path of data
    :param train: is training (including validation) or test data
    :param valid: is validation data used or not
    :param scale: scaling of data ('log', 'min_max')
    :param validation: the ratio of spliting validation dataset from training data
    :param coord: scaling factor for coordination Information
    :return X: [pick-up, drop-off, temporal information, coordination], Y: next-step pick-up
    """

    x_st = load_np_data(os.path.join(dir, 'x_st_train.npz'))
    x_end = load_np_data(os.path.join(dir, 'x_end_train.npz'))
    min_val_st, max_val_st = get_min_max(x_st, scale)
    min_val_end, max_val_end = get_min_max(x_end, scale)

    if train:
        coord = load_np_data(os.path.join(dir, 'coord_train.npz'))/coord
        temporal = load_np_data(os.path.join(dir, 'temporal_train.npz'))
        y = load_np_data(os.path.join(dir, 'y_st_train.npz'))

        #if scale is not 'min_max', return None, None
        x_st = scaler(x_st, scale, inv=False, min_value=min_val_st, max_value=max_val_st)
        x_end = scaler(x_end, scale, inv=False, min_value=min_val_end, max_value=max_val_end)
        y = scaler(y, scale, inv=False, min_value=min_val_st, max_value=max_val_st)
        if True:
            num_train = int(len(x_st)*(1.0-valid_ratio))

            x_st_train, x_st_valid = x_st[:num_train], x_st[num_train:]
            x_end_train, x_end_valid = x_end[:num_train], x_end[num_train:]
            coord_train, coord_valid = coord[:num_train], coord[num_train:]
            temporal_train, temporal_valid = temporal[:num_train], temporal[num_train:]
            y_train, y_valid = y[:num_train], y[num_train:]
        if valid:
            return [x_st_valid, x_end_valid, temporal_valid, coord_valid], y_valid
        else:
            return [x_st_train, x_end_train, temporal_train, coord_train], y_train

    else:
        x_st= load_np_data(os.path.join(dir, 'x_st_test.npz'))
        x_end= load_np_data(os.path.join(dir, 'x_end_test.npz'))
        x_st = scaler(x_st, scale, inv=False, min_value=min_val_st, max_value=max_val_st)
        x_end = scaler(x_end, scale, inv=False, min_value=min_val_st, max_value=max_val_st)

        coord= load_np_data(os.path.join(dir, 'coord_test.npz'))/coord
        temporal= load_np_data(os.path.join(dir, 'temporal_test.npz'))
        y= load_np_data(os.path.join(dir, 'y_st_test.npz'))
        y = scaler(y, scale, inv=False, min_value=min_val_st, max_value=max_val_st)

    return [x_st, x_end, temporal, coord], y


def load_np_data(filename):
    try:
        data = np.load(filename)['arr_0']
        print("[*] Success to load ", filename)
        return data
    except:
        raise IOError("Fail to load data", filename)

def scaler(data, scale_type='log', inv=False, min_value=None, max_value=None):
    if scale_type == 'log':
        if not inv:
            print("[*] ", np.shape(data), ":log scaled")
            return logscale(data)
        else:
            print("[*] ", np.shape(data), ": inverse log scaled")
            return inverse_logscale(data)
    elif scale_type == 'min_max':
        assert (min_value != None) and (max_value != None)
        if not inv:
            return min_max(data, min_value, max_value)
        else:
            return inverse_min_max(data, min_value, max_value)
    else:
        print("[!] unvalid scale type: ", scale_type)
        raise

def logscale(data):
    return np.log(data+1.0)

def inverse_logscale(data):
    result = np.exp(data)-1.0
    result = result.astype(int)
    return result

def get_min_max(data, scale='min_max'):
    if scale=='min_max':
        return np.min(data), np.max(data)
    else:
        return None, None

def min_max(data, min_value, max_value):
    result = data - min_value
    scale = max_value - min_value
    assert scale > 0
    result = result/scale
    return result

def inverse_min_max(data, min_value, max_value):
    scale = max_value - min_value
    result = scale * data + min_value
    return result



############################################################################
### Metric
############################################################################

def rmse(y_true,y_pred):
    rtn = np.sqrt(  np.average( np.square(y_pred-y_true) ) )
    return  rtn

def mape(y_true,y_pred):
    rtn = np.mean(np.abs((y_true - y_pred) / (1.0+y_true)))
    return rtn

#######################################################################
### Treshhold Metric
#######################################################################

def mape_trs(y_true,y_pred, trs=1):
    true_mask = y_true>=trs
    tmp_abs = np.divide(np.abs(y_true-y_pred)[true_mask] , y_true[true_mask])

    rtn = (np.average(tmp_abs))
    return rtn

def rmse_trs(y_true,y_pred, trs=1):
    true_mask = y_true>=trs
    tmp_abs = np.sqrt(np.average(np.square(y_pred-y_true)[true_mask]))
    return tmp_abs



def maa_trs(y_true,y_pred, trs=0.0):
    if trs == 1.0 :
        return 1
    else :
        true_mask = (y_pred>=y_true*(1.0-trs))&(y_pred<=y_true*(1.0+trs))
    return np.average(true_mask)


#######################################################################
## Keras Fit Fuction
#######################################################################

def invlog_mape_tr10(y_true,y_pred):
    y_true = K.exp(y_true)-1
    y_pred = K.exp(y_pred)-1

    true_mask = K.greater(y_true,10)
    true_mask = K.cast(true_mask, dtype=K.floatx())
    return K.mean(K.abs(  ( tf.boolean_mask ( (y_true - y_pred), true_mask )  / tf.boolean_mask(y_true, true_mask) ) ) )

def invlog_rmse_tr10(y_true,y_pred):

    y_true = K.exp(y_true)-1
    y_pred = K.exp(y_pred)-1

    true_mask = K.greater(y_true,10)
    true_mask = K.cast(true_mask, dtype=K.floatx())
    return K.sqrt(K.mean(K.square(tf.boolean_mask((y_true - y_pred), true_mask))))


def invlog_mape(y_true,y_pred):
    y_true = K.exp(y_true)-1
    y_pred = K.exp(y_pred)-1
    return K.mean(K.abs((y_true - y_pred) / (1.0+y_true)))

def invlog_rmse(y_true, y_pred):
    y_true = K.exp(y_true)-1
    y_pred = K.exp(y_pred)-1
    return K.sqrt( K.mean(K.square(y_true - y_pred) ) )


def mae_t1(y_true,y_pred):
    sub = tf.subtract(y_pred[:,:,:,:1], y_true[:,:,:,:1])
    return tf.reduce_mean(tf.abs(sub))

def mae_t2(y_true,y_pred):
    sub = tf.subtract(y_pred[:,:,:,-1:], y_true[:,:,:,-1:])
    return tf.reduce_mean(tf.abs(sub))


if __name__ == '__main__':
	print('test')
