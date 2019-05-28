import numpy as np
import pandas as pd
from utils import *

def output_analysis(pred_inverse, y_inverse, output_path=None):
    assert pred_inverse.shape[:3] == y_inverse.shape[:3]
    num_row, h, w = y_inverse.shape[:3]
    num_col = int(h*w)
    np_pred = np.reshape(pred_inverse, [num_row, num_col])
    np_y = np.reshape(y_inverse, [num_row, num_col])

    return [mape(y_inverse, pred_inverse), rmse(y_inverse, pred_inverse), mape_trs(np_y, np_pred, trs=11), rmse_trs(np_y, np_pred, trs=11)]

def flatten_result(data):
    num_row, h, w = data.shape[:3]
    num_col = int(h*w)
    return np.reshape(data, [num_row, num_col])

def save_test_output(pred_inverse, y_inverse, output_path=None):
    num_row, h, w = pred_inverse.shape[:3]
    num_col = int(h*w)
    assert pred_inverse.shape[:3] == y_inverse.shape[:3]
    if output_path == None:
        output_path = './model_output/temporal_directory'
        print("[!] Please Assign Output Path in Arguments")

    np_pred = flatten_result(pred_inverse) #Enp.reshape(pred_inverse, [num_row, num_col])
    np_y = flatten_result(y_inverse) #np.reshape(y_inverse, [num_row, num_col])

    col_name = ['col_'+str(i) for i in range(0, num_col)]
    index = np.arange(0, num_row)
    df_pred = pd.DataFrame(np_pred, columns=col_name, index=index)
    df_y = pd.DataFrame(np_y, columns=col_name, index=index)

    df_y.to_csv(output_path+'_gt.csv')
    df_pred.to_csv(output_path+'_pred.csv')

    print('MAPE original Data(+1) : %.3f'%mape(y_inverse, pred_inverse))
    print('')

    model_output_chk(output_path)


def model_output_chk(output_path, trs=20, time_avg=4, base_lag = 8):
    df_true = pd.read_csv(output_path+'_gt.csv').drop('Unnamed: 0', axis=1)
    df_pred = pd.read_csv(output_path+'_pred.csv').drop('Unnamed: 0', axis=1)

    # Print Basic Information of Model and Performances
    print('## Model Name : ', output_path.split('/')[-1], '\n')
    print('## ------------ Base Metric -------------')
    print('- True Max %0.0f'%df_true.values.max(), ', Pred Max %.0f'%df_pred.values.max())
    print('- True Avg %0.2f'%np.average(df_true.values), ', Pred Avg %.2f'%np.average(df_pred.values))
    print('- MAPE(+1) : %.3f'%mape(df_true.values,df_pred.values))
    print('- RMSE : %.3f'%rmse(df_true.values,df_pred.values), '\n')


    # Print Performance with Thresholds
    print('## ----------- Trs MAPE Metric ------------')
    for trs in range(15):
        tmp_trs_mape = mape_trs(df_true.values,df_pred.values, trs=trs)
        tmp_trs_rmse = rmse_trs(df_true.values,df_pred.values, trs=trs)
        print('- %.0f or More'%trs, ' MAPE : %.3f'%tmp_trs_mape)
        print('- %.0f or More'%trs, ' RMSE : %.3f'%tmp_trs_rmse)


#######################################################################
## Output Check
#######################################################################

def holiday_marker(temp, dataset='kakao'):
    holi_index = {'NYC': [2,3,-1], 'kakao': [5,6,-1], 'NYCB': [0,6,-1]}
    holi_index = np.array(holi_index[dataset])
    holiday = temp[:,holi_index]
    holiday = np.sum(holiday, axis=1)
    marker = np.min([holiday, np.ones(holiday.shape)], axis=0)
    return marker

def get_thrs(y_st, temporal, alpha=0.05, dataset='kakao', is_holiday=None):
    train_time = np.expand_dims(np.argmax(temporal[:,7:-2], axis=1),axis=1)
    train_holi = np.concatenate([temporal[:,:7], temporal[:,-1:]], axis=1)

    # Make Holiday Marker: 0=weekday, 1: weekend & holiday
    train_holi_marker = np.expand_dims(holiday_marker(train_holi, dataset), axis=1)
    h, w = np.shape(y_st)[1:3]
    st_2d = np.reshape(y_st, [-1,h*w])

    df_st_time = pd.DataFrame(np.concatenate([st_2d, train_time, train_holi_marker], axis=1))
    col_names = df_st_time.columns
    df_st_time = df_st_time.rename(columns={col_names[-2]:'time', col_names[-1]:'holiday'})

    thr_mtx = np.zeros([h*w,48])
    time_list = list(range(48))
    def return_thr(t):
        if is_holiday == None:
            df = df_st_time[df_st_time['time']==t]
        elif is_holiday == 1:
            df = df_st_time[(df_st_time['time']==t) & (df_st_time['holiday']==1)]
        else:
            df = df_st_time[(df_st_time['time']==t) & (df_st_time['holiday']==0)]
        top_num = int(len(df)*alpha)+1
        df_array = df.values[:,:-2].T
        sort_df = np.array(list(map(np.sort, df_array)))
        thrs = np.array(list(map(lambda x:x[-top_num], sort_df)))
        return thrs
    thr_list = np.array(list(map(lambda t:return_thr(t), time_list))).T
    print("[*] Atypical Event Thresholds are Calculated: ", np.shape(thr_list))
    return thr_list


def atypical_index(df, thr_mtx):
    time_list = list(range(48))
    def time_process(_df, t, thr):
        df_ = _df[_df['time']==t]
        df_idx = np.array(df_.index)
        idx = list(np.where(df_[df_.columns[0]]>thr))
        return df_idx[idx]
    index = list(map(lambda x,y:time_process(df, x,y), time_list, thr_mtx))
    result = np.concatenate([arr for arr in index])
    return result


def get_atypical_idx(y_train, train_temporal, y_test, test_temporal, is_holiday=False, alpha=0.05, dataset='kakao'):
    h,w = np.shape(y_test)[1:3]
    test_df = np.reshape(y_test, [-1, h*w])
    test_time = np.expand_dims(np.argmax(test_temporal[:,7:-2], axis=1), axis=1)
    test_holi = np.concatenate([test_temporal[:,:7], test_temporal[:,-1:]], axis=1)
    test_holi_marker = np.expand_dims(holiday_marker(test_holi, dataset), axis=1)

    test_df = pd.DataFrame(np.concatenate([test_df, test_time, test_holi_marker], axis=1))
    col_names = test_df.columns
    test_df = test_df.rename(columns={col_names[-2]:'time', col_names[-1]:'holiday'})
    df_list = [test_df[[i, 'time', 'holiday']] for i in range(h*w)]

    if not is_holiday:
        thr_list = get_thrs(y_train, train_temporal, alpha, dataset, None)
        index = np.array(list(map(lambda x,y:atypical_index(x,y), df_list, thr_list)))
    else:
        thr_list_0 = get_thrs(y_train, train_temporal, alpha, dataset, 0)
        index_0 = np.array(list(map(lambda x,y:atypical_index(x,y), df_list, thr_list_0)))
        thr_list_1 = get_thrs(y_train, train_temporal, alpha, dataset, 1)
        index_1 = np.array(list(map(lambda x,y:atypical_index(x,y), df_list, thr_list_1)))
        for i in range(len(index_0)):
            index_0[i] = np.concatenate([index_0[i], index_1[i]])
        index = index_0

    index_dict = {}
    for i, idx in enumerate(index):
        index_dict[i]=idx
    return index_dict


def event_metric(y_true, y_pred, index_dict):
    #original version: time_ave =8, base_lag = 8
    num_data, h, w = np.shape(y_true)[:3]
    np_pred = np.reshape(y_pred, [-1, h*w])
    np_true = np.reshape(y_true, [-1, h*w])


    ## Key - Col , Item - Index List
    event_dict = index_dict
    event_true = []
    event_pred = []

    for key in event_dict:
        for item in event_dict[key]:
            event_pred.append(np_pred[item, key])
            event_true.append(np_true[item, key])
    event_true = np.asarray(event_true)
    event_pred = np.asarray(event_pred)

    print ('\n## ---- Event Metric -----')
    print ('- True Max %0.0f'%np.max(event_true), ', Pred Max %.0f'%np.max(event_pred))
    print ('- True Avg %0.3f'%np.average(event_true), ', Pred Avg %.3f'%np.average(event_pred))
    print ('- Event MAPE : %.3f'%mape(event_true,event_pred))
    print ('- Event RMSE : %.3f'%rmse(event_true,event_pred))

if __name__ == '__main__':
    #brief test code have to be added
    print('[*] test')
