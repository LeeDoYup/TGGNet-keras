
from utils import *
import output
import os
import math

import keras
import keras.backend as K
from keras import layers
from keras.layers import Input, Dense, Conv2D, AveragePooling2D, Conv2DTranspose, Activation
from keras.layers import concatenate, BatchNormalization, Dropout, Add, RepeatVector, Reshape
from keras.models import Model
from keras import regularizers

from keras.optimizers import SGD, Adam
from keras.utils.training_utils import multi_gpu_model


def gn_block(input, num_c=64, kernel_size=(3,3), strides=(1,1), padding='SAME', activation='relu', dropout=None, regularizer=0.01):
    net = AveragePooling2D(kernel_size, strides, padding)(input)
    net = Conv2D(num_c, kernel_size=(1,1), strides=strides, activation='linear', padding=padding, kernel_regularizer=regularizers.l1(regularizer))(net)

    net_sf = Conv2D(num_c, kernel_size=(1,1), strides=strides, activation='linear', padding=padding, kernel_regularizer=regularizers.l1(regularizer))(input)

    net = Add()([net, net_sf])
    net = concatenate([input, net])
    net = Conv2D(num_c, kernel_size=(1,1), strides=strides, activation=activation, padding=padding, kernel_regularizer=regularizers.l1(regularizer))(net)
    net = BatchNormalization()(net)

    if dropout == None:
        return net
    else:
        net = Dropout(dropout)(net)
        return net

def deconv_block(input, num_c=64, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', dropout=None, regularizer=0.01):
    net = Conv2DTranspose(num_c, kernel_size=kernel_size, strides=strides, activation=activation, padding=padding, kernel_regularizer=regularizers.l1(regularizer))(input)
    net = BatchNormalization()(net)
    if dropout == None:
        return net
    else:
        net = Dropout(dropout)(net)
        return net


class BaseModel():
    def __init__(self, input_shape, args=None):
        self.args = args

        self.lr = args.lr
        self.decay = args.decay
        self.epoch = args.epoch
        self.batch_size = args.batch
        self.model_name = args.model_name
        self.scale = args.scale
        self.input_shape = input_shape
        self._model = self.build_model(input_shape, args)
        if self.args.num_gpu < 2:
            self.model = self._model
        else:
            print("[*] ", args.num_gpu, " number of gpus would be utilized")
            self.model = multi_gpu_model(self._model, gpus=self.args.num_gpu)
        self.make_callbacks()


    def build_model(self, input_shape, args):
        pass

    def make_callbacks(self):
        patience = self.args.patience
        es = self.args.es
        self.tf_graph_dir = './tfgraph/'+self.model_name+'/'
        self.tb_hist = keras.callbacks.TensorBoard(log_dir=self.tf_graph_dir, histogram_freq=0, write_graph=True, write_images=True)
        self.early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=0, mode=es)

        self.check_point = keras.callbacks.ModelCheckpoint(self.tf_graph_dir+'wegihts.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=2, save_best_only=True)

    def load_model(self, save_model_name, save_dir='./model_saved', best=False):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        if not best:
            save_name = save_model_name +'.h5'
        else:
            save_name = save_model_name + '_best.h5'
        model_address = os.path.join(save_dir, save_name)
        self._model.load_weights(model_address)
        print(model_address, ' is loaded')

    def save_model(self, save_model_name, save_dir='./model_saved', best=False):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        if not best:
            save_name = save_model_name + '.h5'
        else:
            save_name = save_model_name + '_best.h5'
        model_address = os.path.join(save_dir, save_name)
        self._model.save(model_address)

    def train(self, x_train=None, y_train=None):
        x_train, y_train = data_loader(dir=self.args.dataset, train=True, scale=self.args.scale)
        self.x_train, self.y_train = x_train, y_train
        x_valid, y_valid = data_loader(dir=self.args.dataset, train=True, valid=True, scale=self.args.scale)

        ### Two stage learning
        ### STAGE ONE: L2 loss pretraining
        self.model.compile(loss=['mean_squared_error'], optimizer=Adam(lr=self.lr, decay=self.decay), metrics=['mean_absolute_error', invlog_mape])
        best_eval_loss_pretrain = 100000.0
        patience = 0
        for idx in range(self.epoch):
            history_1 = self.model.fit(x_train, y_train, batch_size=self.batch_size, epochs=1, shuffle=True, verbose=2)#, validation_data=(x_valid, y_valid))
            eval_loss = self.model.evaluate(x_valid, y_valid, batch_size=self.batch_size, verbose=2)
            patience +=1
            if patience > self.args.patience:
                break
            if best_eval_loss_pretrain > eval_loss[-1]:
                self.save_model(self.model_name+'pretrain_best', save_dir=self.args.save_dir)
                best_eval_loss_pretrain = eval_loss[-1]
                patience = 0
        self.load_model(self.model_name+'pretrain', save_dir=self.args.save_dir, best=True)

        ### STAGE TWO: L1 loss training
        self.model.compile(loss=['mean_absolute_error'], optimizer=Adam(lr=self.lr, decay=self.decay), metrics=['mean_absolute_error', invlog_mape])
        best_eval_loss = 10000.0
        patience = 0
        for idx in range(self.epoch):
            history_2 = self.model.fit(x_train, y_train,
                                batch_size=self.batch_size,
                                epochs=1,
                                shuffle=True,
                                verbose=2,
                                validation_data=(x_valid, y_valid))
            eval_loss = self.model.evaluate(x_valid, y_valid, batch_size=self.batch_size, verbose=2)
            patience += 1
            if patience > self.args.patience:
                break
            if best_eval_loss > eval_loss[-1]:
                self.save_model(self.model_name, self.args.save_dir, best=True)
                best_eval_loss = eval_loss[-1]
                patience = 0

        self.save_model(self.model_name, self.args.save_dir)
        self.load_model(self.model_name, save_dir=self.args.save_dir, best=True)
        mae_11, rmse_11 = self.test()
        return mae_11, rmse_11

    def save_history(self,loss, metric, model_name=None):
        if model_name == None:
            model_name = self.model_name
        output.history_save(loss, metric, model_name)

    def test(self, x_test=None, y_test=None, is_save=True):
        if x_test == None and y_test == None:
            x_train, y_train = data_loader(dir=self.args.dataset, train=True, scale=self.args.scale)
            x_test, y_test = data_loader(dir=self.args.dataset, train=False, scale=self.args.scale)
        else:
            x_train, y_train = self.x_train, self.y_train
        self.load_model(self.model_name, self.args.save_dir, best=True)
        min_val_st, max_val_st = get_min_max(load_np_data(self.args.dataset+'x_st_train.npz'), self.scale)
        pred_output = self.model.predict(x_test)
        atypical_index = output.get_atypical_idx(y_train, x_train[2], y_test, x_test[2], is_holiday=True, alpha=self.args.alpha, dataset=self.args.dataset_name)
        if is_save:
            #assert self.scale == 'log'
            if not os.path.exists(self.args.output_dir):
                os.mkdir(self.args.output_dir)
            output_path = os.path.join(self.args.output_dir, self.model_name)
            #pred_inverse = inverse_logscale(pred_output)
            #y_inverse = inverse_logscale(y_test)
            pred_inverse = scaler(pred_output, self.scale, inv=True, min_value=min_val_st, max_value=max_val_st)
            y_inverse = scaler(y_test, self.scale, inv=True, min_value=min_val_st, max_value=max_val_st)

            output.event_metric(y_inverse, pred_inverse, atypical_index)
            output.save_test_output(pred_inverse, y_inverse, output_path)
        pred_inverse = output.flatten_result(pred_inverse)
        y_inverse = output.flatten_result(y_inverse)
        mae_11 = mape_trs(y_inverse, pred_inverse, 11)
        rmse_11 = rmse_trs(y_inverse, pred_inverse, 11)
        return mae_11, rmse_11
