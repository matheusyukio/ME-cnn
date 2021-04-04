import sys
sys.path.append('./utils')
import os
sys.path.append(os.getcwd())

import logging
import time
import os
from dataprocess import dataTrainAugmentation, dataHoldOutAugmentation, get_mounted_data, transform_image_dataframe_to_matrix

import numpy as np
from keras.utils import np_utils
from helper_callbacks import CustomCallback
from nn_arch import nn_models
from utils import load_array, eval_target
from keras.models import Model
from keras.layers import Dense, Input, Dropout, Lambda
from keras import backend as K
import keras
import pandas as pd
#import tensorflow as tf

from sklearn.model_selection import train_test_split

import scipy.io as sio

from write_plot_history import write_results

class Gater(object):
    def __init__(self):
        self.experts = None
        self.train_dim = None
        self.test_dim = None
        #iteracoes pra pegar o sample da expert
        self.iters = 3
        self.wm_xi = None
        #saida da rede 5 423 - 30 - 34
        self.target = 2
        self.early_stopping = CustomCallback()
        self.c = 1
        self.warn_log = [["Iter", "Expert Training Error", "Expert Val Error"]]

    def get_expert(self, weight_data):
        thresh_hard = int(weight_data.shape[0] / weight_data.shape[1]) + self.c

        # [0.2, 0.1, 0.7, 0.1] -> [2, 0 ,1, 3]
        sort_index = np.argsort(-1 * weight_data)
        thresh_dict = {}

        # {0:0 , 1:0 , 2:0 ..}
        thresh_dict = thresh_dict.fromkeys(list(range(weight_data.shape[1])), 0)
        local_expert = {}

        for k, v in enumerate(sort_index):
            for i in v:
                if thresh_dict[i] < thresh_hard:
                    thresh_dict[i] += 1
                    if i not in local_expert:
                        local_expert[i] = [k]
                    else:
                        local_expert[i].append(k)
                    break
        return local_expert

    def get_random(self):
        local_expert = {}
        # para cada entrada [1, 0, 1, 1,...] escolher qual especialista vai atuar [1, , 1, 1, 0, 0] significa que o especialista 1 vai atuar nas 3 primeras entradas e o especialista 0 vai atuar nas 2 ultimas
        random_bucket = np.random.choice(self.experts, self.train_dim[0])
        for i, e in enumerate(random_bucket):
            if e not in local_expert:
                local_expert[e] = [i]
            else:
                local_expert[e].append(i)
        return local_expert

    def bucket_function(self, i):
        if i == 0:
            return self.get_random()
        else:
            return self.get_expert(self.wm_xi)

    def train_model(self, X, y, x_val, y_val, i):
        model = nn_models()
        model.ip_shape = X.shape
        model = model.linearModel()
        print('-------------Train train_model Expert------------ {}'.format(i))
        history = model.fit(X, y, batch_size=16, epochs=2, validation_data=(x_val, y_val),
                  verbose=1, callbacks=[self.early_stopping])
        HISTORY = []
        HISTORY.append(history)
        VALIDATION_ACCURACY = []
        VALIDATION_LOSS = []
        VALIDATION_ACCURACY.append(1)
        VALIDATION_LOSS.append(2)

        #write_results(str(time.time())+'_Expert.txt', VALIDATION_ACCURACY, VALIDATION_LOSS, HISTORY)

        yhat_train = model.predict(X, batch_size=16)
        yhat_val = model.predict(x_val, batch_size=16)

        train_error = eval_target(yhat_train, y)
        val_error = eval_target(yhat_val, y_val)
        self.warn_log.append([i, train_error, val_error])

        return model

    def tensor_product(self, x):
        a = x[0]
        b = x[1]
        b = K.reshape(b, (-1, self.experts, self.target))
        y = K.batch_dot(b, a, axes=1)
        return y

    def gater(self):
        # ENTRADAS da GATE
        # entrada dos dados de treinamento
        dim_inputs_data = Input(shape=(2, ))
        # entrada da gate com as saidas dos especialista (numero de classes * numero especialistas)
        dim_mlp_yhat = Input(shape=(self.target * self.experts,))

        #Gate com saída linear e 1 neuronio na camada de ativacao
        layer_1 = Dense(1, activation='linear')(dim_inputs_data)
        # camada com a qdt de neuronios igual a qtd de especialistas
        layer_3 = Dense(self.experts, name='layer_op', activation='relu', use_bias=False)(layer_1)

        # Aqui realiza a multiplicação das saidas dos especialistas com os dados de entrada na Gating
        layer_4 = Lambda(self.tensor_product)([layer_3, dim_mlp_yhat])

        # SAIDA da Gating = qtd de classes
        layer_5 = Dense(2, activation='softmax')(layer_4)

        # link de todas as camadas criadas anteriormente
        model = Model(inputs=[dim_inputs_data, dim_mlp_yhat], outputs=layer_5)
        # optimizer = keras.optimizers.RMSprop(0.0099)
        SGD = keras.optimizers.SGD(lr=0.01, nesterov=True)
        model.compile(loss='binary_crossentropy', optimizer=SGD, metrics=['acc'])
        print(model.summary())
        # keras.utils.plot_model(model, to_file='gater1.png',  show_shapes=True, show_layer_names=True)
        return model

    def main(self, x_train, y_train, x_test, y_test, x_val, y_val):
        print("############################# Gater MAIN ################################")

        for i in range(self.iters):
            print("self.iters ============================ {}".format(i))
            # aqui fala onde cada especialista vai atuar em relação a entrada dos dados
            split_buckets = self.bucket_function(i)
            #cria os vetores de saida dos especialistas
            experts_out_train = []
            experts_out_test = []
            experts_out_val = []
            # Aqui separa os dados de treino que vai para cada especialista
            for j in sorted(split_buckets):
                print("========treinando especialista ======== {}".format(j))
                # aqui separa para cada especialista os dados que serao usados para treinar cada especialista
                X = x_train[split_buckets[j]]
                y = y_train[split_buckets[j]]

                model = self.train_model(X, y, x_val, y_val, i)
                yhats_train = model.predict(x_train, batch_size=16)
                yhats_test = model.predict(x_test, batch_size=16)
                yhats_val = model.predict(x_val, batch_size=16)

                exp_tre = eval_target(yhats_train, y_train)
                exp_tte = eval_target(yhats_test, y_test)
                exp_vale = eval_target(yhats_val, y_val)

                print("############################# Expert ################################")
                print('i, train_eval, test_eval, validation_eval')
                print('{}, {}, {}, {}'.format(i, exp_tre, exp_vale, exp_tte))

                experts_out_train.append(yhats_train)
                experts_out_test.append(yhats_test)
                experts_out_val.append(yhats_val)
            # Aqui pega as saidas dos especialista para cada classe
            # especilista 0
            # iteracao 0 = 0.9
            # iteracao 1 = 0.23
            # especilista 1
            # iteracao 0 = 0.8
            # iteracao 1 = 0.50
            # e coloca no formato de coluna
            # iteracao 0  iteracao 1
            # especialista 0 especilista 1 especialista 0 especilista 1
            # [0.9 0.23 0.8 0.50]
            yhat_tr = np.hstack(experts_out_train)
            yhat_tt = np.hstack(experts_out_test)
            yhat_val = np.hstack(experts_out_val)

            model = self.gater()
            print('-------------Train Gater------------')
            history = model.fit( [x_train, yhat_tr], y_train, shuffle=True,
                                batch_size=16, verbose=1,
                                validation_data=([x_val, yhat_val], y_val),
                                epochs=2, callbacks=[self.early_stopping])
            HISTORY = []
            HISTORY.append(history)
            VALIDATION_ACCURACY = []
            VALIDATION_LOSS = []
            VALIDATION_ACCURACY.append(1)
            VALIDATION_LOSS.append(2)

            #write_results(str(time.time()) + '_gater.txt', VALIDATION_ACCURACY, VALIDATION_LOSS, HISTORY)

            yhats_train = model.predict([x_train, yhat_tr], batch_size=16)
            yhats_test = model.predict([x_test, yhat_tt], batch_size=16)
            yhats_val = model.predict([x_val, yhat_val], batch_size=16)

            tre = eval_target(yhats_train, y_train)
            tte = eval_target(yhats_test, y_test)
            vale = eval_target(yhats_val, y_val)

            print("############################# Gater ################################")
            print('i, train_eval, test_eval, validation_eval')
            print('{}, {}, {}, {}'.format(i, tre, vale, tte))
            logging.info('{}, {}, {}, {}'.format(i, tre, vale, tte))

            expert_units = Model(inputs=model.input,
                                 outputs=model.get_layer('layer_op').output)

            self.wm_xi = expert_units.predict([x_train, yhat_tr])

        return None

PATH = os.getcwd()

def main():
    start_time = time.time()
    print('##### NEW EXPERIMENT_' + str(start_time) + '_#####')
    mat_contents = sio.loadmat('dados.mat')
    labels = []
    for i in mat_contents['Y']:
        if i == -1:
            labels.append(0)
        else:
            labels.append(1)
    labels_one_hot = pd.get_dummies(labels).to_numpy()
    #labels_one_hot = np.asarray(labels)
    X_train, X_test, y_train, y_test = train_test_split(mat_contents['X'], labels_one_hot, test_size=0.5, random_state=1, stratify=labels_one_hot)
    X_val = X_test
    y_val = y_test

    #X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1, stratify=y_train)

    gater = Gater()
    gater.experts = 2
    gater.train_dim = X_train.shape
    gater.test_dim = X_test.shape
    print("==================================================")
    print("Experts {}".format(gater.experts))
    print('{}, {}, {}, {}'.format("Training Error", "Val Error", "Test Error", "Time"))
    gater.main(X_train, y_train, X_test, y_test, X_val, y_val)

    for i in gater.warn_log:
        print(i)

if __name__ == "__main__":
    main()