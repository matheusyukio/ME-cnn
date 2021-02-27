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

from sklearn.model_selection import train_test_split

class Gater(object):
    def __init__(self):
        self.experts = None
        self.train_dim = None
        self.test_dim = None
        #iteracoes pra pegar o sample da expert
        self.iters = 3
        self.wm_xi = None
        #saida da rede 5 423 - 30 - 34
        self.target = 34
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
        model = model.LeNet5()
        model.fit(X, y, batch_size=64, epochs=3, validation_data=(x_val, y_val),
                  verbose=1, callbacks=[self.early_stopping])

        yhat_train = model.predict(X, batch_size=64)
        yhat_val = model.predict(x_val, batch_size=64)

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
        #output da saida da rede convolucional
        dim_inputs_data = Input(shape=(84, ))
        dim_mlp_yhat = Input(shape=(self.target * self.experts,))

        layer_1 = Dense(50, activation='relu')(dim_inputs_data)
        layer_1_a = Dropout(0.5)(layer_1)
        layer_2 = Dense(50, activation='relu')(layer_1_a)
        layer_2_a = Dropout(0.5)(layer_2)
        layer_3 = Dense(self.experts, name='layer_op', activation='relu', use_bias=False)(layer_2_a)
        layer_4 = Lambda(self.tensor_product)([layer_3, dim_mlp_yhat])
        #saida da rede 5 423 - 30 - 34
        layer_5 = Dense(34, activation='softmax')(layer_4)

        model = Model(inputs=[dim_inputs_data, dim_mlp_yhat], outputs=layer_5)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def main(self, x_train, y_train, x_test, y_test, x_val, y_val):
        print("############################# Prime Train ################################")
        model_prime = nn_models()
        model_prime.ip_shape = x_train.shape
        model_p = model_prime.LeNet5()

        model_prime = Model(inputs=model_p.input,
                            outputs=model_p.get_layer('dense2').output)

        prime_op_tr = model_prime.predict(x_train)
        prime_op_tt = model_prime.predict(x_test)
        prime_op_v = model_prime.predict(x_val)

        for i in range(self.iters):
            split_buckets = self.bucket_function(i)

            experts_out_train = []
            experts_out_test = []
            experts_out_val = []
            for j in sorted(split_buckets):
                X = x_train[split_buckets[j]]
                y = y_train[split_buckets[j]]

                model = self.train_model(X, y, x_val, y_val, i)
                yhats_train = model.predict(x_train, batch_size=64)
                yhats_test = model.predict(x_test, batch_size=64)
                yhats_val = model.predict(x_val, batch_size=64)

                exp_tre = eval_target(yhats_train, y_train)
                exp_tte = eval_target(yhats_test, y_test)
                exp_vale = eval_target(yhats_val, y_val)

                print("############################# Expert ################################")
                print('i, train_eval, test_eval, validation_eval')
                print('{}, {}, {}, {}'.format(i, exp_tre, exp_vale, exp_tte))

                experts_out_train.append(yhats_train)
                experts_out_test.append(yhats_test)
                experts_out_val.append(yhats_val)

            yhat_tr = np.hstack(experts_out_train)
            yhat_tt = np.hstack(experts_out_test)
            yhat_val = np.hstack(experts_out_val)

            model = self.gater()
            history = model.fit([prime_op_tr, yhat_tr], y_train, shuffle=True,
                                batch_size=64, verbose=1,
                                validation_data=([prime_op_v, yhat_val], y_val),
                                epochs=3, callbacks=[self.early_stopping])

            yhats_train = model.predict([prime_op_tr, yhat_tr], batch_size=64)
            yhats_test = model.predict([prime_op_tt, yhat_tt], batch_size=64)
            yhats_val = model.predict([prime_op_v, yhat_val], batch_size=64)

            tre = eval_target(yhats_train, y_train)
            tte = eval_target(yhats_test, y_test)
            vale = eval_target(yhats_val, y_val)

            print("############################# Gater ################################")
            print('i, train_eval, test_eval, validation_eval')
            print('{}, {}, {}, {}'.format(i, tre, vale, tte))
            logging.info('{}, {}, {}, {}'.format(i, tre, vale, tte))

            expert_units = Model(inputs=model.input,
                                 outputs=model.get_layer('layer_op').output)

            self.wm_xi = expert_units.predict([prime_op_tr, yhat_tr])

        return None

PATH = os.getcwd()

def main():
    start_time = time.time()
    print('##### NEW EXPERIMENT_' + str(start_time) + '_#####')
    TRAIN = os.path.join(PATH, 'training_data_Inception2_5_1')
    TEST = os.path.join(PATH, 'validation_data_Inception2_5_1')
    epoch = 100
    #423 classes para 5 imgs / 34 classes para 30 imgs
    min_images_per_person = [30]
    models = ["DeepFace"]
    num_folds = 2
    batch_sizes = [30]  # [2,4,8]

    #main execution
    #for min_per_person in min_images_per_person:
    min_per_person = min_images_per_person[0]
    multi_data = get_mounted_data(min_per_person, min_per_person)
    data_x, data_y = transform_image_dataframe_to_matrix(multi_data, 250, 250, 'lfw-dataset/lfw-deepfunneled/lfw-deepfunneled/')

    X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=1, stratify=data_y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1, stratify=y_train)

    gater = Gater()
    gater.experts = 5
    gater.train_dim = X_train.shape
    gater.test_dim = X_test.shape
    print("Experts {}".format(gater.experts))
    print('{}, {}, {}, {}'.format("Training Error", "Val Error", "Test Error", "Time"))
    gater.main(X_train, y_train, X_test, y_test, X_val, y_val)

    for i in gater.warn_log:
        print(i)

if __name__ == "__main__":
    main()