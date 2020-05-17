import argparse
import math
import os

import keras
import keras.backend as K
import numpy as np
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

tone_list = ['全色调', '白灰黑低调', '灰黑低调', '白黑低调', '白灰黑高调', '白灰高调', '白黑高调', '低反差']
tone_en = ['full_color', 'wgb_low', 'gb_low', 'wb_low', 'wgb_high', 'wg_high', 'wb_high', 'low_contrast']


class Dataset(keras.utils.Sequence):
    def __init__(self, data_dir, input_dim, batch_size, onehot=True, single_tone=False, idx=None):
        if single_tone:
            x, y = self.load_data_s(data_dir, input_dim, idx)
        else:
            x, y = self.load_data(data_dir, input_dim)

        self.x = x.astype('float32')
        if onehot:
            self.y = keras.utils.to_categorical(y, len(tone_en))
        else:
            self.y = y

        self.batch_size = batch_size

    def load_data_s(self, data_dir, input_dim, idx):
        x = np.load(data_dir)
        y = idx * np.ones(x.shape[0], dtype=int)

        rearrange_idx = [idx for idx in range(len(y))]
        np.random.shuffle(rearrange_idx)
        return x[rearrange_idx], y[rearrange_idx]

    def load_data(self, data_dir, input_dim):
        x = np.ndarray((0, input_dim))
        y = np.ndarray((0))
        for idx in range(len(tone_en)):
            path = os.path.join(data_dir, tone_en[idx] + '.npy')
            hists = np.load(path)
            if idx == 3 or idx == 6:  # wblow and wbhigh are small in number
                x = np.concatenate((x, hists), axis=0)
                y = np.concatenate((y, idx * np.ones(len(hists))))
            x = np.concatenate((x, hists), axis=0)
            y = np.concatenate((y, idx * np.ones(len(hists))))

        rearrange_idx = [idx for idx in range(len(y))]
        np.random.shuffle(rearrange_idx)
        return x[rearrange_idx], y[rearrange_idx]

    def __len__(self):
        return self.x.shape[0] // self.batch_size

    def __getitem__(self, idx):
        batch_size = self.batch_size
        return (self.x[idx * batch_size: (idx + 1) * batch_size], self.y[idx * batch_size: (idx + 1) * batch_size])


def step_decay(epoch):
    inital_rate = 0.05
    drop = 0.1
    interval = 15
    lr = inital_rate * math.pow(drop, ((epoch + 1) // interval))
    return lr


def perceptron(input_dim):
    net = keras.models.Sequential([
        keras.layers.Dense(200, input_shape=(input_dim,), activation='relu'),
        keras.layers.Dense(len(tone_en), activation='softmax')
    ])
    return net


def top2_accuracy_onehot(y_true, y_pred):
    '''y_true must be onehot'''
    return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=2)


def top2_accuracy(y_true, y_pred):
    return K.cast(K.in_top_k(y_pred, y_true, 2), K.floatx())


def train(args):

    train_gen = Dataset(args.train_dir, args.input_dim, args.batch_size)
    test_gen = Dataset(args.test_dir, args.input_dim, args.infer_batch_size)

    model = perceptron(args.input_dim)
    optim = keras.optimizers.adam(learning_rate=0.005)
    model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy', top2_accuracy_onehot])
    model.summary()

    ckpt_name = 'model_{epoch:02d}.hdf5'
    ckpt_helper = keras.callbacks.ModelCheckpoint(os.path.join(
        args.ckpt_path, ckpt_name), monitor='val_loss', verbose=1, save_weights_only=True, mode='auto', period=2)
    lrscheduler = keras.callbacks.LearningRateScheduler(step_decay, verbose=0)

    model.fit_generator(
        train_gen,
        validation_data=test_gen,
        epochs=args.max_epoch,
        verbose=1,
        callbacks=[ckpt_helper, lrscheduler]
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # path
    parser.add_argument('--data_dir', type=str, default='./gray')
    parser.add_argument('--ckpt_path', type=str, default='./records')
    # training
    parser.add_argument('--input_dim', type=int, default=128)
    parser.add_argument('--max_epoch', type=int, default=45)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--infer_batch_size', type=int, default=2)

    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        raise FileNotFoundError('Data dir does not exist.')

    args.train_dir = os.path.join(args.data_dir, 'train')
    args.test_dir = os.path.join(args.data_dir, 'test')

    if not os.path.exists(args.train_dir) or not os.path.exists(args.test_dir):
        raise FileNotFoundError('Train dir or test dir does not exist.')

    os.makedirs(args.ckpt_path, exist_ok=True)
    train(args)
