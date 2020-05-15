import argparse
import os

import keras
import keras.backend as K
import numpy as np
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
scale_name_cn = ['远景', '全景', '中景', '中近景', '近景', '特写']


def perceptron():
    net = keras.models.Sequential([
        keras.layers.Dense(200, input_shape=(52,), activation='relu'),
        keras.layers.Dense(len(scale_name_cn), activation='softmax')
    ])
    return net


def load_data(data_dir):
    x = np.ndarray((0, 52))
    y = np.ndarray((0))
    for idx in range(len(scale_name_cn)):
        path = os.path.join(data_dir, 'pose_' + scale_name_cn[idx] + '.npy')
        raw = np.load(path)
        x = np.concatenate((x, raw), axis=0)
        y = np.concatenate((y, idx * np.ones(len(raw))))

    rearrange_idx = [idx for idx in range(len(y))]
    np.random.shuffle(rearrange_idx)
    test_len = len(y) // 8
    return x[rearrange_idx[:-test_len]], y[rearrange_idx[:-test_len]], x[rearrange_idx[-test_len:]], y[rearrange_idx[-test_len:]],


class Dataset(keras.utils.Sequence):
    def __init__(self, x, y, batch_size, onehot=True):
        self.x = x
        if onehot:
            self.y = keras.utils.to_categorical(y, len(scale_name_cn))
        else:
            self.y = y

        self.batch_size = batch_size

    def __len__(self):
        return self.x.shape[0] // self.batch_size

    def __getitem__(self, idx):
        batch_size = self.batch_size
        return (self.x[idx * batch_size: (idx + 1) * batch_size], self.y[idx * batch_size: (idx + 1) * batch_size])


def top2_accuracy_onehot(y_true, y_pred):
    '''y_true must be onehot'''
    return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=2)


def top2_accuracy(y_true, y_pred):
    return K.cast(K.in_top_k(y_pred, y_true, 2), K.floatx())


def train(args):

    x_train, y_train, x_test, y_test = load_data(args.data_dir)

    train_gen = Dataset(x_train, y_train, args.batch_size)
    test_gen = Dataset(x_test, y_test, args.batch_size)

    model = perceptron()
    optim = keras.optimizers.adam(learning_rate=0.005)
    model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy', top2_accuracy_onehot])
    model.summary()

    ckpt_name = 'model_{epoch:02d}.hdf5'
    ckpt_helper = keras.callbacks.ModelCheckpoint(os.path.join(
        args.ckpt_path, ckpt_name), monitor='val_loss', verbose=1, save_weights_only=True, mode='auto', period=3)

    model.fit_generator(
        train_gen,
        validation_data=test_gen,
        epochs=args.max_epoch,
        verbose=1,
        callbacks=[ckpt_helper]
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=bool, default=True)
    # path
    parser.add_argument('--data_dir', type=str, default='D:/scale_dataset')
    parser.add_argument('--ckpt_path', type=str, default='./records')
    # training
    parser.add_argument('--max_epoch', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=6)

    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        raise FileNotFoundError('Data dir does not exist.')

    os.makedirs(args.ckpt_path, exist_ok=True)
    train(args)
