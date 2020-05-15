import argparse
import os

import keras
import keras.backend as K
import numpy as np
import tensorflow as tf
from keras import layers, models
from keras.callbacks import CSVLogger, ModelCheckpoint

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

pos_list_en = ['sl', 'bsl', 'fsl', 'fol', 'tl', 'frl']
pos_list = ['侧光', '侧逆光', '前侧光', '脚光', '顶光', '顺光']
# above abbrs stand for: sidelight, back-sidelight, front-sidelight,
# footlight, backlight, toplight, frontlight


def LenetBuilder():
    net = models.Sequential([
        layers.Conv2D(10, 5, activation='relu', input_shape=(3, 224, 224)),
        layers.MaxPool2D(strides=2),
        layers.Conv2D(10, 5, activation='relu'),
        layers.MaxPool2D(strides=2),
        layers.Conv2D(10, 6, activation='relu'),
        layers.MaxPool2D(strides=2),
        layers.Conv2D(20, 5, activation='relu'),
        layers.MaxPool2D(strides=2),
        layers.Conv2D(50, 3, activation='relu'),
        layers.MaxPool2D(strides=2),
        layers.Flatten(),
        layers.Dense(300, activation='relu'),
        layers.Dense(6, activation='softmax')
    ])
    return net


def cross_entropy_loss(labels, features):
    return K.categorical_crossentropy(labels, K.softmax(features, axis=-1))


def accuracy(y_true, y_pred):
    return keras.metrics.categorical_accuracy(y_true, K.softmax(y_pred, axis=-1))


def top2_accuracy(y_true, y_pred):
    return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=2)


class Dataset(keras.utils.Sequence):
    def __init__(self, data_dir, batch_size):
        x, y = self.load_data(data_dir)
        x = x.astype('float32')
        mean_img = np.mean(x, axis=0)
        x -= mean_img
        x /= 128.
        self.x = x
        self.y = keras.utils.to_categorical(y, len(pos_list_en), dtype='int32')

        self.batch_size = batch_size

    def load_data(self, data_dir):
        # read pictures and resize
        x = np.ndarray((0, 224, 224, 3), dtype='int32')
        y = np.ndarray((0, 1), dtype='int32')

        for pos_index in range(len(pos_list_en)):
            pos_en = pos_list_en[pos_index]
            t = np.load(os.path.join(data_dir, pos_en + '.npy'))

            x = np.concatenate((x, t), axis=0)
            y = np.concatenate(
                (y, pos_index * np.ones((len(t), 1), dtype='int32')), axis=0)

        x = np.transpose(x, (0, 3, 1, 2))

        rearrange_idx = [x for x in range(len(y))]
        np.random.shuffle(rearrange_idx)
        return x[rearrange_idx], y[rearrange_idx]

    def __len__(self):
        return self.x.shape[0] // self.batch_size

    def __getitem__(self, idx):
        batch_size = self.batch_size
        return (self.x[idx * batch_size: (idx + 1) * batch_size], self.y[idx * batch_size: (idx + 1) * batch_size])


def main_train(args):

    train_gen = Dataset(args.train_dir, args.batch_size)
    test_gen = Dataset(args.test_dir, args.infer_batch_size)

    K.set_image_data_format('channels_first')
    model = LenetBuilder()
    # model = resnet.ResnetBuilder.build_resnet_18([3, 224, 224], len(pos_list_en))

    optimizer = keras.optimizers.rmsprop(learning_rate=0.0008)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy', top2_accuracy])
    model.summary()

    csv_logger = CSVLogger(os.path.join(args.log_dir, 'log.csv'))
    ckpt_name = "model_{epoch:02d}.hdf5"
    ckpt_helper = ModelCheckpoint(os.path.join(args.ckpt_dir, ckpt_name),
                                  monitor='val_loss', verbose=1, save_weights_only=True, mode='auto', period=1)

    model.fit_generator(
        train_gen,
        validation_data=test_gen,
        epochs=args.max_epoch,
        verbose=1,
        max_queue_size=100,
        callbacks=[csv_logger, ckpt_helper])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # path
    parser.add_argument('--data_dir', type=str, default='./dataset/light-pos/')
    parser.add_argument('--log_dir', type=str, default='./run/')
    parser.add_argument('--ckpt_dir', type=str, default='./Records/')
    # training
    parser.add_argument('--max_epoch', type=int, default=35)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--infer_batch_size', type=int, default=2)

    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        raise FileNotFoundError('Data dir does not exist.')

    args.train_dir = os.path.join(args.data_dir, 'train')
    args.test_dir = os.path.join(args.data_dir, 'test')

    if not os.path.exists(args.train_dir) or not os.path.exists(args.test_dir):
        raise FileNotFoundError('Train dir or test dir does not exist.')

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    main_train(args)
