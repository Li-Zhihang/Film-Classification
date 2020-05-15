import argparse
import os

import keras
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


def top2_accuracy_onehot(y_true, y_pred):
    '''y_true must be onehot'''
    return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=2)


def get_prediction(sidx, model):
    x = np.load(os.path.join(args.data_dir, 'pose_' + scale_name_cn[sidx] + '.npy'))
    y = sidx * np.ones((x.shape[0]), dtype=int)
    y_onehot = keras.utils.to_categorical(y, len(scale_name_cn))
    datalen = x.shape[0]

    leftover = not (datalen % args.batch_size == 0)
    batch_num = datalen // args.batch_size + leftover

    preds = np.ndarray((datalen, len(scale_name_cn)), dtype=int)
    logits = np.ndarray((datalen,), dtype=int)
    for batch_index in range(batch_num):
        start = batch_index * args.batch_size
        stop = min((batch_index + 1) * args.batch_size, datalen)

        pred = model.predict(x[start:stop])
        preds[start:stop] = pred
        logits[start:stop] = np.argmax(pred, axis=-1)

    acc = np.equal(logits, y)
    top2_acc = keras.metrics.top_k_categorical_accuracy(y_onehot, preds, k=2)

    print('Scale Type: {}'.format(scale_name_cn[sidx]))
    print('Data Number: {:d}'.format(datalen))
    print('Accuracy: {:.2f}'.format(np.mean(acc)))
    print('Top2 Accuracy: {:.2f}\n'.format(np.mean(top2_acc)))


def main(args):
    model = perceptron()
    model.load_weights('./models/scale_mlp/model_18.hdf5')
    for sidx in range(len(scale_name_cn)):
        get_prediction(sidx, model)
    # get_prediction(4, model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', dest='data_dir', default='D:/scale_dataset')
    parser.add_argument('--batch_size', type=int, default=20)
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        raise FileNotFoundError

    main(args)
