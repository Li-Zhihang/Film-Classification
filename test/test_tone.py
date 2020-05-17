import argparse
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


def perceptron(input_dim):
    net = keras.models.Sequential([
        keras.layers.Dense(200, input_shape=(input_dim,), activation='relu'),
        keras.layers.Dense(len(tone_en), activation='softmax')
    ])
    return net


def top2_accuracy_onehot(y_true, y_pred):
    '''y_true must be onehot'''
    return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=2)


def test(args):
    model = perceptron(args.input_dim)
    model.load_weights('./models/tone_mlp/model_44.hdf5')
    for tidx in range(len(tone_en)):
        path1 = os.path.join(args.data_dir, 'train', tone_en[tidx] + '.npy')
        path2 = os.path.join(args.data_dir, 'test', tone_en[tidx] + '.npy')
        hists1 = np.load(path1)
        hists2 = np.load(path2)
        x = np.concatenate((hists1, hists2), axis=0)
        y = tidx * np.ones((hists1.shape[0] + hists2.shape[0]), dtype=int)
        y_onehot = keras.utils.to_categorical(y, len(tone_en))

        preds = model.predict(x, batch_size=args.batch_size)
        logits = np.argmax(preds, axis=-1)

        acc = np.equal(logits, y)
        top2_acc = keras.metrics.top_k_categorical_accuracy(y_onehot, preds, k=2)

        print('Tone Type: {}'.format(tone_list[tidx]))
        print('Data Number: {:d}'.format(len(y)))
        print('Accuracy: {:.2f}'.format(np.mean(acc)))
        print('Top2 Acc: {:.2f}'.format(np.mean(top2_acc)))





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # path
    parser.add_argument('--data_dir', type=str, default='../tone/gray')
    # training
    parser.add_argument('--input_dim', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=20)

    args = parser.parse_args()

    test(args)
