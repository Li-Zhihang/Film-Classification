import numpy as np
import tensorflow as tf
import keras.backend as K

from .utils import LenetBuilder, crop_face_batch


class LightPose(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        K.set_image_data_format('channels_first')

        self.model = LenetBuilder()
        self.model.load_weights('./models/lenet/model_28.hdf5')
        print('successfully load lightpose model.')

    def get_lightpose(self, imgs):
        datalen = imgs.shape[0]
        # get faces
        faces, hasFace = crop_face_batch(imgs, self.batch_size)

        preds = self.model.predict(faces, batch_size=self.batch_size)

        top2 = tf.nn.top_k(preds, 2)
        light_score = top2[0].numpy()
        light_type = top2[1].numpy()
        # light_score1 = np.max(preds, axis=-1)
        # light_type1 = np.argmax(preds, axis=-1)

        # for k in range(datalen):
        #     if not hasFace[k]:
        #         light_type1[k] = -1

        return light_type, light_score, hasFace
