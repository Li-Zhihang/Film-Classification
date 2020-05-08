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

        # run inference
        leftover = not (datalen % self.batch_size == 0)
        batch_num = datalen // self.batch_size + leftover

        light_type = np.ndarray((datalen, 2), dtype=int)
        light_score = np.ndarray((datalen, 2))
        for batch_idx in range(batch_num):
            sta = batch_idx * self.batch_size
            stp = min(datalen, (batch_idx + 1) * self.batch_size)
            face_inputs = faces[sta: stp]
            preds = self.model.predict(face_inputs)
            top2 = tf.nn.top_k(preds, 2)
            light_score[sta: stp] = top2[0].numpy()
            light_type[sta: stp] = top2[1].numpy()

        return light_type, light_score, hasFace
