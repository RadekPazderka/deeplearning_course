import caffe
from ..dataset_fetcher.DataFetcher import DataFetcher

class DatalayerClassifier(caffe.Layer):

    def setup(self, bottom, top):

        BATCH_SIZE  = 32
        CHANNELS    = 3
        IMG_SIZE    = 227

        self._data_fetcher = DataFetcher("../dataset/data/TRAIN", batch_size=BATCH_SIZE, image_size=IMG_SIZE)
        self._data_fetcher.run()

        top[0].reshape(BATCH_SIZE, CHANNELS, IMG_SIZE, IMG_SIZE)    # input image
        top[1].reshape(BATCH_SIZE, 1)                               # input label

    def forward(self, bottom, top):
        blob = self._data_fetcher.get_data_blob()

        top[0].data[...] = blob["img_blob"]
        top[1].data[...] = blob["label_blob"]


    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

