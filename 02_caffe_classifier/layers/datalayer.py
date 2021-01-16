import caffe


class DatalayerClassifier(caffe.Layer):

    def setup(self, bottom, top):
        self._data_fetcher = None #TODO: here...

        BATCH_SIZE  = 32
        CHANNELS    = 3
        IMG_SIZE    = 224

        top[0].reshape(BATCH_SIZE, CHANNELS, IMG_SIZE, IMG_SIZE)    # input image
        top[1].reshape(BATCH_SIZE, 1)                               # input label

    def forward(self, bottom, top):
        image_blob, label_blob = self._data_fetcher.get_data()

        top[0].data[...] = image_blob
        top[1].data[...] = label_blob


    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

