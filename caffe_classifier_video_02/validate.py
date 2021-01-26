import os
from .utils import sys_paths
import caffe

from caffe_classifier_video_02.dataset_fetcher.DataFetcher import DataFetcher


class ValidateWrapper():
    def __init__(self, deploy_path, caffemodels_dir) -> None:

        self._deploy_path = deploy_path
        self._caffemodels_dir = caffemodels_dir
        self._fetcher = DataFetcher("dataset/data/VAL", 224)
        self._fetcher.run()
        print("Fetching val data...")
        self._val_data = [self._fetcher.get_data_blob() for _ in range(1000)]

    def validate(self):
        result = []

        for file_name in os.listdir(self._caffemodels_dir):
            name, ext = os.path.splitext(file_name)
            if ext != ".caffemodel":
                continue

            correct = 0
            caffemodel_path = os.path.join(self._caffemodels_dir, file_name)
            net = caffe.Net(self._deploy_path, caffemodel_path, caffe.TEST)

            for value in self._val_data:
                img_blob = value["img_blob"]
                label_blob = value["label_blob"]

                net.blobs[name].reshape(*img_blob.shape)
                net.blobs[name].data[...] = img_blob

                output = net.forward()
                output = output["prob"][0]
                arg_max = output.argmax(axis=0)
                predicted_class = output[arg_max]

                if label_blob[0] == predicted_class:
                    correct += 1

            result.append({
                "caffemodel" : caffemodel_path,
                "accuracy": correct / len(self._val_data)
            })
        return result


if __name__ == '__main__':
    DEPLOY_PATH = "prototxt/animal_classifier/squeeze_net/deploy.prototxt"
    CAFFEMODELS_DIR = "model/animal_classifier/squeeze_net/"

    ValidateWrapper(DEPLOY_PATH, CAFFEMODELS_DIR).validate()