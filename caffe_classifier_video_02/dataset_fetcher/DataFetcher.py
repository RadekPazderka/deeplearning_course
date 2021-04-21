import os
from threading import Thread
from functools import wraps
import Queue
import cv2
import numpy as np

# from caffe_classifier_video_02.dataset_fetcher.Augmentator import AugmentatorSeq
from caffe_classifier_video_02.dataset_fetcher.LabelMapping import LABEL_MAPPING


def run_async(func):
    @wraps(func)
    def async_func(*args, **kwargs):
        func_hl = Thread(target = func, args = args, kwargs = kwargs)
        func_hl.daemon = True
        func_hl.start()
        return func_hl
    return async_func

class DataFetcher(object):
    def __init__(self, root_path, image_size, batch_size=32, async_workers=1):
        self._root_path = root_path
        self._image_size = image_size
        self._batch_size = batch_size
        self._async_workers = async_workers
        self._dataset_queue = Queue.Queue(maxsize=100)

    @run_async
    def _run_worker(self):
        worker = Worker(root_dataset_path=self._root_path,
                        batch_size=self._batch_size,
                        image_size=self._image_size,
                        queue=self._dataset_queue)
        worker.run()

    def run(self):
        for i in range(self._async_workers):
            self._run_worker()

    def get_data_blob(self):
        return self._dataset_queue.get()


class DatasetBalancer():
    def __init__(self, dataset_path):
        self._dataset_path = dataset_path
        self._paths, self._dataset_state, self._classes = self._load_paths()

    def _load_paths(self):
        classes = os.listdir(self._dataset_path)
        res = {}
        dataset_state = {}

        for dataset_class_name in classes:
            dataset_class_dir = os.path.join(self._dataset_path, dataset_class_name)
            res[dataset_class_name] = [os.path.join(dataset_class_dir, image_name) for image_name in os.listdir(dataset_class_dir)]
            dataset_state[dataset_class_name] = {
                "current_index": 0,
                "max_index": len(res[dataset_class_name])-1
            }
        return res, dataset_state, classes

    def get_iterator(self):
        while True:
            for dataset_class in self._classes:
                yield self._paths[dataset_class][self._dataset_state[dataset_class]["current_index"]], dataset_class

                self._dataset_state[dataset_class]["current_index"] += 1
                self._dataset_state[dataset_class]["current_index"] %= self._dataset_state[dataset_class]["max_index"]


class Worker():
    def __init__(self, root_dataset_path, batch_size, image_size, queue):
        self._root_dataset_path = root_dataset_path
        self._batch_size = batch_size
        self._image_size = image_size
        self._queue = queue


    def run(self):
        balander = DatasetBalancer(self._root_dataset_path)

        iterator = balander.get_iterator()

        while True:

            images = []
            labels = []

            for i in range(self._batch_size):
                img_path, img_label = next(iterator)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (self._image_size, self._image_size))
                images.append(img)
                labels.append(LABEL_MAPPING[img_label])

            # image_blob = AugmentatorSeq(images=images)
            image_blob = np.concatenate(tuple(map(lambda x: np.expand_dims(x, axis=0), images)), axis=0)
            image_blob = image_blob.transpose((0, 3, 1, 2))
            label_blob = np.expand_dims(np.array(labels), axis=1)

            self._queue_put({
                "img_blob": image_blob,
                "label_blob": label_blob
            })

    def _queue_put(self, item):
        self._queue.put(item)



