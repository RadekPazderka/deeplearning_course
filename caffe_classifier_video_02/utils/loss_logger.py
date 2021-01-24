import csv
import os

from datetime import datetime


class Loss_logger(object):
    FIELD_NAMES = ["iter", "loss", "time"]

    def __init__(self, snapshot_prefix, nth_iter_save=50):
        self._csv_path = snapshot_prefix + "_loss.csv"
        self._nth_iter_save = nth_iter_save
        self._data = {"iter": [], "loss" : []}

    def add_loss(self, iter, loss):
        self._data["iter"].append(int(iter))
        self._data["loss"].append(float(loss))

        size = len(self._data["iter"])
        if size >= self._nth_iter_save:
            fragment = self._data["loss"][0:self._nth_iter_save]
            avg_loss = sum(fragment) / len(fragment)

            self._add_to_csv(self._data["iter"][self._nth_iter_save-1], avg_loss)
            self._data["loss"] = self._data["loss"][self._nth_iter_save :]
            self._data["iter"] = self._data["iter"][self._nth_iter_save :]


    def _add_to_csv(self, iter, loss):

        if not os.path.exists(self._csv_path):
            with open(self._csv_path, mode='w') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=self.FIELD_NAMES)
                writer.writeheader()

        current_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        data = [iter, loss, current_time]
        with open(self._csv_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(data)

