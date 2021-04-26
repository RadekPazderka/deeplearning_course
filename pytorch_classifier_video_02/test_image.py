import os

import torch
import torch.nn as tnn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from tqdm import tqdm

from pytorch_classifier_video_02.model import VGG16



class AnimalTester(object):

    @staticmethod
    def test_image(checkpoint_path: str, image_dir: str):
        testData = dsets.ImageFolder(self._dataset_val_dir, self._transform)
        testLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=self.BATCH_SIZE, shuffle=False)

        for images, labels in tqdm(testLoader):
            images = images.cuda()
            _, outputs = self._vgg16(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.cpu() == labels).sum()
        pass


if __name__ == '__main__':
    checkpoint_path = ""
