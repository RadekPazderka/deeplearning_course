import os

import torch
from torch import nn
from torchvision import transforms

from  PIL import Image

from pytorch_classifier_video_02.label_mapping import DATASET_ID_MAPPING
from pytorch_classifier_video_02.model import VGG16


class AnimalTester(object):

    def __init__(self, checkpoint_path: str):
        self._checkpoint_path = checkpoint_path
        self._vgg16 = VGG16(10)
        self._vgg16.load_state_dict(torch.load(checkpoint_path))
        # self._vgg16.cuda()
        self._vgg16.eval()

        self._transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])


    def test_image_dir(self, image_dir_path):
        for image_name in os.listdir(image_dir_path):
            image_path = os.path.join(image_dir_path, image_name)
            self.test_image(image_path)

    def test_image(self, image_path: str):
        pil_img = Image.open(image_path)

        transformed = self._transform(pil_img)
        batch = transformed.unsqueeze(0)

        with torch.no_grad():
            _, outputs = self._vgg16(batch)
            _, predicted = torch.max(outputs.data, 1)

            class_id = predicted.numpy()[0]
            m = nn.Softmax(dim=1)
            percent = m(outputs).numpy().squeeze()[class_id]
            print("{}: {}".format(DATASET_ID_MAPPING[class_id], percent))


if __name__ == '__main__':
    CHECKPOINT_PATH = r"C:\Users\darkwolf\PycharmProjects\deeplearning_course\pytorch_classifier_video_02\checkpoints\vgg16_pretrained.pkl"
    IMAGE_PATH = r"C:\Users\darkwolf\PycharmProjects\deeplearning_course\caffe_classifier_video_02\dataset\data\VAL\cat"

    animal_tester = AnimalTester(CHECKPOINT_PATH)
    animal_tester.test_image_dir(IMAGE_PATH)

