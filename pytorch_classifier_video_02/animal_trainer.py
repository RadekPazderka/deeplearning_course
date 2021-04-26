import os

import torch
import torch.nn as tnn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from tqdm import tqdm

from pytorch_classifier_video_02.model import VGG16


class AnimalTrainer(object):
    BATCH_SIZE = 10
    LEARNING_RATE = 0.01
    EPOCH = 50

    def __init__(self, dataset_train_dir: str, dataset_val_dir: str, checkpoint_dir: str):
        self._dataset_train_dir = dataset_train_dir
        self._dataset_val_dir = dataset_val_dir
        self._checkpoint_dir = checkpoint_dir

        self._vgg16 = VGG16(n_classes=10)
        # self._vgg16.cuda()

        self._transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

    def train(self):
        train_data = dsets.ImageFolder(self._dataset_train_dir, self._transform)
        train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=self.BATCH_SIZE, shuffle=True)

        # Loss, Optimizer & Scheduler
        cost = tnn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self._vgg16.parameters(), lr=self.LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

        # Train the model
        for epoch in range(self.EPOCH):

            avg_loss = 0
            cnt = 0
            for images, labels in tqdm(train_loader, desc="Epoch: {}".format(epoch)):
                # images = images.cuda()
                # labels = labels.cuda()

                # Forward + Backward + Optimize
                optimizer.zero_grad()
                _, outputs = self._vgg16(images)
                loss = cost(outputs, labels)
                avg_loss += loss.data
                cnt += 1
                print("[E: %d] loss: %f, avg_loss: %f" % (epoch, loss.data, avg_loss / cnt))
                loss.backward()
                optimizer.step()

            scheduler.step(avg_loss)
            self._save_checkpoint(epoch)

    def _save_checkpoint(self, epoch: int):
            # Save model checkpoint
        checkpoint_name = "vgg16_{:04}.pkl".format(epoch)
        checkpoint_path = os.path.join(self._checkpoint_dir, checkpoint_name)
        torch.save(self._vgg16.state_dict(), checkpoint_path)



    def _load_weights(self, path: str):
        self._vgg16.load_state_dict(torch.load(path))
        self._vgg16.eval()

    def validate(self):

        for file_name in os.listdir(self._checkpoint_dir):
            if file_name.endswith(".pkl"):
                checkpoint_path = os.path.join(self._checkpoint_dir, file_name)

                self._load_weights(checkpoint_path)

                correct = 0
                total = 0
                testData = dsets.ImageFolder(self._dataset_val_dir, self._transform)
                testLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=self.BATCH_SIZE, shuffle=False)

                for images, labels in tqdm(testLoader):
                    # images = images.cuda()
                    _, outputs = self._vgg16(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted.cpu() == labels).sum()

                avg_acc =  (100 * correct / total)
                print("{} = avg acc: {}, correct: {}, total: {}".format(file_name, avg_acc, correct, total))


if __name__ == '__main__':
    TRAIN_DIR = r'../caffe_classifier_video_02/dataset/data/TRAIN'
    VAL_DIR = r'../caffe_classifier_video_02/dataset/data/VAL'
    CHECKPOINT_DIR = "checkpoints/"
    print(torch.device("cpu"))
    AnimalTrainer(TRAIN_DIR, TRAIN_DIR, CHECKPOINT_DIR).train()
    # AnimalTrainer(TRAIN_DIR, TRAIN_DIR, CHECKPOINT_DIR).validate()
