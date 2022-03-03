import os
import torch
import torch.nn as tnn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from tqdm import tqdm
from typing import Optional

from pytorch_classifier_video_02.model import VGG16


class AnimalTrainer(object):
    BATCH_SIZE = 10
    LEARNING_RATE = 0.01
    EPOCH = 100

    def __init__(self, dataset_train_dir: str,
                 dataset_val_dir: str,
                 checkpoint_dir: str,
                 pretrained_checkpoint: Optional[str]=None,
                 mode: str="GPU") -> None:
        self._dataset_train_dir = dataset_train_dir
        self._dataset_val_dir = dataset_val_dir
        self._checkpoint_dir = checkpoint_dir
        self._pretrained_checkpoint = pretrained_checkpoint
        self._mode = mode

        self._best_validitation_score = 0.0
        self._best_checkpoint = ""

        self._transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

    def train(self) -> None:
        vgg16 = self._get_vgg_model(self._pretrained_checkpoint)
        if self._mode == "GPU":
            vgg16.cuda()

        train_data = dsets.ImageFolder(self._dataset_train_dir, self._transform)
        train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=self.BATCH_SIZE, shuffle=True)

        # Loss, Optimizer
        cost = tnn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(vgg16.parameters(), lr=self.LEARNING_RATE)

        # Train the model
        for epoch in range(self.EPOCH):

            avg_loss = 0
            cnt = 0
            with tqdm(train_loader, colour='blue') as logger:
                for images, labels in train_loader:

                    if self._mode == "GPU":
                        images = images.cuda()
                        labels = labels.cuda()

                    # Forward + Backward + Optimize
                    optimizer.zero_grad()
                    outputs = vgg16(images)
                    loss = cost(outputs, labels)
                    avg_loss += loss.data
                    cnt += 1
                    loss.backward()
                    optimizer.step()
                    logger.set_description("[E: {}] loss: {}, avg_loss: {}, best checkpoint: {} ({} %)".format(epoch,
                                                                                              loss.data,
                                                                                              avg_loss / cnt,
                                                                                              os.path.basename(self._best_checkpoint),
                                                                                              self._best_validitation_score))
                    logger.update(1)

            checkpoint_path = self._save_checkpoint(epoch, vgg16)
            self._validate_checkpoint(checkpoint_path)

    def _save_checkpoint(self, epoch: int, model: VGG16) -> str:
        checkpoint_name = "vgg16_{:04}.pkl".format(epoch)
        checkpoint_path = os.path.join(self._checkpoint_dir, checkpoint_name)
        torch.save(model, checkpoint_path)
        return checkpoint_path

    def _get_vgg_model(self, pretrained_checkpoint: Optional[str]=None) -> VGG16:
        if (pretrained_checkpoint is None):
            vgg16 = VGG16(10)
        else:
            print("[INFO] Loading pretrained weights. ({})".format(pretrained_checkpoint))
            vgg16 = torch.load(pretrained_checkpoint)
        return vgg16

    def _validate_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint_name = os.path.basename(checkpoint_path)
        vgg16 = self._get_vgg_model(checkpoint_path)
        vgg16.cuda()
        vgg16.eval()

        correct = 0
        total = 0
        testData = dsets.ImageFolder(self._dataset_val_dir, self._transform)
        testLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=self.BATCH_SIZE, shuffle=False)
        with tqdm(testLoader, colour='green') as logger:
            for images, labels in testLoader:
                if self._mode == "GPU":
                    images = images.cuda()
                outputs = vgg16(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted.cpu() == labels).sum()
                curr_acc = (100.0 * float(correct) / float(total))

                logger.set_description("[Checkpoint: {}] Current acc: {:.2f} (correct: {} / {})".format(checkpoint_name, curr_acc, correct, total))
                logger.update(1)

            avg_acc = (100.0 * float(correct) / float(total))

        if avg_acc > self._best_validitation_score:
            self._best_validitation_score = avg_acc
            self._best_checkpoint = checkpoint_path

    def validate(self) -> None:
        for file_name in os.listdir(self._checkpoint_dir):
            if file_name.endswith(".pkl"):
                checkpoint_path = os.path.join(self._checkpoint_dir, file_name)
                self._validate_checkpoint(checkpoint_path)
