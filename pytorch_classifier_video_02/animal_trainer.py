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

    def __init__(self, dataset_train_dir: str, dataset_val_dir: str, checkpoint_dir: str) -> None:
        self._dataset_train_dir = dataset_train_dir
        self._dataset_val_dir = dataset_val_dir
        self._checkpoint_dir = checkpoint_dir
        self._best_validitation_score = 0.0
        self._best_checkpoint = ""

        self._vgg16 = VGG16(n_classes=10)
        self._vgg16.cuda()

        self._transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

    def train(self) -> None:
        train_data = dsets.ImageFolder(self._dataset_train_dir, self._transform)
        train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=self.BATCH_SIZE, shuffle=True)

        # Loss, Optimizer & Scheduler
        cost = tnn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self._vgg16.parameters(), lr=self.LEARNING_RATE)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1000, 0.8)
        # Train the model
        for epoch in range(self.EPOCH):

            avg_loss = 0
            cnt = 0
            for images, labels in tqdm(train_loader, desc="Epoch: {}".format(epoch)):
                images = images.cuda()
                labels = labels.cuda()

                # Forward + Backward + Optimize
                optimizer.zero_grad()
                _, outputs = self._vgg16(images)
                loss = cost(outputs, labels)
                avg_loss += loss.data
                cnt += 1
                print("[E: {}] loss: {}, avg_loss: {}, LR: {}, best checkpoint: {} ({} %)".format(
                    epoch, loss.data, avg_loss / cnt, scheduler.get_lr(), os.path.basename(self._best_checkpoint), self._best_validitation_score ))
                loss.backward()
                optimizer.step()

            scheduler.step(epoch)


            checkpoint_path = self._save_checkpoint(epoch)
            #self._validate_checkpoint(checkpoint_path)


    def _save_checkpoint(self, epoch: int) -> str:
            # Save model checkpoint
        checkpoint_name = "vgg16_{:04}.pkl".format(epoch)
        checkpoint_path = os.path.join(self._checkpoint_dir, checkpoint_name)
        torch.save(self._vgg16.state_dict(), checkpoint_path)
        return checkpoint_path


    def _load_weights(self, path: str) -> None:
        self._vgg16.load_state_dict(torch.load(path))
        self._vgg16.eval()

    def _validate_checkpoint(self, checkpoint_path: str) -> None:
        self._load_weights(checkpoint_path)

        correct = 0
        total = 0
        testData = dsets.ImageFolder(self._dataset_val_dir, self._transform)
        testLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=self.BATCH_SIZE, shuffle=False)

        for images, labels in tqdm(testLoader):
            images = images.cuda()
            _, outputs = self._vgg16(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.cpu() == labels).sum()

        avg_acc = (100 * correct / total)
        if avg_acc > self._best_validitation_score:
            self._best_validitation_score = avg_acc
            self._best_checkpoint = checkpoint_path

        print("{} = avg acc: {}, correct: {}, total: {}".format(os.path.basename(checkpoint_path), avg_acc, correct, total))

    def validate(self) -> None:

        for file_name in os.listdir(self._checkpoint_dir):
            if file_name.endswith(".pkl"):
                checkpoint_path = os.path.join(self._checkpoint_dir, file_name)
                self._validate_checkpoint(checkpoint_path)


