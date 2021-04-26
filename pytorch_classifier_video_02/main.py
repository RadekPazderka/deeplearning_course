
import argparse

from pytorch_classifier_video_02.animal_trainer import AnimalTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--train_dir', dest="train_dir", type=str, required=True, help='Directory of train images.')
    parser.add_argument('--val_dir', dest="val_dir", type=str, required=True, help='Directory of val images.')
    parser.add_argument('--checkpoint_dir', dest="checkpoint_dir", type=str, required=True, help='Directory of model checkpoints.')
    parser.add_argument('--pretrained_model', dest="pretrained_model", default=None, type=str, help='Path to pretrained checkpoint / model.')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # TRAIN_DIR = r'C:\Users\darkwolf\PycharmProjects\deeplearning_course\caffe_classifier_video_02\dataset\data\TRAIN'
    # VAL_DIR = r'C:\Users\darkwolf\PycharmProjects\deeplearning_course\caffe_classifier_video_02\dataset\data\TRAIN'
    # CHECKPOINT_DIR = "checkpoints/"
    args = parse_args()


    animal_trainer = AnimalTrainer(args.train_dir, args.val_dir, args.checkpoint_dir, args.pretrained_model)
    # animal_trainer = AnimalTrainer(TRAIN_DIR, VAL_DIR, CHECKPOINT_DIR)

    animal_trainer.train()
    animal_trainer.validate()
