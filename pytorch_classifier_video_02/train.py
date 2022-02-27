import torch
import argparse
try:
    from pytorch_classifier_video_02.animal_trainer import AnimalTrainer
except:
    from .animal_trainer import AnimalTrainer

def parse_args():
    parser = argparse.ArgumentParser(description='Train + val VGG model')
    parser.add_argument('--train_dir', dest="train_dir", default="./dataset/data/TRAIN", type=str,  help='Directory of train images.')
    parser.add_argument('--val_dir', dest="val_dir", type=str, default="./dataset/data/VAL",  help='Directory of val images.')
    parser.add_argument('--checkpoint_dir', dest="checkpoint_dir", default="./checkpoints", type=str,  help='Directory of model checkpoints.')
    parser.add_argument('--pretrained_model', dest="pretrained_model", default=None, type=str, help='Path to pretrained checkpoint / model.')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    a = torch.cuda.is_available()
    # vgg16 = torch.load(r"/data_slow/course/src/deeplearning_course/pytorch_classifier_video_02/checkpoints/vgg16_0048.pkl", map_location=torch.device('cpu'))
    # torch.save(vgg16.state_dict(), "new_checkpoint.pkl")
    args = parse_args()
    animal_trainer = AnimalTrainer(args.train_dir, args.val_dir, args.checkpoint_dir, args.pretrained_model)

    animal_trainer.train()
    animal_trainer.validate()
