from pytorch_classifier_video_02.image_tests import AnimalTester

"""
download pretrained checkpoint from: 
    https://drive.google.com/file/d/1I53jXYnyKcNoQYxXA_cQv1A0pmKFo3v3/view?usp=sharing
"""


if __name__ == '__main__':
    CHECKPOINT_PATH = r"./pretrained_checkpoints/vgg16_0055.pkl"
    IMAGE_DIR = r"./demo_images"

    animal_tester = AnimalTester(CHECKPOINT_PATH)
    animal_tester.test_image_dir(IMAGE_DIR)
