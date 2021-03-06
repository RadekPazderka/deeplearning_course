try:
    from pytorch_classifier_video_02.image_tests import AnimalTester
except:
    from .image_tests import AnimalTester


if __name__ == '__main__':
    CHECKPOINT_PATH = r"./checkpoints/vgg16_0048.pkl"
    IMAGE_DIR = r"./demo_images"

    animal_tester = AnimalTester(CHECKPOINT_PATH)
    animal_tester.test_image_dir(IMAGE_DIR)

