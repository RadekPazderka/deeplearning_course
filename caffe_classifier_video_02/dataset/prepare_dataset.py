import os
import zipfile
import shutil

"""
1) Download dataset from "https://www.kaggle.com/alessiocorrado99/animals10/download"
2) Set DATASET_PATH
3) Run this script for preparation dataset.
"""

translate = {
    "cane": "dog",
    "cavallo": "horse",
    "elefante": "elephant",
    "farfalla": "butterfly",
    "gallina": "chicken",
    "gatto": "cat",
    "mucca": "cow",
    "pecora": "sheep",
    "scoiattolo": "squirrel",
    "dog": "cane",
    "elephant" : "elefante",
    "butterfly": "farfalla",
    "chicken": "gallina",
    "cat": "gatto",
    "cow": "mucca",
    "spider": "ragno",
    "ragno": "spider",
    "squirrel": "scoiattolo"
}

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

if __name__ == '__main__':
    DATASET_PATH = "archive.zip"

    print("Extract archive: {}".format(DATASET_PATH))
    input_zip=zipfile.ZipFile(DATASET_PATH)
    input_zip.extractall("tmp")

    train_path  = create_dir(os.path.join("data", "TRAIN"))
    val_path    = create_dir(os.path.join("data", "VAL"))

    print("Split dataset into train and val directory...")
    root_dir = os.path.join("tmp", "raw-img")
    for dataset_class_name in os.listdir(root_dir):
        dataset_class_dir = os.path.join(root_dir, dataset_class_name)

        images_names = os.listdir(dataset_class_dir)
        val_images = images_names[:100]
        train_images = images_names[100:]

        output_val_class_dir = create_dir(os.path.join(val_path, translate[dataset_class_name]))
        for val_image in val_images:
            source_image_path = os.path.join(dataset_class_dir, val_image)
            dest_image_path = os.path.join(output_val_class_dir, val_image)
            os.rename(source_image_path, dest_image_path)

        output_train_class_dir = create_dir(os.path.join(train_path, translate[dataset_class_name]))
        for train_image in train_images:
            source_image_path = os.path.join(dataset_class_dir, train_image)
            dest_image_path = os.path.join(output_train_class_dir, train_image)
            os.rename(source_image_path, dest_image_path)

    print("Remove tempolary directory...")
    shutil.rmtree("tmp")

