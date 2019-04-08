import os 
from shutil import copyfile
import random 

cur_dir = os.path.dirname(__file__)
file_name_path = os.path.join(cur_dir, 'data/masked_images/data/trainval/')

# for cats
for breed in ['breeds_cat', 'breeds_dog']:

    path = os.path.join(file_name_path, breed)

    for each_breed in os.listdir(path):
        breed_dir = os.path.join(path, each_breed)
        imageList = []
        for img in os.listdir(breed_dir):
            imageList.append(img)
        random.shuffle(imageList)
        n = len(imageList)
        trainList = imageList[:int(0.8 * n)]
        valList = imageList[int(0.8 * n):]
        
        train_path = os.path.join("data/masked_images/data/train/" + breed, each_breed)
        val_path = os.path.join("data/masked_images/data/Sval/" + breed, each_breed)
        os.makedirs(train_path)
        os.makedirs(val_path)
        # copy image
        for img in trainList:
            copyfile(os.path.join(breed_dir, img), os.path.join(train_path, img))
        for img in valList:
            copyfile(os.path.join(breed_dir, img), os.path.join(val_path, img))

