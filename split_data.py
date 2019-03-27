import os
from shutil import copyfile

if __name__ == '__main__':
    cur_dir = os.path.dirname(__file__)
    file_name_path = os.path.join(cur_dir, 'data/images/')
    i = 0
    trainval = open('data/annotations/trainval.txt', "r")
    test = open('data/annotations/test.txt', "r")
    for line in trainval:
        if '#' in line:
            continue
        l = line.split()
        image_name = l[0] + ".jpg"
        breed_name = l[0][0: l[0].rfind('_')]
        class_id = l[1]
        breed = l[3]
        if l[2] == '1':
            species = 'cat'
        else:
            species = 'dog'
        if not os.path.exists("data/trainval/species/cat/"):
            os.makedirs("data/trainval/species/cat/")
        if not os.path.exists("data/trainval/species/dog/"):
            os.makedirs("data/trainval/species/dog/")
        copyfile(file_name_path + image_name, "data/trainval/species/"+ species + '/' + image_name)
        if species == 'cat':
            if not os.path.exists("data/trainval/breeds_cat/"+ breed_name):
                os.makedirs("data/trainval/breeds_cat/"+ breed_name)
            copyfile(file_name_path + image_name, "data/trainval/breeds_cat/"+ breed_name + '/' + image_name)
        if species == 'dog':
            if not os.path.exists("data/trainval/breeds_dog/" + breed_name):
                os.makedirs("data/trainval/breeds_cat/" + breed_name)
            copyfile(file_name_path + image_name, "data/trainval/breeds_dog/" + breed_name + '/' + image_name)

    for line in test:
        if '#' in line:
            continue
        l = line.split()
        image_name = l[0] + ".jpg"
        breed_name = l[0][0: l[0].rfind('_')]
        class_id = l[1]
        breed = l[3]
        if l[2] == '1':
            species = 'cat'
        else:
            species = 'dog'

        if not os.path.exists("data/test/species/cat/"):
            os.makedirs("data/test/species/cat/")
        if not os.path.exists("data/test/species/dog/"):
            os.makedirs("data/test/species/dog/")
        copyfile(file_name_path + image_name, "data/test/species/"+ species + '/' + image_name)
        if species == 'cat':
            if not os.path.exists("data/test/breeds_cat/"+ breed_name):
                os.makedirs("data/test/breeds_cat/"+ breed_name)
            copyfile(file_name_path + image_name, "data/trainval/breeds_cat/"+ breed_name + '/' + image_name)
        if species == 'dog':
            if not os.path.exists("data/test/breeds_dog/" + breed_name):
                os.makedirs("data/test/breeds_dog/" + breed_name)
            copyfile(file_name_path + image_name, "data/test/breeds_dog/" + breed_name + '/' + image_name)


 # map = {"Abyssinian":197,
    #     "american_bulldog":199,
    #     "american_pit_bull_terrier":199,
    #     "basset_hound":199,
    #     "beagle":199,
    #     "Bengal":199,
    #     "Birman":199,
    #     "Bombay":183,
    #     "boxer":198,
    #     "British_Shorthair":199,
    #     "chihuahua":199,
    #     "Egyptian_Mau":189,
    #     "english_cocker_spaniel":195,
    #     "english_setter":199,
    #     "german_shorthaired":199,
    #     "great_pyrenees":199,
    #     "havanese":199,
    #     "japanese_chin":199,
    #     "keeshond":198,
    #     "leonberger":199,
    #     "Maine_Coon":199,
    #     "miniature_pinscher":199,
    #     "newfoundland":195,
    #     "Persian":199,
    #     "pomeranian":199,
    #     "pug":199,
    #     "Ragdoll":199,
    #     "Russian_Blue":199,
    #     "saint_bernard":199,
    #     "samoyed":199,
    #     "scottish_terrier":198,
    #     "shiba_inu":199,
    #     "Siamese":198,
    #     "Sphynx":199,
    #     "staffordshire_bull_terrier":188,
    #     "wheaten_terrier":199,
    #     "yorkshire_terrier":199
    #     }