import os
import matplotlib.image as mpimg

pairs = "pairs=[np.zeros((batch_size, h, w,1)) for i in range(2)] - list of 2, each is shape of [32 (batch), 105, 105, 1]"
targets = "np.zeros((batch_size,)) - sie of 32 (batch)"

images_pairs_train_text_path = "G:\\My Drive\\BGU\\4th Year\\8th Semester\\Introduction to Deep Learning\\Assignment 2\\Data\\pairsDevTrain.txt"
images_dir_path = "G:\\My Drive\\BGU\\4th Year\\8th Semester\\Introduction to Deep Learning\\Assignment 2\\Data\\lfw2\\lfw2"


# returns a pre number of zeros based on required length
def get_pre_number(image_name, required_digits=4):
    pre_number = ""
    for i in range(len(image_name), required_digits):
        pre_number += "0"

    return pre_number


# returns an image from path
def read_image(image_path):
    return mpimg.imread(image_path)


# returns a pair if same face
def same_face(split_by_tab):
    name = split_by_tab[0]
    image1 = split_by_tab[1]
    image2 = split_by_tab[2].split("\n")[0]

    image1_path = os.path.join(os.path.join(images_dir_path, name),
                               name + "_" + get_pre_number(image1) + image1 + ".jpg")
    image2_path = os.path.join(os.path.join(images_dir_path, name),
                               name + "_" + get_pre_number(image2) + image2 + ".jpg")

    return read_image(image1_path), read_image(image2_path)


# returns a pair if different face
def diff_face(split_by_tab):
    name1 = split_by_tab[0]
    image1 = split_by_tab[1]
    name2 = split_by_tab[2]
    image2 = split_by_tab[3].split("\n")[0]

    image1_path = os.path.join(os.path.join(images_dir_path, name1),
                               name1 + "_" + get_pre_number(image1) + image1 + ".jpg")
    image2_path = os.path.join(os.path.join(images_dir_path, name2),
                               name2 + "_" + get_pre_number(image2) + image2 + ".jpg")

    return read_image(image1_path), read_image(image2_path)


# loads images to x and y arrays based on text file
def load_images():
    x = []
    y = []

    f = open(images_pairs_train_text_path, "r").readlines()
    for line_index, line_value in enumerate(f[1:]):
        split_by_tab = line_value.split("\t")

        # same face
        if len(split_by_tab) == 3:
            x.append(same_face(split_by_tab))
            y.append(1)
        # different face
        else:
            x.append(diff_face(split_by_tab))
            y.append(0)

    return x, y


x, y = load_images()

