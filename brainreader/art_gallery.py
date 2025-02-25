from fileman.file_getter import get_file
from scipy.ndimage.io import imread
from scipy.misc import imresize
import os
# from PIL import Image

__author__ = 'peter'


IMAGE_COLLECTION = {
    'wanderer': 'http://i.imgur.com/lR6wTNw.jpg',
    'dinnertime': 'http://i.imgur.com/nRczfmr.jpg',
    'limbo': 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3c/Follower_of_Jheronimus_Bosch_Christ_in_Limbo.jpg/1024px-Follower_of_Jheronimus_Bosch_Christ_in_Limbo.jpg',
    'manchester_newyear': 'http://i.imgur.com/UDEIMHp.jpg',
    'lenna': 'https://upload.wikimedia.org/wikipedia/en/2/24/Lenna.png',
    'starry_night': 'https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/800px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg',
    'nektarfront': 'https://upload.wikimedia.org/wikipedia/commons/0/00/Tuebingen_Neckarfront.jpg',
    'the_shipwreck_of_the_minotaur': 'https://upload.wikimedia.org/wikipedia/commons/2/2e/Shipwreck_turner.jpg',
    'scream': 'https://en.wikipedia.org/wiki/The_Scream#/media/File:The_Scream.jpg',
    'femme_nue_assise': 'https://upload.wikimedia.org/wikipedia/en/8/8f/Pablo_Picasso%2C_1909-10%2C_Figure_dans_un_Fauteuil_%28Seated_Nude%2C_Femme_nue_assise%29%2C_oil_on_canvas%2C_92.1_x_73_cm%2C_Tate_Modern%2C_London.jpg',
    'composition_vii': 'https://upload.wikimedia.org/wikipedia/commons/b/b4/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg',
    'trump': 'http://media4.s-nbcnews.com/j/newscms/2016_04/1396246/donald-trump-mug_5fea106e0eb494469a75e60d8f2b18ea.nbcnews-fp-320-320.jpeg'
    }

def get_image(name, size = None):
    """
    Get an image by name.
    :param name: A string identifying an image from our dictionary.
    :return:
    """
    assert name in IMAGE_COLLECTION, "We don't have the image '%s' in the gallary" % (name, )
    _, ext = os.path.splitext(IMAGE_COLLECTION[name])
    relative_path = os.path.join('images', name)+ext
    filename = get_file(
        relative_name = relative_path,
        url = IMAGE_COLLECTION[name],
        )
    im_array = imread(filename)
    if size is not None:
        im_array = imresize(im_array, get_new_size(im_array.shape[:2], new_size=size))
    return im_array


def get_new_size((cy, cx), new_size):
    ny, nx = (None, None) if new_size is None else new_size
    aspect_ratio = float(cy)/cx
    if ny is None and nx is None:
        return cy
    elif nx is None:
        nx = int(ny * aspect_ratio)
    elif ny is None:
        ny = int(nx / aspect_ratio)
    return ny, nx


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    im_name = 'starry_night'
    im = get_image('starry_night', size=(128, None))
    plt.imshow(im)
    plt.title(im_name)
    plt.show()
