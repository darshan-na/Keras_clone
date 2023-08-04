import os
import cv2
import numpy as np
def load_images_labels(data_dir, img_size):
    categories = ['car', 'truck']
    images = []
    labels = []

    for category in categories:
        category_path = os.path.join(data_dir, category)
        label = categories.index(category)
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (img_size, img_size))
            images.append(img)
            labels.append(label)

    images = np.array(images).reshape(-1, img_size, img_size, 1)
    images_ = images.reshape(32*32,2059,1)
    labels = np.array(labels)
    labels_ = labels.reshape(1,2059,1)
    print(images_.shape)
    print(labels_.shape)
    return images_, labels_

train_dir = 'D:/cnn/train'
valid_dir = 'D:/cnn/valid'
img_size = 32

train_images, train_labels = load_images_labels(train_dir, img_size)
# valid_images, valid_labels = load_images_labels(valid_dir, img_size)
train_images = train_images / 255.0
# valid_images = valid_images / 255.0

np.save('train_images.npy', train_images)
np.save('train_labels.npy', train_labels)
# np.save('valid_images.npy', valid_images)
# np.save('valid_labels.npy', valid_labels)
