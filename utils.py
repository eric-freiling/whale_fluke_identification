from os.path import join, exists
from os import listdir, makedirs
import cv2
from keras.models import model_from_json


def rgb2gray(im):
    gray = 0.2989 * im[:, :, 0] + 0.5870 * im[:, :, 1] + 0.1140 * im[:, :, 2]

    return gray


def transform_image(im):
    # convert to gray scale
    gray_im = rgb2gray(im)
    resized_im = cv2.resize(gray_im, (100, 50))

    return resized_im


def transform_dir(input_dir, output_dir):
    if not exists(output_dir):
        makedirs(output_dir)
    files = listdir(input_dir)
    num_files = len(files)
    counter = 1
    for f in files:
        if counter % 1000 == 0:
            print("Transformed {}/{} images".format(counter, num_files))
        image = cv2.imread(join(input_dir, f))
        trans_image = transform_image(image)
        cv2.imwrite(join(output_dir, f), trans_image)
        counter += 1


def save_model(model, model_name):
    model.save_weights(model_name + '.h5')

    # Save the model architecture
    with open(model_name + '.json', 'w') as f:
        f.write(model.to_json())


def load_model(model_name):
    # Model reconstruction from JSON file
    with open(model_name + '.json', 'r') as f:
        model = model_from_json(f.read())

    # Load weights into the new model
    model.load_weights(model_name + '.h5')

    return model
