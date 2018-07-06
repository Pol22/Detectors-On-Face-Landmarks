# instal opencv 
# > pip install opencv-python

# install dlib
# download Python Wheel file (.whl) with dlib on Windows
# > pip intall dlib-file.whl

# install numpy
# > pip install numpy

import argparse
import cv2
import dlib
import numpy as np
from os import listdir, makedirs
from os.path import isfile, join, isdir
from time import time


detector = dlib.get_frontal_face_detector()


def img_to_landmarks_array(image, predictor):
    detected = detector(image, 1)
    if len(detected) < 1:
        return np.empty(0)
    shape = predictor(image, detected[0])
    points = np.empty(shape=68*2)
    x_range = detected[0].right() - detected[0].left()
    y_range = detected[0].bottom() - detected[0].top()
    for i in range(68):
        points[2*i] = (shape.part(i).x - detected[0].left()) / x_range
        points[2*i+1] = (shape.part(i).y - detected[0].top()) / y_range
    return points


def write_landmarks_to_file(points, filename):
    with open(filename, 'w') as f:
        for i in range(len(points)):
            f.write(str(points[i]) + ' ')
        f.write('\n')


def create_landmarks_for_dir(dirpath, resultpath, predictor):
    onlyfiles = [f for f in listdir(dirpath) if isfile(join(dirpath, f))]
    if not isdir(resultpath):
        makedirs(resultpath)
    counter = 0
    for filename in onlyfiles:
        try:
            img = cv2.imread(join(dirpath, filename), cv2.IMREAD_GRAYSCALE)
            img = cv2.equalizeHist(img)
            points = img_to_landmarks_array(img, predictor)
            if len(points) == 0:
                print('Landmarks not found in file:', join(dirpath, filename))
                continue
            write_landmarks_to_file(points, join(resultpath, (filename.split('.')[0] + '.txt')))
            counter = counter + 1
        except:
            print('Can\'t open file like image:', join(dirpath, filename))
    print('Landmarks write to directory: %s' % resultpath)
    return counter


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('predictor_file', type=str, help='path to shape predictor file for face landmarks')
    parser.add_argument('images_path', type=str, help='path to directory with images')
    parser.add_argument('landmarks_path', type=str, help='path to directory with landmarks files')
    args = parser.parse_args()
    predictor = dlib.shape_predictor(args.predictor_file)
    time0 = time()
    num_landmarks = create_landmarks_for_dir(args.images_path, args.landmarks_path, predictor)
    time1 = time()
    print('Number of created landmarks: ', num_landmarks)
    print('Average elapsed time for facelandmark detection per image: %0.4f sec' % ((time1 - time0) / num_landmarks))
