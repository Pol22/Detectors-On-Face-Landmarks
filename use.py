import argparse
import dlib
import cv2
import numpy as np
from sklearn.externals import joblib
from os import listdir, makedirs
from os.path import isfile, join, isdir
from create_landmarks import img_to_landmarks_array


mouth_open_threshold = 0.05

def check_mouth_open(points):
    # distances between points (61, 67) (62, 66) (63, 65)
    dist1 = np.sqrt((points[61*2]-points[67*2])**2 +(points[61*2+1]-points[67*2+1])**2)
    dist2 = np.sqrt((points[62*2]-points[66*2])**2 +(points[62*2+1]-points[66*2+1])**2)
    dist3 = np.sqrt((points[63*2]-points[65*2])**2 +(points[63*2+1]-points[65*2+1])**2)
    sum_dist = dist1 + dist2 + dist3
    if sum_dist > mouth_open_threshold:
        return 1.0
    return 0.0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('predictor_file', type=str, help='path to shape predictor file for face landmarks')
    parser.add_argument('images_path', type=str, help='path to directory with image files')
    parser.add_argument('smile_model_file', type=str, help='file name with saved model')
    parser.add_argument('open_mouth_file', type=str, help='file name with image names with open mouth')
    parser.add_argument('smile_file', type=str, help='file name with image names with smile')
    args = parser.parse_args()
    predictor = dlib.shape_predictor(args.predictor_file)
    # Load models 
    smile_clf = joblib.load(args.smile_model_file)
    onlyfiles = [f for f in listdir(args.images_path) if isfile(join(args.images_path, f))]
    with open(args.smile_file, 'w') as f_smile, open(args.open_mouth_file, 'w') as f_mouth:
        for filename in onlyfiles:
            try:
                img = cv2.imread(join(args.images_path, filename), cv2.IMREAD_GRAYSCALE)
                img = cv2.equalizeHist(img)
                points = img_to_landmarks_array(img, predictor)
                if len(points) == 0:
                    print('Landmarks not found in file:', join(args.images_path, filename))
                    continue
                points = np.reshape(points, [1, 136])
                predict_smile = smile_clf.predict(points)
                predict_mouth = check_mouth_open(points[0])
                if predict_smile[0] > 0.5:
                    f_smile.write(filename)
                    f_smile.write('\n')
                if predict_mouth > 0.5:
                    f_mouth.write(filename)
                    f_mouth.write('\n')
            except:
                print('Can\'t open file like image:', join(args.images_path, filename))
    
    print('Image names with smile write to file:', args.smile_file)
    print('Image names with open mouth write to file:', args.open_mouth_file)
