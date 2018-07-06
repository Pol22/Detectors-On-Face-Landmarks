import argparse
import numpy as np
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.externals import joblib

# http://mplab.ucsd.edu/wordpress/?page_id=398
# http://mplab.ucsd.edu/wordpress/wp-content/uploads/genki4k.tar

def create_labels(labelsfile):
    with open(labelsfile, 'r') as f:
        read = f.read()
        arr = read.split()
        result = []
        for i in range(0, len(arr), 4):
            result.append(float(arr[i]))
        return result


def create_features_and_labels(landmarkspath, labels):
    onlyfiles = [f for f in listdir(landmarkspath) if isfile(join(landmarkspath, f))]
    features = []
    ret_labels = []
    for filename in onlyfiles:
        with open(join(landmarkspath, filename), 'r') as f:
            floats = list(map(float, f.read().split()))
            features.append(floats)
            label_num = int(filename.split('.')[0][4:]) - 1
            ret_labels.append(labels[label_num])
    return (np.array(features), np.array(ret_labels))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('labels_file', type=str, help='path to file with labels')
    parser.add_argument('landmarks_path', type=str, help='path to directory with landmarks files')
    parser.add_argument('model_file', type=str, help='file name with saved model')
    args = parser.parse_args()
    all_labels = create_labels(args.labels_file)
    features, labels = create_features_and_labels(args.landmarks_path, all_labels)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=2153)
    clf = GradientBoostingClassifier(n_estimators=200, learning_rate=1.0, random_state=232)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('Confusion matrix:')
    print(confusion_matrix(y_test, y_pred))
    print('Classification report:')
    print(classification_report(y_test, y_pred))
    joblib.dump(clf, args.model_file)
    print('Model saved to file:', args.model_file)
    