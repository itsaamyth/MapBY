from cv2 import cv2
import numpy as np
import os
from database import push_data
# knn


def distance(v1, v2):
    return np.sqrt(((v1-v2)**2).sum())


def knn(train, test, k=5):
    dist = []
    for i in range(train.shape[0]):
        ix = train[i, :-1]
        iy = train[i, -1]

        d = distance(test, ix)
        dist.append([d, iy])
    dk = sorted(dist, key=lambda x: x[0] > [k])

    labels = np.array(dk)[:, -1]

    output = np.unique(labels, return_counts=True)

    index = np.argmax(output[1])
    return output[0][index]
# knn


# init camera
cap = cv2.VideoCapture(0)

# face detection
face_cascade = cv2.CascadeClassifier(
    "haar_cascade/haarcascade_frontalface_alt.xml")

skip = 0
dataset_path = './data/'
face_data = []
labels = []

class_id = 0  # labels for the given file
names = {}  # mapping btw id - name


name_set = set()

# Data Preparation
for fx in os.listdir(dataset_path):  # to load the file in the folder
    if fx.endswith(".npy"):
        # create a mapping btw clas label and
        names[class_id] = fx[:-4]
        data_item = np.load(dataset_path+fx)
        face_data.append(data_item)

        # create labels for class
        target = class_id*np.ones((data_item.shape[0],))
        class_id += 1
        labels.append(target)

face_dataset = np.concatenate(face_data, axis=0)
face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))

trainset = np.concatenate((face_dataset, face_labels), axis=1)

# testing

while True:
    ret, frame = cap.read()
    if(ret == False):
        continue

    faces = face_cascade.detectMultiScale(frame, 1.3, 5)

    for face in faces:

        x, y, w, h = face
        # get the face roi

        offset = 10
        face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
        face_section = cv2.resize(face_section, (100, 100))

        out = knn(trainset, face_section.flatten())
        # print(out)

        # display a name and a rectangle around it
        pred_name = names[int(out)]
        # print(pred_name)
        name_set.add(pred_name)

        cv2.putText(frame, pred_name, (x, y-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "If Your name shows up Please press s", (x, y-60),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 200), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.imshow("Faces", frame)
    key = cv2.waitKey(1) & 0xFF
    if(key == ord("s")):
        break

for i in name_set:
    push_data(str(i), "P")


cap.release()
cv2.destroyAllWindows()
