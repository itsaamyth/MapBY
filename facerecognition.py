from cv2 import cv2
import numpy

# init camera
cap = cv2.VideoCapture(0)

# face detection
face_cascade = cv2.CascadeClassifier(
    "haar_cascade/haarcascade_frontalface_alt.xml")

skip = 0
face_data = []
dataset_path = './data/'

file_name = input("enter the name of person\n")
while(True):
    ret, frame = cap.read()

    if(ret == False):
        continue

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    # print(faces)
    faces = sorted(faces, key=lambda f: f[2]*f[3])

    # picking the last face as it have th largest area
    for face in faces[-1:]:
        x, y, w, h = face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

        # extract (crop out the required image) : region of interest
        offset = 10
        face_selection = frame[y-offset:y+h+offset, x-offset:x+w+offset]
        face_selection = cv2.resize(face_selection, (100, 100))

        # store every 10th face
        skip += 1
        if(skip % 10 == 0):
            face_data.append(face_selection)
            print(len(face_data))

    cv2.imshow("Frame", frame)
    #cv2.imshow("face selection",face_selection)

    key_pressed = cv2.waitKey(1) & 0xFF
    if(key_pressed == ord("s")):
        break
# convert face list in numpy array
face_data = numpy.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0], -1))
print(face_data.shape)

# save into file system

numpy.save(dataset_path+file_name+".npy", face_data)
print("data succesfully saved")

cap.release()
cv2.destroyAllWindows()
