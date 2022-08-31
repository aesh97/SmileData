import os
from PIL import Image
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
import cv2
def main():
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    directory = r'/Users/adam/Desktop/smileData/datasets/train_folder/0'
    for file in os.listdir(directory):
        image = Image.open('/Users/adam/Desktop/smileData/datasets/train_folder/0/' + file).convert('L')
        image = image.resize((128,64))
        image = hog(image, orientations = 8, pixels_per_cell = (8,8), cells_per_block=(2,2)) 
        array= np.array(image)
        X_train.append(array)
        Y_train.append(0)
    directory = r'/Users/adam/Desktop/smileData/datasets/train_folder/1'
    for file in os.listdir(directory):
        image = Image.open('/Users/adam/Desktop/smileData/datasets/train_folder/1/' + file).convert('L')
        image = image.resize((128,64))
        image = hog(image, orientations = 8, pixels_per_cell = (8,8), cells_per_block=(2,2)) 
        array= np.array(image)
        X_train.append(array)
        Y_train.append(1)
    directory = r'/Users/adam/Desktop/smileData/datasets/test_folder/0'
    for file in os.listdir(directory):
        image = Image.open('/Users/adam/Desktop/smileData/datasets/test_folder/0/' + file).convert('L')
        image = image.resize((128,64))
        image = hog(image, orientations = 8, pixels_per_cell = (8,8), cells_per_block=(2,2))
        array= np.array(image)
        X_test.append(array)
        Y_test.append(0)

    directory = r'/Users/adam/Desktop/smileData/datasets/test_folder/1'
    for file in os.listdir(directory):
        image = Image.open('/Users/adam/Desktop/smileData/datasets/test_folder/1/' + file).convert('L')
        image = image.resize((128,64))
        image = hog(image, orientations = 8, pixels_per_cell = (8,8), cells_per_block=(2,2))
        array= np.array(image)

        X_test.append(array)
        Y_test.append(1)
  
    sgd = SGDClassifier(random_state=50, max_iter=100000, tol = 1e-6)
    sgd.fit(X_train, Y_train)
    
    print("Model is now trained")
    
    print("Position the front facing camera until your face is in the center of the image and press the 's' key!")
    cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
        cv2.imshow('Input', frame)
        c = cv2.waitKey(1)
        if c == 115:
            cv2.imshow("Saved", frame)
            
            cv2.imwrite("Saved.jpg", frame)
            image = Image.open('/Users/adam/Desktop/smileData/Saved.jpg').convert('L')
            image = image.resize((128,64))
            image = hog(image, orientations = 8, pixels_per_cell = (8,8), cells_per_block=(2,2))
            array= np.array(image)
            y_pred = sgd.predict([array])
            if (y_pred > 0.5):
                print("You smiled")
            else:
                print("no smile")

            break
    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()
