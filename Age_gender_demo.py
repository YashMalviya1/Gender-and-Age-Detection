
import keras
import json
import sys
import tensorflow as tf
from keras.layers import Input
import numpy as np
import argparse
from wide_resnet import WideResNet
from keras.utils.data_utils import get_file
import face_recognition



gender_model = tf.keras.models.load_model('weights.hdf5')
gender_model.summary()

age_map=[['0-2'],['4-6'],['8-13'],['15-20'],['25-32'],['38-43'],['48-63'],['60+']]


    def detect_face(self):

        frame=cv2.imread("image-07.jpg")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


        #image = face_recognition.load_image_file("image-02.jpg")
        image=frame
        face_locations = face_recognition.face_locations(image)
        print(face_locations)
        cv2.rectangle(frame, (face_locations[0][3], face_locations[0][0]), (face_locations[0][1], face_locations[0][2]), (255, 200, 0), 2)
        

        img=frame[face_locations[0][0]-30: face_locations[0][2]+30, face_locations[0][3]-30: face_locations[0][1]+30]

        # predict ages and genders of the detected faces
        img2= cv2.resize(img, (64, 64))
        img2=np.array([img2]).reshape((1, 64,64,3))
        results = self.model.predict(img2)
        predicted_genders = results[0]
        gen="F" if predicted_genders[0][0] > 0.5 else "M"
        ages = np.arange(0, 101).reshape(101, 1)
        predicted_ages = results[1].dot(ages).flatten()
        # draw results
        pred=""
        print((predicted_ages))
        print(gen)
        pred=str(int(predicted_ages[0]))+" "+str(gen)
        print(pred)
        cv2.putText(frame, pred,(face_locations[0][3],face_locations[0][0]) , cv2.FONT_HERSHEY_SIMPLEX,0.7, (2, 255, 255), 2)
        

        cv2.imshow('Gender and age', frame)
        if cv2.waitKey(50000) == 27:  # ESC key press
            cv2.destroyAllWindows()