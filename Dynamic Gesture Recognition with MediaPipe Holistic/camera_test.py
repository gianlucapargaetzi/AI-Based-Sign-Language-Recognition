"""
This is a python script used to test the camera input for dynamic gesture recognition.
If the frame rate is too low (as it is in most cases in this python script) the prediction brakes down and does not deliver accurate results

Accuracy for small dataset: 82 %
"""
import mediapipe as mp
import tensorflow as tf
import numpy as np

import SequenceProcessing


import cv2
## Tensorflow test
print('Num GPUs available:', len(tf.config.list_physical_devices('GPU')))
print('Tensorflow Version: ',tf. __version__) 
tf.debugging.set_log_device_placement(False)


#####################################################################################
####### Configure these global params
#####################################################################################
doFit = False ## doFit = True: Train Model, doFit = False: load trained model from file
modelCallbacks = False ## Save "best_loss" and "best_accuracy" models when training
numEpochs = 150 ## Amount of epochs to be trained
doPlots = True ## doPlots = True: Show plot of loss and accuracy as well as confusion matrix

####################################################################################

skel_proc = SequenceProcessing.SkeletonProcessor()

model2 = tf.keras.models.load_model('Results/saved_model_v11.h5')


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic



#tf.keras.utils.plot_model(model2, to_file='dynamic_mode.png', show_shapes=True, show_layer_names=False, show_layer_activations=True)


########################################################################################
##### Model Test in Camera Environment
########################################################################################
# For webcam input:
font                   = cv2.FONT_HERSHEY_SIMPLEX
position               = (0,300)
fontScale              = 2
fontColor              = (255,255,0)
fontThickness          = 5
minConfidence          = 0.9
                


seq = []
cap = cv2.VideoCapture(1) ##### Performance is dependant on the camera and processor speed (if the camera input is laggy, predictions breaks down)
with mp_holistic.Holistic(
                min_detection_confidence=0.5,
                smooth_landmarks=True,
                model_complexity=1,
                min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.

            
        results = holistic.process(image)
        image = skel_proc.annotate(image, results)
        vec = skel_proc.process_single(results)
        
        if len(seq) < SequenceProcessing.SEQUENCE_LENGTH:
            seq.append(vec)
            image = cv2.flip(image, 1)
        else:
            del seq[0]
            seq.append(vec) 
            tmp = np.array(seq)
            print(tmp.shape)
            tmp = tf.expand_dims(tmp, 0, name=None)
            predictions = model2.predict(tmp) 
            # Flip the image horizontally for a selfie-view display.
            image = cv2.flip(image, 1)
            if np.max(predictions)>minConfidence:
                label = SequenceProcessing.classes[np.argmax(predictions)]                
            else:
                label = '-'
            cv2.putText(image,label,position,font,fontScale,fontColor,thickness=fontThickness)   
            
        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
cv2.destroyAllWindows()            

