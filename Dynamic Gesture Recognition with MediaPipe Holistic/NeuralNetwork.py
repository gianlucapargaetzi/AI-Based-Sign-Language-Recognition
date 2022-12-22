"""
This is a python script used to train and test a simple classifier model for ASL Alphabet using MediaPipe Hands

The Skip Connections Model with about 23k Params is trained with a .csv File containing 
hand keypoint coordinates (created with MediaPipeHands) representing the letters A-Z (excl. Z and J) 

Accuracy for Unaugmented Data: 85 %
Accuracy for Augmented Data: 96 %
"""
import mediapipe as mp
import tensorflow as tf
import numpy as np

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

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


model2 = tf.keras.models.load_model('saved_model_v11.h5')


tf.keras.utils.plot_model(model2, to_file='dynamic_mode.png', show_shapes=True, show_layer_names=False, show_layer_activations=True)