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

import cv2
import os
import csv   

from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


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

selDataset = 'Augmented' ## Either 'Augmented' or 'Unaugmented'

## Use next line if filedialog should be used to select test dataset
# train_dataset_path = filedialog.askopenfilename()

train_dataset_path = 'data.csv' ## Training CSV-File
val_dataset_path = 'validation.csv' ## Validatrion CSV-File

classes = ['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']

if selDataset == 'Augmented':
    train_dataset_path = 'data.csv'
    batchSize = 30000
elif selDataset == 'Unaugmented':
    train_dataset_path = 'RawData.csv'
    batchSize = 1000  
    
main_path = 'SkipConnections/' + selDataset ## Directory where Training results (model + training_log) will be / are stored
training_log_path = main_path + '/training.log'
####################################################################################

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

plt.rcParams['figure.dpi'] = 350


"""
This function reads a .csv file and extracts the datavector (21 3D keypoint coordinates) and corresponding labels
filename:    path to file
test_size:   amount of data used for test dataset
val_size:    amount of data used for the validation dataset
"""
def convertCSVtoData(filename, test_size=0, val_size = 0):
    
    #load daa
    train_data = np.genfromtxt(filename,delimiter=';')

    
    train_labels = train_data[:,0].astype(np.uint8)
    train_data = train_data[:,1:]
    train_data = train_data.reshape((train_data.shape[0],21,3))
    test_data = np.zeros((test_size,21,3))
    test_labels = np.zeros(test_size)    
    val_data = np.zeros((val_size,21,3))
    val_labels = np.zeros(val_size) 
    
    # take test_size amount of random samples of train_data to create test dataset
    for i in range(test_size):
        idx= np.random.randint(train_data.shape[0]) 
        test_data[i,:,:] = train_data[idx,:,:]
        test_labels[i] = train_labels[idx]
        train_data = np.delete(train_data,idx,0)
        train_labels = np.delete(train_labels,idx)        
    
    # take val_size amount of random samples of train_data to create ValidationDataset
    for i in range(val_size):
        idx= np.random.randint(train_data.shape[0]) 
        val_data[i,:,:] = train_data[idx,:,:]
        val_labels[i] = train_labels[idx]
        train_data = np.delete(train_data,idx,0)
        train_labels = np.delete(train_labels,idx)  
        
    return train_data, train_labels,test_data, test_labels, val_data, val_labels


## estimator class used to plot confusion matrix using scikit
class estimator:
    _estimator_type = ''
    classes_=[]
    def __init__(self, model, classes):
        self.model = model
        self._estimator_type = 'classifier'
        self.classes_ = classes
    def predict(self, X):
        y_prob= self.model.predict(X)
        y_pred = y_prob.argmax(axis=1)
        return y_pred

"""
file path has to point to the input data as .csv file ('RawData.csv', 'data.csv')
"""
# First file dialog to select Input csv data (Testdata)

print('Training Dataset: ' + train_dataset_path)
train_data,train_label, _, _, _,_ = convertCSVtoData(train_dataset_path)
print('Amount of Training Data: ' + str(train_data.shape[0]))

print('Validation Dataset: ' + val_dataset_path)
val_data,val_label, _ , _ , _ , _  = convertCSVtoData(val_dataset_path)  
print('Amount of Validation Data: ' + str(val_data.shape[0]))


#tf.keras.utils.plot_model(model1, to_file='simple_fully_connected_model.png', show_shapes=True, show_layer_names=False, show_layer_activations=True)

########################################################################################
##### Skip Connections Network Structure : Best results with layer-wise batch normalization
########################################################################################
input_net = tf.keras.layers.Input(shape=(21,3))
flat = tf.keras.layers.Flatten()(input_net)
bn0 = tf.keras.layers.BatchNormalization()(flat)

dense1 = tf.keras.layers.Dense(63,activation='swish')(bn0)
bn1 = tf.keras.layers.BatchNormalization()(dense1)

dense2 = tf.keras.layers.Dense(63,activation='swish')(dense1)
bn2 = tf.keras.layers.BatchNormalization()(dense2)

dense3 = tf.keras.layers.Dense(63,activation='swish')(dense2)
dropout1 = tf.keras.layers.Dropout(0.2)(dense3)
bn3 = tf.keras.layers.BatchNormalization()(dropout1)

add1 = tf.keras.layers.Add()([bn2, bn3])
dense4 = tf.keras.layers.Dense(63,activation='swish')(add1)
bn4 = tf.keras.layers.BatchNormalization()(dense4)

add2 = tf.keras.layers.Add()([bn4, bn1])

dense5 = tf.keras.layers.Dense(63,activation='swish')(add2)
bn5 = tf.keras.layers.BatchNormalization()(dense5)
add3 = tf.keras.layers.Add()([bn0, bn5])

out = tf.keras.layers.Dense(len(classes))(add3)

model2 = tf.keras.models.Model(inputs=input_net, outputs=out)
model2.summary()
#tf.keras.utils.plot_model(model2, to_file='skip_connections_model.png', show_shapes=True, show_layer_names=False, show_layer_activations=True)




########################################################################################
##### Model Training
########################################################################################
callback_csv = tf.keras.callbacks.CSVLogger(training_log_path,separator=';')


model2.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


if doFit is True:
    # save weights to file
    if modelCallbacks:
        # Save model with best results (Loss and Accuracy)
        best_loss_path = main_path + '/BestLoss/cp.ckpt'
        cp_callback_loss = tf.keras.callbacks.ModelCheckpoint(filepath=best_loss_path,
                                                                monitor='val_loss',
                                                                mode='min',
                                                                save_best_only=True,
                                                                save_weights_only=True,
                                                                verbose=0)

        best_accuracy_path = main_path + '/BestAccuracy/cp.ckpt'
        cp_callback_acc = tf.keras.callbacks.ModelCheckpoint(filepath=best_accuracy_path,
                                                                monitor='val_accuracy',
                                                                mode='max',
                                                                save_best_only=True,
                                                                save_weights_only=True,
                                                                verbose=0)    
        callback_list = [cp_callback_loss,cp_callback_acc,callback_csv]
    else:
        callback_list = [callback_csv]
    # train model with imported training data
    history = model2.fit(train_data, train_label, 
                        epochs=numEpochs,
                        batch_size=batchSize, 
                        validation_data=(val_data, val_label),
                        callbacks=callback_list)
    
    model2.save(main_path + '/saved_model.h5')
    
else:
    model2 = tf.keras.models.load_model(main_path + '/saved_model.h5')
    
print("Model Validation Results:")   
results = model2.evaluate(val_data, val_label)

probability_model = tf.keras.Sequential([model2, 
                                         tf.keras.layers.Softmax()])



########################################################################################
##### Plot Model Results
########################################################################################     
if doPlots:
    results = np.array(np.genfromtxt(training_log_path, delimiter=';',skip_header=1)) ## Load training_log

    fig = plt.figure(figsize=(18,10))
    gs = fig.add_gridspec(2,2,width_ratios=[1.2,2])
    ax_acc = fig.add_subplot(gs[0, 0])
    plt.plot(results[:,0], results[:,1]*100,'-',linewidth=3, color='darkgreen') ## Training accuracy
    plt.plot(results[:,0], results[:,3]*100,'-', linewidth=2, color='green') ## Validation accuracy
    plt.legend(['Training', 'Validation'])
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy [%]')
    plt.xlim([0,np.max(results[:,0])])
    plt.ylim([0,100])
    plt.grid()

    ax_loss = fig.add_subplot(gs[1, 0])
    plt.plot(results[:,0], results[:,2],'-',linewidth=3, color='darkgreen') ## Training loss
    plt.plot(results[:,0], results[:,4],'-', linewidth=2, color='green') ## Validation loss
    plt.legend(['Training', 'Validation'])
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()

    ax_cm = fig.add_subplot(gs[:, 1])
    classifier = estimator(probability_model, classes)
    ConfusionMatrixDisplay.from_estimator(estimator=classifier, X=val_data, y=val_label,display_labels=classes, 
                                          normalize='true',cmap='Blues',ax=ax_cm,colorbar=False)
    ax_cm.set_title('Confusion Matrix')


    if doFit:
        plt.savefig(main_path+'/results.svg', format='svg')


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

cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=1,               # Critical Parameter: Simplified Model (model_complexity < 1) delivers worse results
    min_detection_confidence=0.5,
    max_num_hands=1,
    min_tracking_confidence=0.8) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        vec = np.zeros((21,3))
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                for j in range(21):
                    lm = np.array([hand_landmarks.landmark[j].x,hand_landmarks.landmark[j].y,hand_landmarks.landmark[j].z], dtype=float)
                    vec[j,:] = lm
            input_vec = np.array([vec])
        
            # Flip the image horizontally for a selfie-view display.
            predictions = probability_model.predict(input_vec)
            image = cv2.flip(image, 1)
            if np.max(predictions)>minConfidence:
                label = classes[np.argmax(predictions)]                
            else:
                label = '-'
            cv2.putText(image,label,position,font,fontScale,fontColor,thickness=fontThickness)              

        else:
            image = cv2.flip(image, 1) 
        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
cv2.destroyAllWindows()            