"""
This is a python script used to train the recognition model to dynamic gestures with an lstm structure
"""
import mediapipe as mp
import sklearn.model_selection
import tensorflow as tf

import NetworkStructures

import numpy as np

import cv2
import os
import csv

from tqdm import tqdm

import time

import SequenceProcessing
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from SequenceProcessing import SkeletonProcessor

#####################################################################################
####### Configure these global params
#####################################################################################
doFit = True ## doFit = True: Train Model, doFit = False: load trained model from file
doPlots = True
modelCallbacks = False ## Save "best_loss" and "best_accuracy" models when training
numEpochs = 150 ## Amount of epochs to be trained
batchSize = 4000
doPlots = True ## doPlots = True: Show plot of loss and accuracy as well as confusion matrix


model2 = NetworkStructures.LSTM_V3()

main_path = os.path.join('Out')

csv_path = os.path.join(main_path + 'log_LSTM_V2_full_V11.log')

train_dataset_path = 'train/train_merged/KeyPoints' ## Training CSV-File
validation_dataset_path = 'val/val_merged/KeyPoints' ## Training CSV-File



skel_proc = SkeletonProcessor()

label_map = {label:num for num, label in enumerate(SequenceProcessing.classes)}
print(label_map)


sequences, labels = [], []
if doFit:
    for letter in SequenceProcessing.classes:
        sequence_list = os.listdir(os.path.join(train_dataset_path, letter))

        for sequence in tqdm(range(len(sequence_list)), desc=letter + ': Import Sequences'):
            window = []
            for frame_num in range(SequenceProcessing.SEQUENCE_LENGTH):
                res = np.load(train_dataset_path + "/" + letter + "/" + sequence_list[sequence] + "/{}.npy".format(frame_num))
                window.append(res)

            sequences.append(window)
            labels.append(label_map[letter])

    X = np.array(sequences)

    y = np.array(labels)
    
    x_train, sequences, y_train, labels = sklearn.model_selection.train_test_split(X,y,test_size=0.05)
       
    sequences, labels = [], []
    
    for letter in SequenceProcessing.classes:
        sequence_list = os.listdir(os.path.join(validation_dataset_path, letter))

        for sequence in tqdm(range(len(sequence_list)), desc=letter + ': Import Sequences'):
            window = []
            for frame_num in range(SequenceProcessing.SEQUENCE_LENGTH):
                res = np.load(validation_dataset_path + "/" + letter + "/" + sequence_list[sequence] + "/{}.npy".format(frame_num))
                window.append(res)

            sequences.append(window)
            labels.append(label_map[letter])

    X = np.array(sequences)

    
    x_val = X
    y_val = np.array(labels)


########################################################################################
##### Model Training
########################################################################################
csv_callback = tf.keras.callbacks.CSVLogger(csv_path, separator=";", append=False)


model2.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])


if doFit is True:
    # save weights to file
    # train model with imported training data
    history = model2.fit(x_train, y_train,
                         epochs=numEpochs,
                         batch_size = batchSize,
                         callbacks=[csv_callback],
                         validation_data=(x_val, y_val))
    model2.evaluate(x_val, y_val)
    model2.save(main_path + '/saved_model_v11.h5')

    
## estimator class used to plot confusion matrix using scikit
class estimator:
    _estimator_type = ''
    classes_=[]
    def __init__(self, model, classes):
        self.model = model
        self._estimator_type = 'classifier'
        self.classes_ = SequenceProcessing.classes
    def predict(self, X):
        y_prob= self.model.predict(X)
        y_pred = y_prob.argmax(axis=1)
        return y_pred


if doPlots:
    plt.rcParams['figure.dpi'] = 350

    results_augmented = np.array(np.genfromtxt(csv_path, delimiter=';',skip_header=1))


    fig = plt.figure(figsize=(14,4))
    gs = fig.add_gridspec(1,2)
    ax_acc = fig.add_subplot(gs[0, 0])
    plt.plot(results_augmented[:,0], results_augmented[:,1]*100,'--',linewidth=1.5, color='darkblue')

    plt.plot(results_augmented[:,0], results_augmented[:,3]*100,'-', linewidth=2.5, color='blue')

    print('Augmented Accuracy: ', np.max(results_augmented[:,3]*100))

    plt.legend(['Training', 'Validation'])
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy [%]')
    plt.xlim([0,np.max(results_augmented[:,0])])
    plt.ylim([0,100])
    plt.grid()

    ax_loss = fig.add_subplot(gs[0, 1])
    plt.plot(results_augmented[:,0], results_augmented[:,2],'--',linewidth=1.5, color='darkblue')

    plt.plot(results_augmented[:,0], results_augmented[:,4],'-', linewidth=2.5, color='blue')


    plt.legend(['Training', 'Validation'])

    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xlim([0,np.max(results_augmented[:,0])])
    plt.grid()

    plt.savefig(main_path +'/results.svg', format='svg')
    
    results = np.array(np.genfromtxt(csv_path, delimiter=';',skip_header=1)) ## Load training_log

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
    classifier = estimator(model2, SequenceProcessing.classes)
    ConfusionMatrixDisplay.from_estimator(estimator=classifier, X=x_val, y=y_val,display_labels=SequenceProcessing.classes, 
                                          normalize='true',cmap='Blues',ax=ax_cm,colorbar=False)
    ax_cm.set_title('Confusion Matrix')
    plt.savefig(main_path + '/conf.svg', format='svg')

        
        
        
