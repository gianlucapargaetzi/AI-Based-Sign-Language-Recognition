<p>This repository is the result of the project "AI-Based Sign Language Recognition", which
implements the MediaPipe Pipepline to estimate and classify hand gestures from the ASL Alphabet. 

The first part implements the recognition of static gestures with MediaPipe Hands and a proposed skip connections model for the classifier, trained with a big skeleton dataset extracted from image datasets available online
The trained classifier model was implemented in a simple iOS-Demo-Application to show the Real-Time Potential on an Edge Device
Additionally, a proof-of-principle for the recognition of dynamic Gestures was realised by training an LSTM-Network with Skeleton Sequences extracted with MediaPipe Holistic


------------------------------------------------------------------------------------------------------------

# /DatasetCreator for Static Skeletons

------------------------------------------------------------------------------------------------------------

This folder contains different python tools which were programmed to collect a dataset and augment it for
neural network training.

------------------------------------------------------------------------------------------------------------

## ./DataSelection.ipynb

<p> With this file, a specified amount of data is copied (or selected: deleting non selected data) to the desired output folder. Herefore,
every image is processed with MediaPipe. If hand was detected with high confidence picture is a candidate for the output dataset. 
Of all candidate files, then the amount specified gets randomly selected for the dataset. </p>
<p> Additional annotation can be set. To sort out bad image files (either no or wrong hand skeleton) every skeleton was visually controlled with the annotated skeleton </p>

------------------------------------------------------------------------------------------------------------

## ./Image Augmentation/DataAugMain.python

<p> This program is used to augment an input dataset (in this case the collection of data created with DataSelection.ipynb). 
In the GUI, input and output is selectable as well as the augmentation options can be configured 
(Which and how many augmentation should be created and if they should be annotated with the keypoints)</p>

------------------------------------------------------------------------------------------------------------

## ./CSVGenerator.ipynb

<p> This program exports the image testdata in a directory to a .csv files containing the media pipe keypoint coordinates.</p>
<p> To do so, the programm calculates mean value and standard deviation for the distance of each keypoint to the wrist. 
Datasets with at least one keypoint distance bigger than (sigma_mult) x (calculated standard deviation) will be discarded</p>

------------------------------------------------------------------------------------------------------------

# /Static Gesture Recognition with MediaPipe Hands

------------------------------------------------------------------------------------------------------------

This is the proof of principle for static hand gesture recognition for the self built dataset 
in form of a .csv file.

<p>
The NeuralNetwork.ipynb programm implements a network, its training and the real-time camera environmnent to
test mediapipe hands performance and gesture classication.
The SkipConnection Network delivers very good result and is a reliable network for static gesture classification
</p>

------------------------------------------------------------------------------------------------------------

## training_camera_test.py

- Implementation and Training of Networks with available .csv datasets 
- Camera-Interface for RealTime testing (Live Skeleton Estimation + Gesture Classification)


### Model2 (optimized Model): Skip Connections
- Implementation of Fully Connected Network with Skip Connections (around 23k Params)
- Loss landscape simplification
- Training results with augmented dataset (ca. 150k Images): 95.5%


------------------------------------------------------------------------------------------------------------

## /SkipConnections/

Saved Pre Trained Model Weights with data.csv, "Model2" structure in NeuralNetwork.ipynb

------------------------------------------------------------------------------------------------------------

# /DatasetCreator for Dynamic Gestures

------------------------------------------------------------------------------------------------------------

This folder contains different python tools which were programmed to record and generate a dataset of skeleton sequences

------------------------------------------------------------------------------------------------------------

## ./Sequence_Generator.ipynb

<p> With this file gestures can be recorded using a normal PC webcam <p>
Of all candidate files, then the amount specified gets randomly selected for the dataset. </p>
<p> Additional annotation can be set. To sort out bad image files (either no or wrong hand skeleton) every skeleton was visually controlled with the annotated skeleton </p>

------------------------------------------------------------------------------------------------------------

## ./Image Augmentation/DataAugMain.python

<p> This program is used to augment an input dataset (in this case the collection of data created with DataSelection.ipynb). 
In the GUI, input and output is selectable as well as the augmentation options can be configured 
(Which and how many augmentation should be created and if they should be annotated with the keypoints)</p>

------------------------------------------------------------------------------------------------------------

## ./SequencAugmentationTool

<p> This program analyses the recorded gesture videos and is used to export the skeleton sequence with the specified amount of augmentations </p>

------------------------------------------------------------------------------------------------------------

# /Dynamic Gesture Recognition with MediaPipe Holistic

------------------------------------------------------------------------------------------------------------

This is the proof of principle for dynamic hand gesture recognition for a small self-recorded Dataset

------------------------------------------------------------------------------------------------------------

# /Edge Device Implementation in iOS App
------------------------------------------------------------------------------------------------------------

This is the edge-device implementation of the recognition of static gestures in an iOS-App, consisting of the Xcode project and the converter script to generate the Core ML model
