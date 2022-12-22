This is the proof of principle for **static hand gesture recognition** for the self built dataset 
in form of a .csv file.

The NeuralNetwork.ipynb programm implements a network, its training and the real-time camera environmnent to
test mediapipe hands performance and gesture classication.
The SkipConnection Network delivers very good results and is a reliable network for static gesture classification

------------------------------------------------------------------------------------------------------------

# NeuralNetwork.ipynb / NeuralNetwork.py
- Implementation and Training of Networks with available .csv datasets 
- Camera-Interface for RealTime testing (Live Skeleton Estimation + Gesture Classification)

Model1 (Prototype Model): Fully Connected; NOT RECOMMENDED!
- Implementation of Fully Connected Network (around 1.27M Params)
- first deep network as simple prototype: Slow loss minimization, tends to overfitting
- With less deep networks, real-time performance is poor

Model2 (optimized Model): Skip Connections
- Implementation of Fully Connected Network with Skip Connections (around 23k Params)
- Loss landscape simplification: Faster and more efficient training, more reliable results than Model1

## NOTES FOR USERS:

Adjustable Params:  

- doFit: 		  {True,False} 			    Wether model should be retrained or the pretrained model should be loaded
- modelCallbacks: {True,False} 			    Wether model weights (min. loss, max. accuracy) should be saved during training
- doPlots:		  {True,False} 			    Wether training results should be plotted
- numEpochs: 	  int			 		    Amount of training epochs
- selDataset: 	  {Augmented, Unaugmented}  Which Dataset / Modelweights should be used for training, or camera interface

If everything works properly after running the python script a camera window should open, in which a real-time
Alphabet Recognition is implemented (refer to Google Pictures for Alphabet in ASL)

------------------------------------------------------------------------------------------------------------


------------------------------------------------------------------------------------------------------------

# RawData.csv			
Raw dataset (selection of internet dataset), unaugmented  
Nr of TestData: 5856  

Accuracy with Skip Connections Model: 85%  

------------------------------------------------------------------------------------------------------------


------------------------------------------------------------------------------------------------------------

# data.csv
Augmented dataset (RawData.csv: Flipped + 30 Augmentations per Gestures)  
Nr of TestData: 153254  

Accuracy with Skip Connections Model: 95%  

------------------------------------------------------------------------------------------------------------


------------------------------------------------------------------------------------------------------------

# validation.csv
Validation dataset (Skeletons of around 1000 Hand Pose Images created by ourselves)  
Nr of TestData: 1063  

------------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------------------------------------------

# /SkipConnections/
Saved Pre Trained Model + Training Results for Unaugmented and Augmented Datasets  

------------------------------------------------------------------------------------------------------------


