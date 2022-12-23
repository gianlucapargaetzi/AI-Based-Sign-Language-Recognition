"""#########################################################################"""
"""Import necessary packages"""
"""#########################################################################"""
"""Imports"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # Supress unnecessary messages
#from keras.models import load_model
import keras
import coremltools as ct
import pickle
import numpy as np

"""#########################################################################"""
"""Define color constants and path to model"""
"""#########################################################################"""
"""Path to model"""
pathModel = 'saved_model.h5'
"""Colors for messages"""
HEADER = '\033[95m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'
"""Print the Script started message"""
print('')
print('*************************Script started****************************')

"""#########################################################################"""
"""Class labels of the neural network"""
"""#########################################################################"""
class_labels = ['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P',
                'Q','R','S','T','U','V','W','X','Y']

"""#########################################################################"""
"""Try to load the trained neural network"""
"""#########################################################################"""
print('************************Import .h5 model***************************')
try:
    model = keras.models.load_model(pathModel)
except:
    print(f"{FAIL}[ERROR] Keras model could not be imported!{ENDC}")
    exit() #End Programm execution
else:
    print(f"{OKGREEN}[INFO] Imported keras model successfully!{ENDC}")

"""#########################################################################"""
"""Convert the model to coreml format"""
"""#########################################################################"""
print('******************Convert Model to coreml model********************')
input_shape = ct.Shape(shape=(21,3))
try:
    coreml_model = ct.convert(model,
                              inputs = [ct.TensorType(shape=input_shape, name="input_1")])
except:
    print(f"{FAIL}[ERROR] Keras model could not be converted to coreml model!{ENDC}")
    exit() #End Programm execution
else:
    print(f"{OKGREEN}[INFO] Keras model converted to coreml model successfully!{ENDC}")

"""#########################################################################"""
"""Test converted model with two predictions"""
"""#########################################################################"""
print('************************Test coreml model**************************')
test_input_A = np.array([[0.45512241,0.61341757,-3.26E-07],
                       [0.5074839,0.61496252,-0.0180229],
                       [0.56401211,0.5559324,-0.0201079],
                       [0.58505827,0.49206376,-0.0205996],
                       [0.58729452,0.4424372,-0.0172431],
                       [0.54393399,0.48367754,-0.0125163],
                       [0.56421655,0.43195719,-0.0347082],
                       [0.55273074,0.48224673,-0.0407095],
                       [0.5368638,0.51478773,-0.0394332],
                       [0.51207584,0.47177467,-0.0110409],
                       [0.53018183,0.42672583,-0.0350037],
                       [0.51708877,0.49388084,-0.0374278],
                       [0.50379848,0.52014875,-0.0327787],
                       [0.47837114,0.46822676,-0.0131793],
                       [0.49674463,0.42416936,-0.0407545],
                       [0.48920161,0.49561775,-0.0322177],
                       [0.47947341,0.524131,-0.0189786],
                       [0.44154605,0.46991152,-0.0161757],
                       [0.46195579,0.44096142,-0.0353138],
                       [0.46103215,0.48942119,-0.0283114],
                       [0.45384455,0.51065391,-0.0180876]])
test_input_G = np.array([[0.68178546,0.58866483,-3.28E-07],
                         [0.63810694,0.44520846,-0.0032826],
                         [0.52477872,0.34163949,-0.025947],
                         [0.39568439,0.33202997,-0.0445021],
                         [0.3074989,0.34100389,-0.0662366],
                         [0.5216471,0.25357896,-0.0759828],
                         [0.35606918,0.25633937,-0.1132128],
                         [0.25840679,0.28051192,-0.13274],
                         [0.18555748,0.30592597,-0.1460699],
                         [0.50703108,0.36195883,-0.093323],
                         [0.3409605,0.40303636,-0.1205799],
                         [0.37724286,0.44504318,-0.1105219],
                         [0.43035132,0.44559401,-0.1065551],
                         [0.49266294,0.48180193,-0.1076859],
                         [0.3516233,0.50811613,-0.129116],
                         [0.39606425,0.53829163,-0.0984534],
                         [0.44439453,0.53631455,-0.0806619],
                         [0.48209161,0.60250068,-0.1214426],
                         [0.37648517,0.60495889,-0.135681],
                         [0.41688856,0.62394607,-0.1110043],
                         [0.45567146,0.62109721,-0.0934447]])

"""Test letter A"""
results = coreml_model.predict({'input_1': test_input_A})
resultA = class_labels[np.argmax(results['Identity'][0])]
if (resultA == 'A'):
    print(f"{OKGREEN}", '[INFO] Result of Test A: ',
          class_labels[np.argmax(results['Identity'][0])], f"{ENDC}")
else:
    print(f"{WARNING}", '[WARNING] Result of Test A: ',
          class_labels[np.argmax(results['Identity'][0])], f"{ENDC}")

"""Test letter G"""
results = coreml_model.predict({'input_1': test_input_G})
resultG = class_labels[np.argmax(results['Identity'][0])]
if (resultG == 'G'):
    print(f"{OKGREEN}", '[INFO] Result of Test G: ',
          class_labels[np.argmax(results['Identity'][0])], f"{ENDC}")
else:
    print(f"{WARNING}", '[WARNING] Result of Test G: ',
          class_labels[np.argmax(results['Identity'][0])], f"{ENDC}")


"""#########################################################################"""
"""Save the model to disk at same directory""
"""#########################################################################"""
print('************************Save coreml model**************************')
output = pathModel.rsplit('.',1)[0] + ".mlmodel"
try:
    coreml_model.save(output)
except:
    print(f"{FAIL}[ERROR] Coreml model could not be saved!{ENDC}")
    exit() #End Programm execution
else:
    print(f"{OKGREEN}", '[INFO] Coreml model saved successfully as ',
          output, '!', f"{ENDC}")

print('*************************End of Script*****************************')
print('')