"""*******************************************************************************************
**** Testdata skeleton and augmentation Program
****
**** Takes a source folder and edits the images to save them in a destination folder. A skeleton
**** is drawn in with Mediapipe. It is also possible to apply various augemntations.
"""

"""####################################################################################################################
#######################################################################################################################
                                                    Imports
#######################################################################################################################
####################################################################################################################"""
import os as os
from dataclasses import dataclass
import dataclasses
import tkinter as tk
from tkinter import filedialog as fd
from ctypes import windll
windll.shcore.SetProcessDpiAwareness(1)
import mediapipe as mp
import tensorflow as tf
import cv2
import numpy as np
from google.protobuf.json_format import MessageToDict
import matplotlib.pyplot as plt
import sys
import random


"""####################################################################################################################
#######################################################################################################################
                                        Global Data Structures/Variables
#######################################################################################################################
####################################################################################################################"""
# Global Datastructure
@dataclass
class option:
    state: bool
    stateSkeleton: bool
    num: int
    attribute: int
    attribute2: float
    attribute3: float
    saveIfFound: bool
    destPathList: list[str] = dataclasses.field(default_factory=list)
@dataclass
class glbDataStruct:
    # Global Data
    src: str
    dest: str
    orig: option
    opt2: option
    opt3: option
    opt4: option
    doFlipOnAll: bool
    letterList: list[str] = dataclasses.field(default_factory=list)
    def __post_init__(self):
        self.orig = option(**self.orig)
        self.opt2 = option(**self.opt2)
        self.opt3 = option(**self.opt3)
        self.opt4 = option(**self.opt4)
data = {
    'src': '',
    'dest': '',
    'orig':{
        'state': False,
        'stateSkeleton': False,
        'num': 0,
        'attribute': 0,
        'attribute2': 0.0,
        'attribute3': 0.0,
        'saveIfFound': False,
        'destPathList': ['/Original/Original', '/Original/Original with Skeleton']
    },
    'opt2':{
        'state': False,
        'stateSkeleton': False,
        'num': 0,
        'attribute': 0,
        'attribute2': 0.0,
        'attribute3': 0.0,
        'saveIfFound': False,
        'destPathList': ['/Random Rotation/Random Rotation',
                         '/Random Rotation/Random Rotation with Skeleton']
    },
    'opt3': {
        'state': False,
        'stateSkeleton': False,
        'num': 0,
        'attribute': 0,
        'attribute2': 0.0,
        'attribute3': 0.0,
        'saveIfFound': False,
        'destPathList': ['/Random Shear/Random Shear',
                         '/Random Shear/Random Shear with Skeleton']
    },
    'opt4': {
        'state': False,
        'stateSkeleton': False,
        'num': 0,
        'attribute': 0,
        'attribute2': 0.0,
        'attribute3': 0.0,
        'saveIfFound': False,
        'destPathList': ['/Random Resize/Random Resize',
                         '/Random Resize/Random Resize with Skeleton']
    },
    'doFlipOnAll': False,
    'letterList': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
                   'V', 'W', 'X', 'Y']
}
# Create global variable
glbData = glbDataStruct(**data)
"""####################################################################################################################
#######################################################################################################################
                                                        GUI
#######################################################################################################################
####################################################################################################################"""
# Parameters
scaleFactor = 1.0
desWidth = int(scaleFactor*180)  # Desired width of main window in mm
desHeight = int(scaleFactor*130) # Desired height of main window in mm
maxValueEntry = 100  # Max Value of a GUI entry
confidenceMp = 0.75 # Minimum confidence for Media Pipe
maxSkeletonTrys = 10 # Maximum trys to find a Skeleton in an augmented picture

# Variables
srcPath = ''
destPath = ''

# Functions
def fileDialogSrc():
    global srcPath
    srcPath = fd.askdirectory(
        title='Choose source folder',
        initialdir='/',
    )

def fileDialogDest():
    global destPath
    destPath = fd.askdirectory(
        title='Choose destination folder',
        initialdir='/',
    )

def switchState(var, label, disableList, disChkBoxList):
    if var.get() == 1:
        label.config(text='ON')
        if disableList != 'None':
            for participant in disableList:
                participant.config(state='normal')
        if disChkBoxList != 'None':
            for participant in disChkBoxList:
                participant.config(state='normal')

    elif var.get() == 0:
        label.config(text='OFF')
        if disableList != 'None':
            for participant in disableList:
                participant.delete(0, "end")
                participant.insert(0, 0)
                participant.config(state='disabled')
        if disChkBoxList != 'None':
            for participant in disChkBoxList:
                participant.config(state='disabled')
    else:
        label.config(text='error!')

def validateInput(win, btnStateList, numList, pathList, opt4FloatList, opt4FloatState):
    global dataValid
    dataValid = True
    # Validate the values of the entrys if the corresponding button is active
    for btnstate, num in zip(btnStateList, numList):
        if(dataValid == True):
            if(btnstate.get() == 1):
                if(num.get() <= 0):
                    tk.messagebox.showerror('Number not valid', 'Error: If option is active, choose a value between 1 and '
                                            + str(maxValueEntry) + '!')
                    dataValid = False
                elif(num.get() > maxValueEntry):
                    tk.messagebox.showerror('Number not valid', 'Error: If option is active, choose a value between 1 and '
                                            + str(maxValueEntry) + '!')
                    dataValid = False
    # Validate the paths
    for path in pathList:
        if (dataValid == True):
            if(path=='/' or path==''):
                tk.messagebox.showerror('Path not valid', 'One or both of the paths are not valid!')
                dataValid = False
    # Check random Shift values
    if dataValid is True:
        if validateRandShift(opt4FloatList, opt4FloatState) is False:
            dataValid = False
    # Cancel main loop if data is valid
    if dataValid is True:
        win.destroy()

def validateRandShift(attList, floatState):
    dataValid_ = True
    if floatState.get() == 1:
        for att in attList:
            if att.get()<0.01 or att.get()>0.95:
                if dataValid_ == True:
                    tk.messagebox.showerror('Number not valid',
                                            'Random Shift: Max height/width has to be a value between 0.01 and 0.95')
                dataValid_ = False
    return dataValid_





# Create the main window
mainWin = tk.Tk()

# Scale window to a size in mm
screen_widthPix = mainWin.winfo_screenwidth()
screen_heightPix = mainWin.winfo_screenheight()
screen_width = mainWin.winfo_screenmmwidth()
screen_height = mainWin.winfo_screenmmheight()
widthFact = screen_widthPix/screen_width
heightFact = screen_heightPix/screen_height
widthStr = str(int(desWidth*widthFact))
heightStr = str(int(desHeight*heightFact))
geomStr = widthStr + "x" + heightStr

# Some options of the main window
mainWin.title("Options and Filepath menu")
mainWin.geometry(geomStr)
mainWin.resizable(False, False)

# Create the labels and place it
srcPathLabel = tk.Label(
    text="Choose the source files:",
    width=25,
    height=1,
    justify="left",
    anchor="w"
    )
destPathLabel = tk.Label(
    text="Choose the destination folder:",
    width=25,
    height=1,
    justify="left",
    anchor="w"
    )

optLabel = tk.Label(
    text="Options:",
    width=10,
    height=1,
    justify="left",
    anchor="w"
)
origLabel = tk.Label(
    text="-Copy orig. Images:",
    width=30,
    height=1,
    justify="left",
    anchor="w"
)
opt2Label = tk.Label(
    text="-Random Rotation:",
    width=15,
    height=1,
    justify="left",
    anchor="w"
)
origSkeletonLabel = tk.Label(
    text="-Do Skeleton:",
    width=15,
    height=1,
    justify="left",
    anchor="w"
)
opt2SkeletonLabel = tk.Label(
    text="-Do Skeleton:",
    width=15,
    height=1,
    justify="left",
    anchor="w"
)
opt3Label = tk.Label(
    text="-Random Shear:",
    width=15,
    height=1,
    justify="left",
    anchor="w"
)
opt3SkeletonLabel = tk.Label(
    text="-Do Skeleton:",
    width=15,
    height=1,
    justify="left",
    anchor="w"
)
opt4Label = tk.Label(
    text="-Random Resize:",
    width=15,
    height=1,
    justify="left",
    anchor="w"
)
opt4SkeletonLabel = tk.Label(
    text="-Do Skeleton:",
    width=15,
    height=1,
    justify="left",
    anchor="w"
)
origStateLabel = tk.Label(
    text ="OFF",
    width=15,
    height=1
)
origSkeletonStateLabel = tk.Label(
    text ="OFF",
    width=15,
    height=1
)
opt2StateLabel = tk.Label(
    text ="OFF",
    width=15,
    height=1
)
opt2SkeletonStateLabel = tk.Label(
    text ="OFF",
    width=15,
    height=1
)
opt2SIFStateLabel = tk.Label(
    text ="OFF",
    width=15,
    height=1
)
opt3StateLabel = tk.Label(
    text ="OFF",
    width=15,
    height=1
)
opt3SkeletonStateLabel = tk.Label(
    text ="OFF",
    width=15,
    height=1
)
opt4StateLabel = tk.Label(
    text ="OFF",
    width=15,
    height=1
)
opt4SkeletonStateLabel = tk.Label(
    text ="OFF",
    width=15,
    height=1
)
opt2NumLabel = tk.Label(
    text="Number:",
    width=10,
    height=1,
    justify="left",
    anchor="w"
)
opt3NumLabel = tk.Label(
    text="Number:",
    width=10,
    height=1,
    justify="left",
    anchor="w"
)
opt4NumLabel = tk.Label(
    text="Number:",
    width=10,
    height=1,
    justify="left",
    anchor="w"
)
opt2AttLabel = tk.Label(
    text="Max degrees:",
    width=12,
    height=1,
    justify="left",
    anchor="w"
)
opt2SIFLabel = tk.Label(
    text="Save only if Skeleton Found:",
    width=24,
    height=1,
    justify="left",
    anchor="w"
)
opt3AttLabel = tk.Label(
    text="Max intensity:",
    width=12,
    height=1,
    justify="left",
    anchor="w"
)
opt3SIFLabel = tk.Label(
    text="Save only if Skeleton Found:",
    width=24,
    height=1,
    justify="left",
    anchor="w"
)
opt3SIFStateLabel = tk.Label(
    text ="OFF",
    width=15,
    height=1
)
opt4AttLabel = tk.Label(
    text="Height fact:",
    width=12,
    height=1,
    justify="left",
    anchor="w"
)
opt4AttLabel2 = tk.Label(
    text="Width fact:",
    width=12,
    height=1,
    justify="left",
    anchor="w"
)
opt4SIFLabel = tk.Label(
    text="Save only if Skeleton Found:",
    width=24,
    height=1,
    justify="left",
    anchor="w"
)
opt4SIFStateLabel = tk.Label(
    text ="OFF",
    width=15,
    height=1
)
optGlbLabel = tk.Label(
    text="Global options:",
    width=24,
    height=1,
    justify="left",
    anchor="w"
)
doFlipLabel = tk.Label(
    text="Augment also flipped Pictures (does flip):",
    width=34,
    height=1,
    justify="left",
    anchor="w"
)
doFlipStateLabel = tk.Label(
    text ="OFF",
    width=15,
    height=1
)

srcPathLabel.place(x=25,y=40)
destPathLabel.place(x=25,y=140)
optLabel.place(x=25,y=220)
origLabel.place(x=140,y=220)
opt2Label.place(x=140,y=340)
opt3Label.place(x=140,y=460)
opt4Label.place(x=140,y=580)
origStateLabel.place(x=600,y=220)
origSkeletonStateLabel.place(x=600,y=270)
opt2StateLabel.place(x=600,y=340)
opt2SkeletonStateLabel.place(x=600,y=390)
opt2SIFStateLabel.place(x=1130,y=390)
opt3StateLabel.place(x=600,y=460)
opt3SkeletonStateLabel.place(x=600,y=510)
opt3SIFStateLabel.place(x=1130,y=510)
opt4StateLabel.place(x=600,y=580)
opt4SkeletonStateLabel.place(x=600,y=630)
opt4SIFStateLabel.place(x=1130,y=680)
opt2NumLabel.place(x=780,y=340)
opt3NumLabel.place(x=780,y=460)
opt4NumLabel.place(x=780,y=580)
origSkeletonLabel.place(x=400,y=270)
opt2SkeletonLabel.place(x=400,y=390)
opt3SkeletonLabel.place(x=400,y=510)
opt4SkeletonLabel.place(x=400,y=630)
opt2AttLabel.place(x=1040,y=340)
opt2SIFLabel.place(x=780,y=390)
opt3AttLabel.place(x=1040,y=460)
opt3SIFLabel.place(x=780,y=510)
opt4AttLabel.place(x=1040,y=580)
opt4AttLabel2.place(x=1040,y=630)
opt4SIFLabel.place(x=780,y=680)
optGlbLabel.place(x=25,y=750)
doFlipLabel.place(x=140,y=800)
doFlipStateLabel.place(x=600,y=800)

# Create entrys and place it
opt2Num = tk.IntVar()
opt2AttNum = tk.IntVar()
opt3Num = tk.IntVar()
opt3AttNum = tk.IntVar()
opt4Num = tk.IntVar()
opt4AttNum1 = tk.DoubleVar()
opt4AttNum2 = tk.DoubleVar()

opt2Entry = tk.Entry(mainWin,
                     width=7,
                     justify='right',
                     state='disabled',
                     textvariable=opt2Num)
opt2AttEntry = tk.Entry(mainWin,
                     width=7,
                     justify='right',
                     state='disabled',
                     textvariable=opt2AttNum)
opt3Entry = tk.Entry(mainWin,
                     width=7,
                     justify='right',
                     state='disabled',
                     textvariable=opt3Num)
opt3AttEntry = tk.Entry(mainWin,
                     width=7,
                     justify='right',
                     state='disabled',
                     textvariable=opt3AttNum)
opt4Entry = tk.Entry(mainWin,
                     width=7,
                     justify='right',
                     state='disabled',
                     textvariable=opt4Num)
opt4AttEntry1 = tk.Entry(mainWin,
                     width=7,
                     justify='right',
                     state='disabled',
                     textvariable=opt4AttNum1)
opt4AttEntry2 = tk.Entry(mainWin,
                     width=7,
                     justify='right',
                     state='disabled',
                     textvariable=opt4AttNum2)

opt2Entry.place(x=900, y=342)
opt2AttEntry.place(x=1210, y=342)
opt3Entry.place(x=900, y=462)
opt3AttEntry.place(x=1210, y=462)
opt4Entry.place(x=900, y=582)
opt4AttEntry1.place(x=1210, y=582)
opt4AttEntry2.place(x=1210, y=632)


# Create Buttons and place it
srcPathBtn = tk.Button(
    mainWin,
    text="Open filedialog",
    command=lambda: fileDialogSrc(),
    width=20,
    height=2,
)
destPathBtn = tk.Button(
    mainWin,
    text="Open filedialog",
    command=lambda: fileDialogDest(),
    width=20,
    height=2,
)
switch_on = tk.PhotoImage(width=20, height=20)
switch_off = tk.PhotoImage(width=20, height=20)
switch_on.put(("green",), to=(0, 0, 18, 18))
switch_off.put(("red",), to=(0, 0, 19, 19))
origState = tk.IntVar()
origSkeletonState = tk.IntVar()
opt2State = tk.IntVar()
opt2SkeletonState = tk.IntVar()
opt2SIFState = tk.IntVar()
opt3State = tk.IntVar()
opt3SkeletonState = tk.IntVar()
opt3SIFState = tk.IntVar()
opt4State = tk.IntVar()
opt4SkeletonState = tk.IntVar()
opt4SIFState = tk.IntVar()
doFlipBtnState = tk.IntVar()

origBtn = tk.Checkbutton(mainWin,
                        image=switch_off,
                        selectimage=switch_on,
                        onvalue=1, offvalue=0,
                        variable=origState,
                        indicatoron=False,
                        command=lambda: switchState(origState, origStateLabel, 'None', [origSkeletonBtn]))
origSkeletonBtn = tk.Checkbutton(mainWin,
                        image=switch_off,
                        selectimage=switch_on,
                        onvalue=1, offvalue=0,
                        variable=origSkeletonState,
                        indicatoron=False,
                        state='disabled',
                        command=lambda: switchState(origSkeletonState, origSkeletonStateLabel, 'None', 'None'))
opt2Btn = tk.Checkbutton(mainWin,
                        image=switch_off,
                        selectimage=switch_on,
                        onvalue=1, offvalue=0,
                        variable=opt2State,
                        indicatoron=False,
                        command=lambda: switchState(opt2State, opt2StateLabel, [opt2Entry, opt2AttEntry],
                                                    [opt2SkeletonBtn, opt2SIFBtn]))
opt2SkeletonBtn = tk.Checkbutton(mainWin,
                        image=switch_off,
                        selectimage=switch_on,
                        onvalue=1, offvalue=0,
                        variable=opt2SkeletonState,
                        indicatoron=False,
                        state='disabled',
                        command=lambda: switchState(opt2SkeletonState, opt2SkeletonStateLabel, 'None', 'None'))
opt2SIFBtn = tk.Checkbutton(mainWin,
                        image=switch_off,
                        selectimage=switch_on,
                        onvalue=1, offvalue=0,
                        variable=opt2SIFState,
                        indicatoron=False,
                        state='disabled',
                        command=lambda: switchState(opt2SIFState, opt2SIFStateLabel, 'None', 'None'))
opt3Btn = tk.Checkbutton(mainWin,
                        image=switch_off,
                        selectimage=switch_on,
                        onvalue=1, offvalue=0,
                        variable=opt3State,
                        indicatoron=False,
                        command=lambda: switchState(opt3State, opt3StateLabel, [opt3Entry, opt3AttEntry],
                                                    [opt3SkeletonBtn, opt3SIFBtn]))
opt3SkeletonBtn = tk.Checkbutton(mainWin,
                        image=switch_off,
                        selectimage=switch_on,
                        onvalue=1, offvalue=0,
                        variable=opt3SkeletonState,
                        indicatoron=False,
                        state='disabled',
                        command=lambda: switchState(opt3SkeletonState, opt3SkeletonStateLabel, 'None', 'None'))
opt3SIFBtn = tk.Checkbutton(mainWin,
                        image=switch_off,
                        selectimage=switch_on,
                        onvalue=1, offvalue=0,
                        variable=opt3SIFState,
                        indicatoron=False,
                        state='disabled',
                        command=lambda: switchState(opt3SIFState, opt3SIFStateLabel, 'None', 'None'))
opt4Btn = tk.Checkbutton(mainWin,
                        image=switch_off,
                        selectimage=switch_on,
                        onvalue=1, offvalue=0,
                        variable=opt4State,
                        indicatoron=False,
                        command=lambda: switchState(opt4State, opt4StateLabel,[opt4Entry, opt4AttEntry1, opt4AttEntry2],
                                                    [opt4SkeletonBtn, opt4SIFBtn]))
opt4SkeletonBtn = tk.Checkbutton(mainWin,
                        image=switch_off,
                        selectimage=switch_on,
                        onvalue=1, offvalue=0,
                        variable=opt4SkeletonState,
                        indicatoron=False,
                        state='disabled',
                        command=lambda: switchState(opt4SkeletonState, opt4SkeletonStateLabel, 'None', 'None'))
opt4SIFBtn = tk.Checkbutton(mainWin,
                        image=switch_off,
                        selectimage=switch_on,
                        onvalue=1, offvalue=0,
                        variable=opt4SIFState,
                        indicatoron=False,
                        state='disabled',
                        command=lambda: switchState(opt4SIFState, opt4SIFStateLabel, 'None', 'None'))
doFlipBtn= tk.Checkbutton(mainWin,
                        image=switch_off,
                        selectimage=switch_on,
                        onvalue=1, offvalue=0,
                        variable=doFlipBtnState,
                        indicatoron=False,
                        command=lambda: switchState(doFlipBtnState, doFlipStateLabel,'None','None'))
goForItBtn = tk.Button(
    mainWin,
    text="Do augmentation/skeleton",
    width=25,
    height=2,
    command=lambda: validateInput(mainWin,
                                  [opt2State, opt3State,opt4State],
                                  [opt2Num, opt3Num, opt4Num],
                                  [srcPath, destPath],
                                  [opt4AttNum1, opt4AttNum2],
                                  opt4State)
)


srcPathBtn.place(x=600,y=12)
destPathBtn.place(x=600,y=112)
origBtn.place(x=600,y=226)
origSkeletonBtn.place(x=600,y=276)
opt2Btn.place(x=600,y=346)
opt2SkeletonBtn.place(x=600,y=396)
opt2SIFBtn.place(x=1130,y=396)
opt3Btn.place(x=600,y=466)
opt3SkeletonBtn.place(x=600,y=516)
opt3SIFBtn.place(x=1130,y=516)
opt4Btn.place(x=600,y=586)
opt4SkeletonBtn.place(x=600,y=636)
opt4SIFBtn.place(x=1130,y=686)
goForItBtn.place(x=25,y=870)
doFlipBtn.place(x=600,y=806)

# Wait for the user to end the options session
mainWin.mainloop()
if 'dataValid' not in locals():
    print('End programm because of no valid configuration.')
    exit()
elif dataValid == False:
    print('End programm because of no valid configuration.')
    exit()
"""####################################################################################################################
#######################################################################################################################
                                            Assign to global Data Struct
#######################################################################################################################
####################################################################################################################"""
# Assign data of GUI to global struct
glbData.src = srcPath
glbData.dest = destPath
glbData.orig.state = bool(origState.get())
glbData.opt2.state = bool(opt2State.get())
glbData.opt3.state = bool(opt3State.get())
glbData.opt4.state = bool(opt4State.get())
glbData.orig.stateSkeleton = bool(origSkeletonState.get())
glbData.opt2.stateSkeleton = bool(opt2SkeletonState.get())
glbData.opt3.stateSkeleton = bool(opt3SkeletonState.get())
glbData.opt4.stateSkeleton = bool(opt4SkeletonState.get())
glbData.opt2.num = int(opt2Num.get())
glbData.opt3.num = int(opt3Num.get())
glbData.opt4.num = int(opt4Num.get())
glbData.opt2.attribute = int(opt2AttNum.get())
glbData.opt3.attribute = int(opt3AttNum.get())
glbData.opt4.attribute2 = float(opt4AttNum1.get())
glbData.opt4.attribute3 = float(opt4AttNum2.get())
glbData.opt2.saveIfFound = bool(opt2SIFState.get())
glbData.opt3.saveIfFound = bool(opt3SIFState.get())
glbData.opt4.saveIfFound = bool(opt4SIFState.get())
glbData.doFlipOnAll = bool(doFlipBtnState)

# Print Data after validation
print("\n*******************************************Data from user:*************************************************")

print("Source Paths", glbData.src)
print("Destination Paths", glbData.dest)
print("Original:    ",
      "State: ",glbData.orig.state,
      "  Skeleton State: ",glbData.orig.stateSkeleton)
print("Option 2:    ",
      "State: ",glbData.opt2.state,
      "  Skeleton State: ",glbData.opt2.stateSkeleton,
      "  Number: ",glbData.opt2.num,
      "  Max. Degrees: ",glbData.opt2.attribute,
      "  Safe only if Skeleton found: ",glbData.opt2.saveIfFound)
print("Option 3:    ",
      "State: ",glbData.opt3.state,
      "  Skeleton State: ",glbData.opt3.stateSkeleton,
      "  Number: ",glbData.opt3.num,
      "  Max. Degrees: ",glbData.opt3.attribute,
      "  Safe only if Skeleton found: ",glbData.opt3.saveIfFound)
print("Option 4:    ",
      "State: ",glbData.opt4.state,
      "  Skeleton State: ",glbData.opt4.stateSkeleton,
      "  Number: ",glbData.opt4.num,
      "  Max. Height: ",glbData.opt4.attribute2,
      "  Max. Width: ",glbData.opt4.attribute3,
      "  Safe only if Skeleton found: ",glbData.opt3.saveIfFound)
print("Global Options: Do Flip:", glbData.doFlipOnAll)
print("***********************************************************************************************************\n")

"""####################################################################################################################
#######################################################################################################################
                                            Create Destination Directory
#######################################################################################################################
####################################################################################################################"""
# Create destination directory of original Skeleton option
if glbData.orig.state is True:
    for letter in glbData.letterList:
        if not os.path.exists(glbData.dest + glbData.orig.destPathList[0] + "/" + letter):
            os.makedirs(glbData.dest + glbData.orig.destPathList[0] + "/" + letter)
        if glbData.orig.stateSkeleton is True:
            if not os.path.exists(glbData.dest + glbData.orig.destPathList[1] + "/" + letter):
                os.makedirs(glbData.dest + glbData.orig.destPathList[1] + "/" + letter)

# Create directory of option 2
if glbData.opt2.state is True:
    for letter in glbData.letterList:
        if not os.path.exists(glbData.dest + glbData.opt2.destPathList[0] + "/" + letter):
            os.makedirs(glbData.dest + glbData.opt2.destPathList[0] + "/" + letter)
        if glbData.opt2.stateSkeleton is True:
            if not os.path.exists(glbData.dest + glbData.opt2.destPathList[1] + "/" + letter):
                os.makedirs(glbData.dest + glbData.opt2.destPathList[1] + "/" + letter)

# Create directory of option 3
if glbData.opt3.state is True:
    for letter in glbData.letterList:
        if not os.path.exists(glbData.dest + glbData.opt3.destPathList[0] + "/" + letter):
            os.makedirs(glbData.dest + glbData.opt3.destPathList[0] + "/" + letter)
        if glbData.opt3.stateSkeleton is True:
            if not os.path.exists(glbData.dest + glbData.opt3.destPathList[1] + "/" + letter):
                os.makedirs(glbData.dest + glbData.opt3.destPathList[1] + "/" + letter)

# Create directory of option 4
if glbData.opt4.state is True:
    for letter in glbData.letterList:
        if not os.path.exists(glbData.dest + glbData.opt4.destPathList[0] + "/" + letter):
            os.makedirs(glbData.dest + glbData.opt4.destPathList[0] + "/" + letter)
        if glbData.opt4.stateSkeleton is True:
            if not os.path.exists(glbData.dest + glbData.opt4.destPathList[1] + "/" + letter):
                os.makedirs(glbData.dest + glbData.opt4.destPathList[1] + "/" + letter)

"""####################################################################################################################
#######################################################################################################################
                                            Draw Skeleton with MediaPipe
#######################################################################################################################
####################################################################################################################"""
def printStatus(status):
    sys.stdout.write('\r')
    sys.stdout.flush()
    sys.stdout.write(status)
    sys.stdout.flush()

def clearStatus():
    sys.stdout.write('\r')
    sys.stdout.flush()

def textOnImg(img, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    position = (10, 10)
    fontScale = 0.2
    fontColor = (255, 255, 0)
    cv2.putText(img, text, position, font, fontScale, fontColor)

# For static images:
def drawSkeleton(image, destinationPath, confidence, labelImg=False, doSaveImg = True, onlySaveWithSkel = True):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=confidence) as hands:
        # Convert the BGR image to RGB before processing.
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # Print result in image
        if labelImg is True:
            for idx, hand_handedness in enumerate(results.multi_handedness):
                imgLabel = str(MessageToDict(hand_handedness))
        # Print handedness and draw hand landmarks on the image.
        #print('Handedness:', results.multi_handedness)
        if results.multi_hand_landmarks:
            image_height, image_width, _ = image.shape
            annotated_image = image.copy()
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
            if labelImg is True:
                textOnImg(annotated_image, imgLabel)
            if doSaveImg is True:
                cv2.imwrite(destinationPath, cv2.flip(annotated_image, 1))
            retval = True
        else:
            if doSaveImg is True and onlySaveWithSkel is False:
                cv2.imwrite(destinationPath, cv2.flip(image, 1))
            retval = False

        return retval

"""####################################################################################################################
#######################################################################################################################
                                            Original Images
#######################################################################################################################
####################################################################################################################"""
def processOrigImages(name, doFlip=False):
    if glbData.orig.state is True:
        # Copy original images to destination if desired
        totalCount = 0
        count = 0
        srcDir = os.listdir(glbData.src)
        for dir in srcDir:
            subDir = os.listdir(glbData.src + "/" + dir)
            printStatus(str('Copy '+ name + ' section ' + dir))
            for sDir in subDir:
                totalCount = totalCount + 1
                try:
                    image_file = (glbData.src + "/" + dir + "/" + sDir)
                    if doFlip is False:
                        image = cv2.flip(cv2.imread(image_file), 1)
                        tempDestPath = (glbData.dest + glbData.orig.destPathList[0] + "/" + dir + "/" + sDir)
                    else:
                        image = cv2.imread(image_file)
                        splitString = sDir.split('.')
                        tempDestPath = (glbData.dest + glbData.orig.destPathList[0] + "/" + dir + "/" + splitString[0]
                                        + "_flip" + "." + splitString[1])
                    cv2.imwrite(tempDestPath, cv2.flip(image, 1))
                finally:
                    count = count + 1
        clearStatus()
        print('Copy '+ name + ': ', count, ' of ', totalCount, ' pictures were transfered!')

        # Draw Skeleton into original images if desired
        if glbData.orig.stateSkeleton is True:
            totalCount = 0
            count = 0
            for dir in srcDir:
                subDir = os.listdir(glbData.src + "/" + dir)
                printStatus(str('Do Skeleton on '+ name + ' section ' + dir))
                for sDir in subDir:
                    totalCount = totalCount + 1
                    image_file = (glbData.src + "/" + dir + "/" + sDir)
                    foundSkeleton = False
                    splitString = sDir.split('.')
                    if doFlip is False:
                        image = cv2.flip(cv2.imread(image_file), 1)
                        tempDestPath = (glbData.dest + glbData.orig.destPathList[1] + "/" + dir + "/" + splitString[0]
                                        + "_skel" + "." + splitString[1])
                        foundSkeleton = drawSkeleton(image, tempDestPath, confidenceMp, doSaveImg=True)
                    else:
                        image = cv2.imread(image_file)
                        tempDestPath = (glbData.dest + glbData.orig.destPathList[1] + "/" + dir + "/" + splitString[0]
                                        + "skel_flip" + "." + splitString[1])
                        foundSkeleton = drawSkeleton(image, tempDestPath, confidenceMp, doSaveImg=True)
                    if foundSkeleton is True:
                        count = count + 1
        clearStatus()
        print('Do Skeleton in '+ name + ': ', count, ' of ', totalCount, ' Skeletons were found and saved!')


print('Start processing:\n')
processOrigImages("original images")
if glbData.doFlipOnAll is True:
    processOrigImages("original images flip", doFlip=True)

"""####################################################################################################################
#######################################################################################################################
                                            Global augmentation processing function
#######################################################################################################################
####################################################################################################################"""
def doAugmentation(option, augmentationMethod, name, doFlip=False):
    if option.state is True:
        srcDir = os.listdir(glbData.src)
        for dir in srcDir:
            subDir = os.listdir(glbData.src + "/" + dir)
            printStatus(str(name + ' of section ' + dir))
            totalCount = 0
            count = 0
            for idx, sDir in enumerate(subDir):
                image_file = (glbData.src + "/" + dir + "/" + sDir)
                # Read an image, flip it around y-axis for correct handedness output (see above).
                if doFlip is False:
                    imageRGB = cv2.flip(cv2.imread(image_file), 1)
                else:
                    imageRGB = cv2.imread(image_file)
                splitString = sDir.split('.')
                for i in range(option.num):
                    # Set destination paths
                    totalCount = totalCount + 1
                    if doFlip is False:
                        destPath = (glbData.dest + option.destPathList[0] + "/" + dir + "/"
                                    + splitString[0] + "_aug_" + str(i) + "." + splitString[1])
                        destPathSkel = (glbData.dest + option.destPathList[1] + "/" + dir + "/"
                                     + splitString[0] + "_aug_skel" + str(i) + "." + splitString[1])
                    else:
                        destPath = (glbData.dest + option.destPathList[0] + "/" + dir + "/"
                                    + splitString[0] + "_aug_flip_" + str(i) + "." + splitString[1])
                        destPathSkel = (glbData.dest + option.destPathList[1] + "/" + dir + "/"
                                     + splitString[0] + "_aug_flip_skel" + str(i) + "." + splitString[1])

                    # Just augment and save the images
                    if option.stateSkeleton is False and option.saveIfFound is False:
                        augImage = augmentationMethod(imageRGB)
                        cv2.imwrite(destPath, cv2.flip(augImage, 1))
                        count = count + 1
                    # Save the augmented image only if a skeleton was found in it
                    elif option.stateSkeleton is False and option.saveIfFound is True:
                        skeletonFound = False
                        tempErrCount = 0
                        while skeletonFound is False and tempErrCount <= maxSkeletonTrys:
                            augImage = augmentationMethod(imageRGB)
                            skeletonFound = drawSkeleton(augImage, destPath, confidenceMp, doSaveImg=False)
                            tempErrCount = tempErrCount + 1
                        if skeletonFound is True:
                            cv2.imwrite(destPath, cv2.flip(augImage, 1))
                            count = count + 1
                    # Save the augmented image with and without Skeleton
                    elif option.stateSkeleton is True and option.saveIfFound is False:
                        augImage = augmentationMethod(imageRGB)
                        cv2.imwrite(destPath, cv2.flip(augImage, 1))
                        drawSkeleton(augImage, destPathSkel, confidenceMp, doSaveImg=True, onlySaveWithSkel=False)
                        count = count + 1
                    # Save the augmented image with and without skeleton if the skeleton was found in it
                    else:
                        skeletonFound = False
                        tempErrCount = 0
                        while skeletonFound is False and tempErrCount <= maxSkeletonTrys:
                            augImage = augmentationMethod(imageRGB)
                            skeletonFound = drawSkeleton(augImage, destPathSkel, confidenceMp, doSaveImg=True)
                            tempErrCount = tempErrCount + 1
                        if skeletonFound is True:
                            cv2.imwrite(destPath, cv2.flip(augImage, 1))
                            count = count + 1
        clearStatus()
        print(str('Do ' + name + ' original images: '), count, ' of ', totalCount, ' were successfull augmented!')

"""####################################################################################################################
#######################################################################################################################
                                            Random Rotation Augmentation
#######################################################################################################################
####################################################################################################################"""
def randRot(image):
    img = tf.keras.preprocessing.image.random_rotation(
        image,
        glbData.opt2.attribute,
        row_axis=0,
        col_axis=1,
        channel_axis=2,
        fill_mode='nearest',
        cval=0.0,
        interpolation_order=1
    )
    return img


# Do augmentation
doAugmentation(glbData.opt2, randRot, 'Random Rotation')
if glbData.doFlipOnAll is True:
    doAugmentation(glbData.opt2, randRot, 'Random Rotation Flip', doFlip = True)

"""####################################################################################################################
#######################################################################################################################
                                            Random Shear Augmentation
#######################################################################################################################
####################################################################################################################"""
def randShear(image):
    img = tf.keras.preprocessing.image.random_shear(
        image,
        glbData.opt3.attribute,
        row_axis=0,
        col_axis=1,
        channel_axis=2,
        fill_mode='nearest',
        cval=0.0,
        interpolation_order=1
    )
    return img


# Do augmentation
doAugmentation(glbData.opt3, randShear, 'Random Shear')
if glbData.doFlipOnAll is True:
    doAugmentation(glbData.opt3, randShear, 'Random Shear Flip', doFlip=True)

"""####################################################################################################################
#######################################################################################################################
                                            Random Resize Augmentation
#######################################################################################################################
####################################################################################################################"""
def randResize(image):
    height = image.shape[0]
    width = image.shape[1]
    scaleWidth = int(random.random()*glbData.opt4.attribute3*width + width)
    scaleHeight = int(random.random()*glbData.opt4.attribute2*height + height)
    dimensions = (scaleWidth, scaleHeight)
    img = cv2.resize(image, dimensions, interpolation=cv2.INTER_LINEAR)
    return img


# Do augmentation
doAugmentation(glbData.opt4, randResize, 'Random Resize')
if glbData.doFlipOnAll is True:
    doAugmentation(glbData.opt4, randResize, 'Random Resize Flip',doFlip=True)

