import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd

from SequenceProcessing import SequenceProcessor

from PIL import ImageTk, Image

PADY = 5
PADX = 5


class SequenceProcessingGUI(tk.Frame):
    def load_video(self):
        file_path = tk.filedialog.askopenfilename()

        if not self.sequence_processor.load_video(file_path)[0]:
            tk.messagebox.showerror(title='Loading Video', message='Import of Video not successful!')
        else:
            self.varImDir.set(file_path)
            self.imDir.config(state='disabled')
            self.set_scales(self.sequence_processor.get_video_info())

    def select_output_dir(self):
        output_dir = tk.filedialog.askdirectory()

        if not self.sequence_processor.set_output_path(output_dir):
            tk.messagebox.showerror(title='Loading Video', message='Empty path selected!')
        else:
            self.varSaveDir.set(output_dir)
            self.saveDir.config(state='disabled')

    def show_frames(self, fr1, fr2):
        self.im1 = ImageTk.PhotoImage(Image.fromarray(fr1).resize((self.view_width, self.view_height)))
        self.im2 = ImageTk.PhotoImage(Image.fromarray(fr2).resize((self.view_width, self.view_height)))

        self.frame1.configure(image=self.im1)
        self.frame2.configure(image=self.im2)

    def handle_frames(self, s):
        ret, frame1, frame2 = self.sequence_processor.set_sequence_limits(self.get_sequence_limits())
        if ret:
            self.show_frames(frame1, frame2)

    def handle_checkbox_buttons(self):
        if self.var_sel_AugmentData.get() and self.var_sel_exportSkeletonVector.get():
            self.sel_amountAugmentations.config(state='normal')
        else:
            self.sel_amountAugmentations.config(state='disabled')

        if self.var_sel_exportSkeletonVector.get():
            self.sel_amountSequences.config(state='normal')
        else:
            self.sel_amountSequences.config(state='disabled')

        if self.var_sel_AugmentData.get() and self.var_sel_exportSkeletonVector.get():
            self.sel_amountAugmentations.config(state='normal')
        else:
            self.sel_amountAugmentations.config(state='disabled')

        if self.var_sel_exportSkeletonVector.get() or self.var_sel_exportVideo.get():
            self.saveButton.config(state='normal')
        else:
            self.saveButton.config(state='disabled')

        vs, seq, aug = self.get_export_options()
        self.sequence_processor.set_export_options(vs, seq, aug)

    def get_sequence_limits(self):
        out = (float(self.scale1.get()), float(self.scale2.get()))
        return out

    def get_export_options(self):
        video_skeleton = (bool(self.var_sel_exportVideo.get()), bool(self.var_sel_exportSkeletonVector.get()))
        sequence = (25, int(self.sel_amountSequences.get()))
        augmentation = (bool(self.var_sel_AugmentData.get()), int(self.sel_amountAugmentations.get()))
        return video_skeleton, sequence, augmentation

    def set_scales(self, video_info):
        fps = video_info[0]
        length = video_info[1]
        self.scale1.config(resolution=float(1/fps))
        self.scale1.config(to=float(length))
        self.scale1.set(0)

        self.scale2.config(resolution=float(1/fps))
        self.scale2.config(to=float(length))
        self.scale2.set(length)

    def process(self):
        vs, seq, aug = self.get_export_options()
        self.sequence_processor.set_export_options(vs, seq, aug)

        if not self.sequence_processor.export_data():
            tk.messagebox.showerror(title='Loading Video', message='Export of Video not successful!')

    def __init__(self, parent):

        tk.Frame.__init__(self, parent)

        self.sequence_processor = SequenceProcessor()
        self.view_height = 270
        self.view_width = 480

        self.varScale1 = tk.DoubleVar()
        self.varScale2 = tk.DoubleVar()
        self.varImDir = tk.StringVar()
        self.varSaveDir = tk.StringVar()
        self.classLabel = tk.StringVar()

        self.varRotation = tk.DoubleVar()
        self.varRotation.set(45)
        self.varTranslation = tk.DoubleVar()
        self.varTranslation.set(50)


        self.var_sel_exportVideo = tk.IntVar()
        self.var_sel_exportSkeletonVector = tk.IntVar()
        self.var_sel_AugmentData = tk.IntVar()

        # File System Objects (Save- and Load-Directory)
        self.openDirButton = tk.Button(parent, text='Open Video File', command=self.load_video)
        self.openDirButton.grid(column=1, row=0, columnspan=1, sticky='we', pady=PADY, padx=PADX)
        self.imDir = tk.Entry(parent, textvariable=self.varImDir)
        self.imDir.grid(column=0, row=0, columnspan=1, sticky='we', pady=PADY, padx=PADX)

        self.saveDirButton = tk.Button(parent, text='Select Output Directory', command=self.select_output_dir)
        self.saveDirButton.grid(column=1, row=1, columnspan=1, sticky='we', pady=PADY, padx=PADX)
        self.saveDir = tk.Entry(parent, textvariable=self.varSaveDir)
        self.saveDir.grid(column=0, row=1, columnspan=1, sticky='we', pady=PADY, padx=PADX)

        # Timeline (Selection of start and end )
        self.im1 = ImageTk.PhotoImage(Image.new('RGB', (self.view_width, self.view_height), color=(255, 255, 255)))
        self.frame1 = tk.Label(parent)
        self.frame1.grid(column=0, row=2, columnspan=1, sticky='we', pady=PADY, padx=PADX)

        self.im2 = ImageTk.PhotoImage(Image.new('RGB', (self.view_width, self.view_height), color=(255, 255, 255)))
        self.frame2 = tk.Label(parent)
        self.frame2.grid(column=1, row=2, columnspan=1, sticky='we', pady=PADY, padx=PADX)

        self.scale1 = tk.Scale(parent, orient='horizontal', length=600, from_=float(0), to=float(100),
                               variable=self.varScale1, command=self.handle_frames)
        self.scale1.grid(column=0, row=3, columnspan=2, sticky='we', pady=PADY, padx=PADX)

        self.scale2 = tk.Scale(parent, orient='horizontal', length=600, from_=float(0), to=float(100),
                               variable=self.varScale2, command=self.handle_frames)
        self.scale2.grid(column=0, row=4, columnspan=2, sticky='we', pady=PADY, padx=PADX)

        self.sel_exportVideo = tk.Checkbutton(parent, text='Video exportieren',
                                              variable=self.var_sel_exportVideo,
                                              command=self.handle_checkbox_buttons)
        self.sel_exportVideo.grid(column=0, row=5, columnspan=1, sticky='we', pady=PADY, padx=PADX)

        self.sel_exportKeyPointSequence = tk.Checkbutton(parent, text='Keypoint-Sequenz exportieren',
                                                         variable=self.var_sel_exportSkeletonVector,
                                                         command=self.handle_checkbox_buttons)

        self.sel_exportKeyPointSequence.grid(column=1, row=5, columnspan=1, sticky='we', pady=PADY, padx=PADX)

        """
        self.label_sequenceLength = tk.Label(parent, text='Sequenzlänge:')
        self.label_sequenceLength.grid(column=0, row=6, columnspan=1, sticky='we', pady=PADY, padx=PADX)
        self.sel_sequenceLength = tk.Spinbox(parent, from_=2, to=50)
        self.sel_sequenceLength.grid(column=1, row=6, columnspan=1, sticky='we', pady=PADY, padx=PADX)
        """

        self.label_amountSequences = tk.Label(parent, text='Anzahl Sequenzen:')
        self.label_amountSequences.grid(column=0, row=7, columnspan=1, sticky='we', pady=PADY, padx=PADX)
        self.sel_amountSequences = tk.Spinbox(parent, from_=1, to=1000)
        self.sel_amountSequences.grid(column=1, row=7, columnspan=1, sticky='we', pady=PADY, padx=PADX)

        self.sel_augmentData = tk.Checkbutton(parent, text='Keypoint-Sequenz augmentieren',
                                              variable=self.var_sel_AugmentData,
                                              command=self.handle_checkbox_buttons)

        self.sel_augmentData.grid(column=0, row=8, columnspan=2, sticky='we', pady=PADY, padx=PADX)

        self.label_amountAugmentations = tk.Label(parent, text='Anzahl Augmentierungen:')
        self.label_amountAugmentations.grid(column=0, row=9, columnspan=1, sticky='we', pady=PADY, padx=PADX)
        self.sel_amountAugmentations = tk.Spinbox(parent, from_=1, to=1000)
        self.sel_amountAugmentations.grid(column=1, row=9, columnspan=1, sticky='we', pady=PADY, padx=PADX)
        """
        self.label_Rotation = tk.Label(parent, text='Winkelaugmentierung [°]')
        self.label_Rotation.grid(column=0, row=10, columnspan=1, sticky='we', pady=PADY, padx=PADX)
        self.scaleRotation = tk.Scale(parent, orient='horizontal', from_=0, to=45,
                                      variable=self.varRotation, command=self.handle_frames)
        self.scaleRotation.grid(column=1, row=10, columnspan=1, sticky='we', pady=PADY, padx=PADX)

        self.label_Translation = tk.Label(parent, text='Translationsaugmentierung [%]')
        self.label_Translation.grid(column=0, row=11, columnspan=1, sticky='we', pady=PADY, padx=PADX)
        self.scaleTranslation = tk.Scale(parent, orient='horizontal', from_=0, to=100,
                                         variable=self.varTranslation, command=self.handle_frames)
        self.scaleTranslation.grid(column=1, row=11, columnspan=1, sticky='we', pady=PADY, padx=PADX)
        """

        self.saveButton = tk.Button(parent, text='Save Sequence', command=self.process)
        self.saveButton.grid(column=0, row=12, columnspan=2, sticky='we', pady=PADY, padx=PADX)



        self.handle_checkbox_buttons()
