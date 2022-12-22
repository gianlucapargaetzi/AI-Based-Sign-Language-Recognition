from SequenceProcessingGUI import SequenceProcessingGUI
import tkinter as tk
import tensorflow as tf

if __name__ == "__main__":
    tf.get_logger().setLevel('ERROR')
    root = tk.Tk()
    gui1 = SequenceProcessingGUI(root)
    gui1.place(x=0, y=0, relwidth=1, relheight=1)
    root.mainloop()
