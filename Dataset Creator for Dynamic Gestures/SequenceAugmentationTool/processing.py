import os

import SequenceProcessing


FILE_PATH = 'C:/Users/gianl/Documents/PA/Dynamic/train_fixed'

OUTPUT_DIR = 'C:/Users/gianl/Documents/PA/Dynamic/train_fixed_out'


def main_whole():
    dir_list = os.listdir(FILE_PATH)
    print(dir_list)

    processor = SequenceProcessing.SequenceProcessor(amount_sequences=1, amount_augmentations=3, whole_sequence=True)
    processor.set_export_video(False)
    processor.set_output_path(OUTPUT_DIR)

    for dir in dir_list:
        file_list = os.listdir(os.path.join(FILE_PATH, dir))

        print(file_list)

        tmp = os.path.join(OUTPUT_DIR, dir)

        if not os.path.exists(tmp):
            os.mkdir(tmp)

        for file in file_list:
            processor.load_video(os.path.join(FILE_PATH, dir, file))
            processor.select_whole_sequence()
            processor.export_data()


def main_amount():
    dir_list = os.listdir(FILE_PATH)
    print(dir_list)

    processor = SequenceProcessing.SequenceProcessor(amount_sequences=135, amount_augmentations=2, whole_sequence=False)
    processor.set_export_video(False)
    processor.set_output_path(OUTPUT_DIR)

    for dir in dir_list:
        file_list = os.listdir(os.path.join(FILE_PATH, dir))

        print(file_list)

        tmp = os.path.join(OUTPUT_DIR, dir)

        if not os.path.exists(tmp):
            os.mkdir(tmp)

        for file in file_list:
            processor.load_video(os.path.join(FILE_PATH, dir, file))
            processor.select_whole_sequence()
            processor.export_data()

if __name__ == "__main__":
    main_whole()

