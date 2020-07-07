import configparser
import os


def init_path(output_path, source_path):
    if not os.path.isdir(output_path) or not os.path.isdir(source_path):
        os.mkdir(output_path)
        os.mkdir(source_path)


def run_detection(video_path, result_csv_path):
    pass


def run_landmarks(video_path, input_csv_path, result_npy_path, result_frames_path):
    pass


def run_tts(model_path, subtitle_path, result_voice_path):
    pass

if __name__ == '__main__':
    config_parser = configparser.ConfigParser()

    config_parser.read('config.ini')

    output_path = config_parser['SYSTEM']['output_path']
    source_path = config_parser['SYSTEM']['source_path']

    init_path(output_path, source_path)

    run_detection(source_path+"original.mp4", output_path+"")
    run_landmarks()
    run_tts()
