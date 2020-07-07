import configparser
import os
import sys

# Modules for landmark detection
from modules.landmarks.id_detect import csv_reader, detect_from_id
# Modules for face detection
from modules.detection.fd_on_video import face_detection

# Modules for TTS
path = os.path.join(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'modules'), 'tts');
sys.path.append(path)
import modules.tts
from modules.tts.combine_generated_audio import combine

def init_path(output_path, source_path):
    if not os.path.isdir(output_path) or not os.path.isdir(source_path):
        os.mkdir(output_path)
        os.mkdir(source_path)


def run_detection(video_path, result_csv_path):
    face_detection(video_path, result_csv_path)
    pass


def run_landmarks(video_path, input_csv_path,
                  result_npy_path = "output/landmarks/", result_frames_path = "output/landmarks/frames"):
    # Default output npy dimension 2D, if want 3D
    # detect_from_id(video_path, csv_reader(input_csv_path), result_npy_path, result_frames_path, "2D")
    if (result_npy_path == "output/landmarks/") & (result_frames_path == "output/landmarks/frames"):
        detect_from_id(video_path, csv_reader(input_csv_path),
                       os.path.join(result_npy_path, video_path.split("/")[-1][:-4]),
                       os.path.join(result_frames_path, video_path.split("/")[-1][:-4]))
    else:
        detect_from_id(video_path, csv_reader(input_csv_path), result_npy_path, result_frames_path)


def run_tts(model_path, subtitle_path, face_path, result_voice_path):
    modules.tts.synthesis(
        checkpoint_path=model_path, 
        preset=os.path.join(os.path.dirname(model_path), 'deepvoice3_vctk2.json'),
        dst_dir=os.path.join(result_voice_path, "persub"),
        srt_path=subtitle_path,
        face_path= face_path
    )
    combine()


if __name__ == '__main__':
    config_parser = configparser.ConfigParser()

    config_parser.read('config.ini')

    output_path = config_parser['SYSTEM']['output_path']
    source_path = config_parser['SYSTEM']['source_path']

    init_path(output_path, source_path)

    run_detection(source_path+"original.mp4", os.path.join(output_path, "detection", "original.csv"))
    run_landmarks(source_path+"original.mp4", os.path.join(output_path, "detection", "original.csv"))

    run_tts(
        model_path=os.path.join(source_path, "tts", "model.pth"), 
        subtitle_path=os.path.join(source_path, "tts", "original.txt"),
        face_path=os.path.join(output_path, "detection", "original.csv"),
        result_voice_path=os.path.join(output_path, "tts")
    )
    tts_output=os.path.join(output_path, "tts", "result.wav")
