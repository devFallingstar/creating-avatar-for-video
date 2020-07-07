import re
import os
import cv2
import dlib
import time
import numpy as np

from imutils import video
from modules.landmarks.utils import LandmarkExtractor

def csv_reader(csv_path):
    with open(csv_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:]
        file_result = []
        for line in lines:
            values = line.split(',')
            # frame_number, id, lt, br, score
            left = int(re.findall(r'\d+', values[2])[0])
            top = int(re.findall(r'\d+', values[3])[0])
            right = int(re.findall(r'\d+', values[4])[0])
            bottom = int(re.findall(r'\d+', values[5])[0])
            face_box = dlib.rectangle(left, top, right, bottom)
            if values[1] == "Unknown":
                file_result.append({"frame":int(values[0]), "ID":values[1], "face_box":face_box})
            else:
                file_result.append({"frame":int(values[0]), "ID":int(values[1]), "face_box":face_box})
    return file_result

def detect_from_id(video_path, csv_file_result, result_npy_path, result_frames_path, dim = "2D"):
    if not os.path.exists(result_npy_path):
        os.makedirs(result_npy_path)
    if not os.path.exists(result_frames_path):
        os.makedirs(result_frames_path)
    cap = cv2.VideoCapture(video_path)
    fps = video.FPS().start()
    count = 0

    LE = LandmarkExtractor(dim=dim)

    if cap.isOpened() == False:
        print("Error opening video stream")

    while cap.isOpened():
        t = time.time()

        ret, frame = cap.read()
        for result in csv_file_result:
            if result["frame"] == count:
                print(result["face_box"])
                landmarks = LE.landmark_extractor(frame, [result["face_box"]])
                if not os.path.exists(os.path.join(result_npy_path, str(count))):
                    os.mkdir(os.path.join(result_npy_path, str(count)))
                np.save(os.path.join(result_npy_path, str(count), str(result["ID"])), landmarks)
                print("npy saved at ", result_npy_path, str(count), str(result["ID"]))
                cv2.imwrite(os.path.join(result_frames_path, "{}_{}.png".format(count, result["ID"])),
                            LE.draw_landmark(frame, [result["face_box"]], landmarks))
                print("Saved {} video {} frame".format(video_path.split('/')[-1], count))

        count += 1
        fps.update()
        print('[INFO] {} frame elapsed time: {:.2f}'.format(count, time.time() - t))

        if count == 5000:
            break
        elif cv2.waitKey(1) & 0xFF == ord('q'):
            break
    fps.stop()
    cap.release()
    cv2.destroyAllWindows()

def landmark_reader(npy_path):
    npy_info = []
    for files in os.listdir(npy_path):
        if files.endswith("npy"):
            npy_contents = files.split('_')
            npy_info.append({"VideoName":npy_contents[0], "frame_number":int(npy_contents[1]), "ID":int(npy_contents[2][:-4]), "landmarks":np.load(os.path.join(npy_path, files))})
            print(npy_info)
    return npy_info