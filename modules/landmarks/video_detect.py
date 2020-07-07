import os
import numpy as np
import cv2
import time
from imutils import video
from modules.landmarks.utils import FaceDetector, LandmarkExtractor

def detect_only_vid(video_path, dim = "2D"):
    # video_path = "parasite-trailer1-result.avi"
    # LS: Landmarks and Images
    result_npy_path = "../output/landmarks/{}".format(video_path.split("/")[-1][:-4])
    result_img_path = "../output/landmarks/frames/{}".format(video_path.split("/")[-1][:-4])
    if not os.path.exists(result_npy_path):
        os.makedirs(result_npy_path)
        os.makedirs(result_img_path)
    print(result_npy_path)
    cap = cv2.VideoCapture(video_path)
    fps = video.FPS().start()
    count = 0

    FD = FaceDetector()
    LE = LandmarkExtractor(dim = dim)

    if cap.isOpened() == False:
        print("Error opening video stream")

    while cap.isOpened():
        t = time.time()

        ret, frame = cap.read()
        faces = FD.detect_face(frame)
        if len(faces) is not 0:
            landmarks = LE.landmark_extractor(frame, faces)
            np.save(os.path.join(result_npy_path, "{}".format(count), landmarks))
            print(faces)
            cv2.imwrite(os.path.join(result_img_path, "{}.png".format(count)), LE.draw_landmark(frame, faces, landmarks))
            print("Saved {} video {} frame".format(video_path.split('/')[-1], count))

        count += 1
        fps.update()
        print('[INFO] {} frame elapsed time: {:.2f}'.format(count, time.time() - t))

        if count == 3000:
            break
        elif cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fps.stop()
    cap.release()
    cv2.destroyAllWindows()

# if __name__ == "__main__":
#     # execute only if run as a script
#     detect_only_vid("../source/parasite-trailer1-result.avi")