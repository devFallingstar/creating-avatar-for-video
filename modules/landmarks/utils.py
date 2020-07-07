import numpy as np
import cv2
import dlib
import face_alignment

class FaceDetector(object):
    def __init__(self, detector = None):
        super(FaceDetector, self).__init__()
        if detector is None:
            self.detector = dlib.get_frontal_face_detector()
            dlib.get_frontal_face_detector()
        else:
            self.detector = dlib.shape_predictor(detector)

    def detect_face(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 3 means upsampled 2 times. It makes everything bigger, and allows to detect more
        return self.detector(gray, 2)

class LandmarkExtractor(object):
    def __init__(self, landmark_model_file = None, dim = "2D"):
        super(LandmarkExtractor, self).__init__()

        if (landmark_model_file == None) & (dim == "2D"):
            self.FANET = True
            self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cuda')
        elif (landmark_model_file == None) & (dim == "3D"):
            self.FANET = True
            self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cuda')
        else:
            self.FANET = False
            self.fa = dlib.shape_predictor(landmark_model_file)

    def reshape_for_polyline(self, array):
        if np.asarray(array).shape[1] == 2:
            return np.array(array, np.int32).reshape((-1, 1, 2))
        else:
            return np.array(array, np.int32).reshape((-1, 1, 3))

    def landmark_extractor(self, image, faces):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if (len(faces) is not 0) & self.FANET:
            face_boxes = [[face.left(), face.top(), face.right(), face.bottom()] for face in faces]
            landmarks = self.fa.get_landmarks(gray, face_boxes)
            return landmarks
        elif len(faces) is not 0:
            landmarks = []
            for face in faces:
                detected_landmarks = self.fa(gray, face).parts()
                landmarks.append([[p.x, p.y] for p in detected_landmarks])
            return landmarks
        else:
            print("No face detected")
            return None

    def draw_landmark(self, image, faces, landmarks = None):
        black_image = np.zeros(image.shape, np.uint8)
        for face in faces:
            cv2.rectangle(image, (face.left(), face.top()), (face.right(), face.bottom()), color=(0, 255, 0))

        # perform if there is a landmark
        if landmarks is not None:
            for landmark in landmarks:
                jaw = self.reshape_for_polyline(landmark[0:17])
                left_eyebrow = self.reshape_for_polyline(landmark[22:27])
                right_eyebrow = self.reshape_for_polyline(landmark[17:22])
                nose_bridge = self.reshape_for_polyline(landmark[27:31])
                lower_nose = self.reshape_for_polyline(landmark[30:35])
                left_eye = self.reshape_for_polyline(landmark[42:48])
                right_eye = self.reshape_for_polyline(landmark[36:42])
                outer_lip = self.reshape_for_polyline(landmark[48:60])
                inner_lip = self.reshape_for_polyline(landmark[60:68])

                color = (255, 255, 255)
                thickness = 1

                cv2.polylines(black_image, [jaw[:, :, :2]], False, color, thickness)
                cv2.polylines(black_image, [left_eyebrow[:, :, :2]], False, color, thickness)
                cv2.polylines(black_image, [right_eyebrow[:, :, :2]], False, color, thickness)
                cv2.polylines(black_image, [nose_bridge[:, :, :2]], False, color, thickness)
                cv2.polylines(black_image, [lower_nose[:, :, :2]], True, color, thickness)
                cv2.polylines(black_image, [left_eye[:, :, :2]], True, color, thickness)
                cv2.polylines(black_image, [right_eye[:, :, :2]], True, color, thickness)
                cv2.polylines(black_image, [outer_lip[:, :, :2]], True, color, thickness)
                cv2.polylines(black_image, [inner_lip[:, :, :2]], True, color, thickness)

            return cv2.add(image, black_image)
        else:
            print("No landmark detected")
            return image