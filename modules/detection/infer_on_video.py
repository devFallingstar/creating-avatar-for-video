import cv2
from PIL import Image
import argparse
from pathlib import Path
import torch
from config import get_config
from mtcnn import MTCNN
from Learner import face_learner
from utils import load_facebank, draw_box_name, prepare_facebank
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-f", "--file_name", help="video file name",default='video.mp4', type=str)
    parser.add_argument("-s", "--save_name", help="output file name",default='recording', type=str)
    parser.add_argument('-th','--threshold',help='threshold to decide identical faces',default=0.5, type=float)
    parser.add_argument("-u", "--update", help="whether perform update the facebank",action="store_true", default=True)
    parser.add_argument("-tta", "--tta", help="whether test time augmentation",action="store_true")
    parser.add_argument("-c", "--score", help="whether show the confidence score",action="store_true", default=True)
    parser.add_argument("-b", "--begin", help="from when to start detection(in seconds)", default=0, type=int)
    parser.add_argument("-d", "--duration", help="perform detection for how long(in seconds)", default=0, type=int)

    args = parser.parse_args()

    conf = get_config(False)
    mtcnn = MTCNN(select_largest=False, keep_all=True)
    print('mtcnn loaded')
    
    learner = face_learner(conf, True)
    learner.threshold = args.threshold
    if conf.device.type == 'cpu':
        learner.load_state(conf, 'cpu_final.pth', True, True)
    else:
        learner.load_state(conf, 'final.pth', True, True)
    learner.model.eval()
    print('learner loaded')
    
    if args.update:
        targets, names = prepare_facebank(conf, learner.model, mtcnn, tta = args.tta)
        print('facebank updated')
    else:
        targets, names = load_facebank(conf)
        print('facebank loaded')
        
    cap = cv2.VideoCapture(str(conf.facebank_path/args.file_name))
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.set(cv2.CAP_PROP_POS_MSEC, args.begin * 1000)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_writer = cv2.VideoWriter(str(conf.facebank_path/'{}.avi'.format(args.save_name)),
                                   cv2.VideoWriter_fourcc(*'XVID'), int(fps), (int(cap.get(3)),int(cap.get(4))))
    #video_writer = cv2.VideoWriter(str(conf.facebank_path/'{}.avi'.format(args.save_name)),
    #                               cv2.CAP_OPENCV_MJPEG, int(fps), (int(cap.get(3)),int(cap.get(4))))
    
    if args.duration != 0:
        i = 0
    #### csv
    df = pd.DataFrame(columns=['frame_number', 'ID', 'LT', 'RB', 'score'])

    
    framecounter = 0
    while cap.isOpened():
        isSuccess,frame = cap.read()
        if isSuccess:            
#             image = Image.fromarray(frame[...,::-1]) #bgr to rgb
            image = Image.fromarray(frame)
            try:
                bboxes, faces = mtcnn.align_multi(image, conf.face_limit, 16)
                # mtcnn 에서 검출된 얼굴 BB 와 5point landmark
            except:
                bboxes = []
                faces = []
            if len(bboxes) == 0:
                print('no face')
                # continue
            else:
                bboxes = bboxes[:,:-1] #shape:[10,4],only keep 10 highest possibiity faces
                bboxes = bboxes.astype(int)
                bboxes = bboxes + [-1,-1,1,1] # personal choice
                # 사람 identification   
                results, score = learner.infer(conf, faces, targets, True)
                for idx,bbox in enumerate(bboxes):
                    if args.score:
                        frame = draw_box_name(bbox, names[results[idx] + 1] + '_{:.2f}'.format(score[idx]), frame)
                    else:
                        frame = draw_box_name(bbox, names[results[idx] + 1], frame)
                    df = df.append({'frame_number':framecounter, 'ID':names[results[idx] + 1], 'LT':(bbox[0],bbox[1]), 'RB':(bbox[2],bbox[3]), 'score':'{:.2f}'.format(score[idx])}, ignore_index=True)

            video_writer.write(frame)
            print('{0:.2f}' .format(framecounter/duration*100))
            framecounter +=1
        else:
            break
        if args.duration != 0:
            i += 1
            if i % 25 == 0:
                print('{} second'.format(i // 25))
            if i > 25 * args.duration:
                break        
    cap.release()
    video_writer.release()
    df.to_csv('./data/facebank/res.csv', index=False)
