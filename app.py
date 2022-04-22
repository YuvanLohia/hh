from flask import Flask, render_template, Response
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import dlib
import constants as _constant
import sounddevice as sd
import numpy as np
import eye
import multiprocessing
import threading
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

currentframe = 0
currentChar = ''

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import LoadStreams
from utils.general import (LOGGER,check_img_size, check_imshow,
                           increment_path, non_max_suppression, scale_coords)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, time_sync

@torch.no_grad()
def run():
    
    
    weights = "best.pt"
    imgsz = [416,416]
    conf_thres=0.5
    half = False
    im0 = None
    app = Flask(__name__)
    def gen_frames():  # generate frame by frame from camera
        while True:
           
            ret, buffer = cv2.imencode('.jpg', im0)
            frame = buffer.tobytes()
                
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


    @app.route('/video_feed')
    def video_feed():
        #Video streaming route. Put this in the src attribute of an img tag
        return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


    @app.route('/')
    def index():
        """Video streaming home page."""
        return render_template('index.html')
    
    virtual =  False
    if virtual:
        #cam = pyvirtualcam.Camera(1920, 1080, 30, fmt=PixelFormat.BGR)
        cam = pyvirtualcam.Camera(640, 480, 30, fmt=PixelFormat.BGR)
    def flask_app_run(p):
        app.run(debug=False)
    v = threading.Thread(target=flask_app_run,args=(0,))
    v.start()
    save_img = True
    
    save_dir = increment_path(Path(ROOT / 'runs/detect') / "exp", exist_ok=False)  # increment run
    (save_dir / 'labels' if False else save_dir).mkdir(parents=True, exist_ok=True)

    # Load model
    device = select_device('')
    model = DetectMultiBackend(weights, device=device, dnn=False)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= (pt or jit or engine) and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()

    # Dataloader
   
    view_img = check_imshow()
    cudnn.benchmark = True  # set True to speed up constant image size inference
    dataset = LoadStreams("0", img_size=imgsz, stride=stride, auto=pt)
    bs = len(dataset)  # batch_size
    
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if False else False
        pred = model(im, augment=False, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, 0.45, None, False, max_det=1000)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            
            p, im0, frame = path[i], im0s[i].copy(), dataset.count
            s += f'{i}: '
            
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if False else im0
            annotator = Annotator(im0, line_width=3, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    if {names[int(c)]} != "right" and {names[int(c)]} != "left":
                        s += f"{names[int(c)]}"  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    

                    if save_img or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if False else (names[c] if False else f'{names[c]}')
                        if label != None and label != "left" and label != "right":
                            annotator.box_label(xyxy, label, color=colors(c, True))
                        

            # Print time (inference-only)
            LOGGER.info(s)

            # Stream results
            im0 = annotator.result()
            #   cv2.imshow("Frame", im0)
            if virtual:
                
                #imj = cv2.flip(im0,1)
                imj = im0
                cam.send(imj)
                cam.sleep_until_next_frame()
            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)
            gray = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            if len(faces) > 1:
                print("2")
                #raise IndexError
            elif len(faces) == 0:
                print("0")
                #raise IndexError
            else: 
            
                gaze = 0
                landmarks = predictor(gray, faces[0])

                # brinking events
                left_blink_ratio = eye.get_blinking_ratio(_constant.left_eye, landmarks, im0)
                right_blink_ratio = eye.get_blinking_ratio(_constant.right_eye, landmarks, im0)
                both_eye = (left_blink_ratio + right_blink_ratio) / 2
                # print(both_eye)
                if (both_eye > _constant.blinking_ratio):
                    pass

                else:
                # # gazing events
                    left_gaze_ratio = eye.get_gaze_ratio(_constant.left_eye, landmarks, im0, gray)
                    right_gaze_ratio = eye.get_gaze_ratio(_constant.right_eye, landmarks, im0, gray)
                    if left_gaze_ratio != None and right_gaze_ratio != None:
                        average_age_ratio = (left_gaze_ratio + right_gaze_ratio) / 2
                        gaze = average_age_ratio
                        
                        
                        if gaze > 5.1 or gaze < 0.4:
                            #raise IndexError
                            pass
                        
            
            


    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    




h = True
def print_sound(indata, outdata, frames, time, status):
    global h
    volume_norm = np.linalg.norm(indata)*10
    print ("|" * int(volume_norm))
    if int(volume_norm) > 3 and int(volume_norm) < 10:
        h = False
def run1():
    global h
    while h == True:
        with sd.Stream(callback=print_sound):
            sd.sleep(1000)
        print("yo")

p = multiprocessing.Process(target=run) 
c = multiprocessing.Process(target=run1)




    


'''


if __name__ == "__main__":
    #c.start()
    p.start()
    
    while True:
        #if c.is_alive() == False:
            
           # p.terminate()
           # break
        #if p.is_alive() == False:
            
            #c.terminate()
            # break
'''
if __name__ == "__main__":
    p.start()
