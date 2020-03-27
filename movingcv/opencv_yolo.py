# USAGE
# python opencv_yolo.py --yolo yolo-coco
# python opencv_yolo.py --video dashcam_boston.mp4 --tracker csrt --yolo yolo-coco

# import the necessary packages
import os
import numpy as np
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2
import GS_timing
import serial
import math

'''Some useful functions'''
        
#initialize the serial port to Arduino COM!
ser = serial.Serial('COM3', 115200)

def switchDirection(ser):
    #may need to be hardcoded.
    global u_scan, u_prev, T_sample_inc, error
    u_scan = u_scan*(-1)
    u = u_prev*(-1)
    increments=smooth_u(u, u_prev)
    
    for speed in increments:
        u_prev = speed
        t_start_inc=GS_timing.millis()
        #write out speeds in increments. 
        try:
            #print("Writing: ", str.encode(str(convert_degS_code(speed,error)) + '\n'))
            ser.write(str.encode(str(convert_degS_code(speed,error)) + '\n'))
            #time.sleep(T_sample)
        except (OSError, serial.SerialException):
            #print("Serial Exception Raised")
            ser.close()
            ser = initialize_serial(ser)
        except (OSError, serial.SerialTimeoutException):
            #print("Serial Timeour Exception Raised")
            ser.close()
            ser = initialize_serial(ser)
        
        while (GS_timing.millis() - t_start_inc < T_sample_inc*1000):
            pass #do nothing 
        
    return ser
    
def scanStop(ser, t_pan):	
    global u_scan, u_prev, T_sample_inc	
    t_scan = GS_timing.millis()
    increments=smooth_u(u_scan, u_prev)
    #write rotation that has accumulated since previous iteration.
    for speed in increments:
        u_prev = speed
        t_start_inc=GS_timing.millis()
        #write out speeds in increments. 
        try:
            #print("Writing: ", str.encode(str(convert_degS_code(speed,error)) + '\n'))
            ser.write(str.encode(str(convert_degS_code(speed,1.5)) + '\n'))
            #time.sleep(T_sample)
        except (OSError, serial.SerialException):
            #print("Serial Exception Raised")
            ser.close()
            ser = initialize_serial(ser)
        except (OSError, serial.SerialTimeoutException):
            #print("Serial Timeour Exception Raised")
            ser.close()
            ser = initialize_serial(ser)
        
        while (GS_timing.millis() - t_start_inc < T_sample_inc*1000):
            pass #do nothing 
            
    while (GS_timing.millis() - t_scan < t_pan):
        pass #do nothing	
        
    try:	
        ser.write(str.encode(str(convert_degS_code(0,1.5)) + '\n'))	
        #time.sleep(T_sample)	
    except (OSError, serial.SerialException):	
        #print("Serial Exception Raised")	
        ser.close()	
        ser = initialize_serial(ser)	
    except (OSError, serial.SerialTimeoutException):	
        #print("Serial Timeour Exception Raised")	
        ser.close()	
        ser = initialize_serial(ser)
        
    return ser

def initialize_serial(ser):
    while (ser.isOpen()!=1):
        try:
            ser.close()
            ser = serial.Serial('COM3', 115200)
        except (OSError, serial.SerialException):
            #print("Exception raised in initialize_serial()")
            ser.close()
        except (OSError, serial.SerialTimeoutException):
            ser.close()
            #print("Timeout exception raised in initialize_serial()")
    return ser

def convert_degS_code(degS, error):
    #input is radial speed e (-250,250)deg/s
    #output is digital code between (65, 125) OR (135, 195)
    if degS<=-1:
        return int(degS*(25/250) + 90 + 70*np.heaviside(1-abs(error), 0))
    elif degS>=1:
        return int(degS*(25/250) + 100 + 70*np.heaviside(1-abs(error), 0))
    else:
        #map rotation code less than 1 degS to 0. 
        return 95
        
def convert_degS_code2(degS, error):
    #modified
    #input is radial speed e (-250,250)deg/s
    #output is digital code between (65, 125) OR (135, 195)
    return int(degS*(25/250) + 95 + 70*np.heaviside(3-abs(error), 0))


'''control init'''
#sample time
T_sample = 0.05
T = 0

#B matrix
B = 1

#C matrix
C = 1

#controller gain; error
K1 = -20

#controller gain; goose
K2 = [0, 1]

#state observer gain
L1 = [[-60], [-1600]]

#exosystem observer gain
L2 = -20

#exosystem
S = [[0, 1], [0, 0]]

w1_hat = 0
w2_hat = 0

R = [1, 0]

#observer state
x_hat = 0

#error estimate
e_hat = 0

#controller output
u = 0

#integral
u_int = 0

def update_discrete(error, type):
    global T, u, u_int, w1_hat, w2_hat, x_hat
    T += T_sample
    e = error
    
    if type == 'const':
        u = -0.5/T_sample * e
        
    elif type == 'exo':
        print("exo Not implemented")
        
    elif type == 'exo2':
        print("exo2 Not implemented")

    if abs(u) > 0.7*360:
        u = 0.7*360 * u/abs(u)

    return u

def update(error, type):
    global T, u, u_int, w1_hat, w2_hat, x_hat
    
    T += T_sample
    e = error
    
    if type == 'const':
        
        u = K1 * e
    
    elif type == 'exo':
        
        u_int += u*T_sample
        
        e_hat = C*B*u_int - (R[0]*w1_hat + R[1]*w1_hat)
        
        w1_hat_dot = S[0][0]*w1_hat + S[0][1]*w2_hat + L1[0][0]*(e-e_hat)
        w2_hat_dot = S[1][0]*w1_hat + S[0][1]*w2_hat + L1[1][0]*(e-e_hat)
        
        w1_hat += w1_hat_dot*T_sample
        w2_hat += w2_hat_dot*T_sample
        
        u = K1*e + K2[0]*w1_hat + K2[1]*w2_hat
        
    elif type == 'exo2':
        
        e_hat = C*x_hat - (R[0]*w1_hat + R[1]*w1_hat)
        
        w1_hat_dot = S[0][0]*w1_hat + S[0][1]*w2_hat + L1[0][0]*(e-e_hat)
        w2_hat_dot = S[1][0]*w1_hat + S[0][1]*w2_hat + L1[1][0]*(e-e_hat)
        
        x_hat_dot = B*u + L2*(e-e_hat)
        
        w1_hat += w1_hat_dot*T_sample
        w2_hat += w2_hat_dot*T_sample
        
        x_hat += x_hat_dot*T_sample
        
        u = K1*e + K2[0]*w1_hat + K2[1]*w2_hat
        
    else:
        print('Invalid type')
        return
    
    #saturation at 0.7 rot/s. u given in deg/s
    if abs(u) > 0.01*360:
        u = 0.01*360 * u/abs(u)
    if abs(u) < 10:
        u = 0
    
    return u

'''cv starts here'''

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str,
                help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="kcf",
                help="OpenCV object tracker type")
ap.add_argument("-y", "--yolo", required=True,
                help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
ap.add_argument("--threshold", type=float, default=0.3,
                help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent TWO possible class labels
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(2, 3),
                           dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream and frame dimensions
if not args.get("video", False):
    # if a video path was not supplied, grab the reference
    # to the webcam
    vs = VideoStream(src=0).start()
else:
    # otherwise, grab a reference to the video file
    vs = cv2.VideoCapture(args["video"])
writer = None
(W, H) = (None, None)

# define the lower and upper boundaries of the "red"
# boundaries in the HSV color space
redLower = (81, 102, 40)
redUpper = (253, 255, 87)

# initialize the two classes we care about
BIRD_CLASS_ID = 14
PERSON_CLASS_ID = 0

# initialize a dictionary that maps strings to their corresponding
# OpenCV object tracker implementations
OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    "boosting": cv2.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
    "tld": cv2.TrackerTLD_create,
    "medianflow": cv2.TrackerMedianFlow_create,
    "mosse": cv2.TrackerMOSSE_create
}
tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()

# initialize the bounding box coordinates of the object we are going
# to track
initBB = None

# if a video path was not supplied, grab the reference to the web cam
if not args.get("video", False):
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(1.0)

# otherwise, grab a reference to the video file
else:
    vs = cv2.VideoCapture(args["video"])

# initialize the FPS throughput estimator
fps = None

#initialize goose position in frame
coord = 0

#store previous speed, LED value. 
error = 0
u_scan = 100
u_prev = 0

#loop start time
t_start = GS_timing.millis()

# loop over frames from the video stream
while True:

    
    # grab the current frame, then handle if we are using a
    # VideoStream or VideoCapture object
    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame

    # check to see if we have reached the end of the stream
    if frame is None:
        print('End of stream')
        break

    # rotate and resize the frame (so we can process it faster) and grab the
    # frame dimensions
    frame = imutils.rotate_bound(frame, 90)
    frame = imutils.resize(frame, height=500)
    (H, W) = frame.shape[:2]
    
    #get x coordinate of frame centre
    frame_centre = W/2

    # blur the frame and convert it to the HSV color space
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # construct a mask for the color "red", then perform
	# a series of dilations and erosions to remove any small
	# blobs left in the mask
    mask = cv2.inRange(hsv, redLower, redUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # find contours in the mask and initialize the current
	# (x, y) center of the red boundary
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    boundaryCenter = None

    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        boundaryCenter = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        #print(f"DANGER: RED BOUNDARY DETECTED AT {boundaryCenter}")
        ser = switchDirection(ser)
    # check to see if we are currently tracking an object
    if initBB is not None:
        # grab the new bounding box coordinates of the object
        (success, box) = tracker.update(frame)

        # check to see if the tracking was a success
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h),
                          (0, 255, 0), 2)
                          
            #get x coord of goose
            coord = x + w/2

            # update the FPS counter
            fps.update()
            fps.stop()

            # initialize the set of information we'll be displaying on
            # the frame
            info = [
                ("Tracker", args["tracker"]),
                ("Success", "Yes" if success else "No"),
                ("FPS", "{:.2f}".format(fps.fps())),
            ]

            # loop over the info tuples and draw them on our frame
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        else:
            initBB = None
            print('Nothing detected.')
            coord = 0
            error = 9000
            u = 0
            #write control signal u (desired speed) 
            try:
                ser.write(str.encode(str(convert_degS_code2(u,error)) + '\n'))
            except (OSError, serial.SerialException):
                ser.close()
                ser = initialize_serial(ser)
            except (OSError, serial.SerialTimeoutException):
                ser.close()
                ser = initialize_serial(ser)
           

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if not initBB or not success:
        # construct a blob from the input frame and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes
        # and associated probabilities
        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        #start = time.time()
        layerOutputs = net.forward(ln)
        #end = time.time()

        # initialize our lists of detected bounding boxes, confidences,
        # and class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []

        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability)
                # of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > args["confidence"] and classID in {BIRD_CLASS_ID, PERSON_CLASS_ID}:
                    # scale the bounding box coordinates back relative to
                    # the size of the image, keeping in mind that YOLO
                    # actually returns the center (x, y)-coordinates of
                    # the bounding box followed by the boxes' width and
                    # height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top
                    # and and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates,
                    # confidences, and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # apply non-maxima suppression to suppress weak, overlapping
        # bounding boxes
        idxs = cv2.dnn.NMSBoxes(
            boxes, confidences, args["confidence"], args["threshold"])

        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                # draw a bounding box rectangle and label on the frame
                color = [int(c) for c in COLORS[0 if classIDs[i]
                                                == PERSON_CLASS_ID else 1]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                cv2.putText(frame, text, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                if (classIDs[i] == BIRD_CLASS_ID):
                    initBB = tuple(boxes[i])

                    # start OpenCV object tracker using the supplied bounding box
                    # coordinates, then start the FPS throughput estimator as well
                    tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
                    tracker.init(frame, initBB)
                    fps = FPS().start()
                else:
                    print("Human detected!\n")

        # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    '''Control'''
    
    #tracking mode
    if initBB is not None:
        error = 0.0847 * (frame_centre - coord)
        u = update_discrete(error, 'const')
    else:
        error = 9000
        u = 0
        ser = scanStop(ser, 100e-3)
    
    u_prev = u
        
    print('u: '+str(u)+'; error: '+str(error)+'; arduino: '+str(convert_degS_code2(u,error)))
    
    #wait until at least T_sample seconds have elapsed before next loop
    while (GS_timing.millis() - t_start < T_sample * 1000):
        pass
        
    #write control signal u (desired speed) 
    try:
        ser.write(str.encode(str(convert_degS_code2(u,error)) + '\n'))
    except (OSError, serial.SerialException):
        ser.close()
        ser = initialize_serial(ser)
    except (OSError, serial.SerialTimeoutException):
        ser.close()
        ser = initialize_serial(ser)
        
    #loop start time
    t_start = GS_timing.millis()


# if we are using a webcam, release the pointer
if not args.get("video", False):
    vs.stop()

# otherwise, release the file pointer
else:
    vs.release()

# close all windows
cv2.destroyAllWindows()
