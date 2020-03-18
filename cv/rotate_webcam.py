# Opens a window with the USB webcam feed through terminal
from imutils.video import VideoStream

import numpy as np
import cv2
import time
import imutils

# Open the device at the ID 0
# vs = cv2.VideoCapture(0)
vs = VideoStream(src=0).start()


#Check whether user selected camera is opened successfully.

# if not (vs.isOpened()):
# 	print("Could not open video device")

while(True):
	# load the image from disk
	# image = cv2.imread(args["image"])
	# Capture frame-by-frame
	# ret, frame = vs.read()
	frame = vs.read()
	rotated = imutils.rotate_bound(frame, 90)

	# Display the resulting frame
	cv2.imshow('preview',rotated)

	#Waits for a user input to quit the application
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# vs.stop()
vs.release()
cv2.destroyAllWindows()

print("[INFO] cleaning up...")



