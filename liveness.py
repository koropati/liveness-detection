# python liveness.py --shape-predictor shape_predictor_68_face_landmarks.dat --video blink_detection_demo.mp4
# python liveness.py --shape-predictor shape_predictor_68_face_landmarks.dat
# python liveness.py

from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2


def eye_aspect_ratio(eye):
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	C = dist.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", default="shape_predictor_68_face_landmarks.dat",
	help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="",
	help="path to input video file")
args = vars(ap.parse_args())

EYE_AR_THRESH = 0.21
RATIO_THRESH = 0.0017
EYE_AR_CONSEC_FRAMES = 3

COUNTER = 0
TOTAL = 0

print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

print("[INFO] starting video stream thread...")
vs = FileVideoStream(args["video"]).start()
fileStream = True
vs = VideoStream(src=0).start()

fileStream = False
time.sleep(1.0)

arrBlink = []
numberOfFace = 0


while True:
	if fileStream and not vs.more():
		break

	frame = vs.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	rects = detector(gray, 0)


	if numberOfFace != len(rects):
		if len(arrBlink) > numberOfFace :
			arrBlink[numberOfFace:]=[]
		for i in range (len(rects) - numberOfFace):
			arrBlink.append(0)
	numberOfFace = len(rects)

	indexRect = 0
	for rect in rects:

		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		(x, y, w, h) = face_utils.rect_to_bb(rect)

		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)

		# average the eye aspect ratio together for both eyes
		ear = (leftEAR + rightEAR) / 2.0
		ear_ratio = ear/w


		if ear_ratio < RATIO_THRESH:
			COUNTER += 1
		else:
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				TOTAL += 1
				arrBlink[indexRect] += 1
			COUNTER = 0
		
		if arrBlink[indexRect] > 0 :
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
			cv2.putText(frame, "Asli", (x, y),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
		else:
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
			cv2.putText(frame, "Palsu", (x, y),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		indexRect += 1
  
	# show the frame
	cv2.imshow("Dewok Liveness Detection", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
