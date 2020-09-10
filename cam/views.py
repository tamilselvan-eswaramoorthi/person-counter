import os
import cv2
import imutils
import time
import dlib
import json
import datetime
import numpy as np
from collections import OrderedDict
from scipy.spatial import distance as dist

from django.shortcuts import render
from django.views.decorators import gzip
from django.http import StreamingHttpResponse

class CentroidTracker:
    def __init__(self, maxDisappeared=50, maxDistance=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared
        self.maxDistance = maxDistance

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects
        inputCentroids = np.zeros((len(rects), 2), dtype="int")

        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])

        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            usedRows = set()
            usedCols = set()
            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                if D[row, col] > self.maxDistance:
                    continue
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])
        return self.objects

class TrackableObject:
    def __init__(self, objectID, centroid):
        self.objectID = objectID
        self.centroids = [centroid]
        self.counted = False



base_path=os.path.abspath(os.path.dirname(__file__))

class VideoCamera():
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.writer = None
        self.W = None
        self.H = None
        self.ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
        self.trackers = []
        self.trackableObjects = {}
        self.totalFrames = 0
        self.totalDown = 0
        self.totalUp = 0
        self.skip = 15
        self.status = "Waiting"
        self.rects = []
        self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat","bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant", "sheep","sofa", "train", "tvmonitor"]
        self.net = cv2.dnn.readNetFromCaffe(base_path+'/mobilenet_ssd/MobileNetSSD_deploy.prototxt', base_path+'/mobilenet_ssd/MobileNetSSD_deploy.caffemodel')


    def __del__(self):
        self.video.release()

    def face_eyes(self):
        success, frame = self.video.read()
        if not success:
            return []
        frame = imutils.resize(frame, width=500)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.rects = []
        if self.W is None or self.H is None:
            (self.H, self.W) = frame.shape[:2]

        if self.totalFrames % self.skip == 0:
            self.status = "Detecting"
            self.trackers = []
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (self.W, self.H), 127.5)
            self.net.setInput(blob)
            detections = self.net.forward()
            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.4:
                    idx = int(detections[0, 0, i, 1])
                    if self.CLASSES[idx] != "person":
                        continue
                    box = detections[0, 0, i, 3:7] * np.array([self.W, self.H, self.W, self.H])
                    (startX, startY, endX, endY) = box.astype("int")
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    tracker.start_track(rgb, rect)
                    self.trackers.append(tracker)
        else:
            for tracker in self.trackers:
                self.status = "Tracking"
                tracker.update(rgb)
                pos = tracker.get_position()
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())
                self.rects.append((startX, startY, endX, endY))
        cv2.line(frame, (0, self.H // 2), (self.W, self.H // 2), (0, 255, 255), 2)
        objects = self.ct.update(self.rects)
        for (objectID, centroid) in objects.items():
            to = self.trackableObjects.get(objectID, None)
            if to is None:
                to = TrackableObject(objectID, centroid)
            else:
                y = [c[1] for c in to.centroids]
                direction = centroid[1] - np.mean(y)
                to.centroids.append(centroid)
                if not to.counted:
                    if direction < 0 and centroid[1] < self.H // 2:
                        self.totalUp += 1
                        to.counted = True
                    elif direction > 0 and centroid[1] > self.H // 2:
                        self.totalDown += 1
                        to.counted = True
            self.trackableObjects[objectID] = to
            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
        info = [
            ("outside", self.totalUp),
            ("inside", self.totalDown),
            ("Status", self.status),
        ]
        t = np.ones((frame.shape[0], 100, 3), np.uint16())*255
        cv2.line(t, (0, self.H // 2), (self.W, self.H // 2), (0, 255, 255), 2)

        cv2.putText(t, "outside", (20, self.H - 320),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.putText(t, str(self.totalUp), (20, self.H - 260),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        cv2.putText(t, "inside", (20, self.H - 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.putText(t, str(self.totalDown), (20, self.H - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        frame = np.concatenate((frame, t), axis = 1)


        ret, jpeg = cv2.imencode('.jpg', frame)
        self.totalFrames+=1
        return jpeg.tobytes(), self.totalDown, self.totalUp

ti = None
def gen(camera):
    while True:
        global ti
        frame, a, b = camera.face_eyes()
        st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
        date, t = st.split(' ')
        if t != ti or ti == None:
            with open('resources/log.csv', 'a') as f:
                f.write(str(date)+','+str(t)+','+str(a)+','+str(b)+'\n')
            ti = t
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@gzip.gzip_page
def cam(request):
        return StreamingHttpResponse(gen(VideoCamera()), content_type = "multipart/x-mixed-replace;boundary=frame")
