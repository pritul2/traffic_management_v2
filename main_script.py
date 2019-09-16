"""
Description
@file dynamicTrafficManagement.py

The dataset for this model is taken from the Universitetsboulevarden,
Aalborg, Denmark

This file calculate the density of the traffic and set the value of the timer
for each four lane

"""

"""
# Import statements #
"""
import cv2
import numpy as np
from sklearn.externals import joblib
from threading import *
import time
import yolo_main

# importing linearRegression model pickle file#
model = joblib.load("model.cpickle")

# Calculating the frame number from the given time as an argument#
def calcFrame(x, y):
    frame_time = int((x * 60 + y) * 35)
    return frame_time


def process(frame):
    vidClone = frame.copy()
    print("entered for processing")
    #Finding the roi#
    roi=np.zeros((frame.shape[0],frame.shape[1]),"uint8")
    cv2.rectangle(roi, (62, 60), (242, 180), 255, -1)
    frame=cv2.bitwise_and(frame,frame,mask=roi)

    #Yolo Logic#
    num=yolo_main.detect(frame)
    print("detected vehicles",num)
    arr=np.array(num)
    arr=arr.reshape(-1,1)

    #Obtaining the time#
    time = model.predict(arr)
    print("predicted time is",time)

    if time==0:
        time=30
        print("time not predicted default has been set")
    print("completed processing")
    return time

#Main function for input and output displaying#

if __name__ == "__main__":

    refIm = cv2.imread("refFrame.jpg")
    vid1 = cv2.VideoCapture('latestData.mp4')
    vid2 = cv2.VideoCapture('latestData.mp4')
    vid3 = cv2.VideoCapture('latestData.mp4')
    vid4 = cv2.VideoCapture('latestData.mp4')
    temp = np.zeros(refIm.shape,"uint8")
    timer = temp.copy()

    # setting the video frame for different lanes#
    #For lane1 #
    lane1_start_time = calcFrame(1, 90)
    lane1_end_time = calcFrame(2, 26)
    vid1.set(1, lane1_start_time)
    _, frame1 = vid1.read()

    #For lane2 #
    lane2_start_time = calcFrame(2, 52)
    lane2_end_time = calcFrame(3, 25)
    vid2.set(1, lane2_start_time)
    _, frame2 = vid2.read()

    #For lane3#
    lane3_start_time = calcFrame(6, 56)
    lane3_end_time = calcFrame(7, 26)
    vid3.set(1, lane3_start_time)
    _, frame3 = vid3.read()

    #For lane4#
    lane4_start_time = calcFrame(12, 22)
    lane4_end_time = calcFrame(12, 52)
    vid4.set(1, lane4_start_time)
    _, frame4 = vid4.read()

    # display window. fWin is the final Video#
    st0 = np.hstack((temp, frame1, temp))
    st1 = np.hstack((frame4, timer, frame2))
    st2 = np.hstack((temp, frame3, temp))
    fWin = np.vstack((st0, st1, st2))
    next_predected_time = 0


    while True:
        if next_predected_time == 0:
            predected_time = process(frame1)//2
        else:
            predected_time = next_predected_time//2

        vid1.set(1, calcFrame(2, 15))
        #print("predicted time is",predected_time)
        t0 = time.clock()
        t0 = time.time()

        while (time.time()-t0<=predected_time):
            print("frame 1")
            ret1, frame1 = vid1.read()
            st0 = np.hstack((temp, frame1, temp))
            st1 = np.hstack((frame4, timer, frame2))
            st2 = np.hstack((temp, frame3, temp))
            x, y = int(fWin.shape[0] / 2) + 50, int(fWin.shape[1] / 2) - 80
            cv2.putText(fWin, 'Green Window for Lane 1:', (x - 50, y - 50), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0))
            cv2.putText(fWin, str(predected_time), (x + 10, y), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255))

            fWin = np.vstack((st0, st1, st2))
            cv2.imshow("frame", fWin)
            cv2.waitKey(1)
            rem_time=predected_time-(time.time()-t0)
            if int(rem_time)==5:
                print("processing frame 2")
                next_predected_time=process(frame2)
            print(rem_time)

        predected_time=next_predected_time//2


        #For Frame2#
        t0 = time.clock()
        t0 = time.time()
        next_predected_time = 0
        while (time.time() - t0 <= predected_time):
            print("frame 2")
            ret2, frame2 = vid2.read()
            st0 = np.hstack((temp, frame1, temp))
            st1 = np.hstack((frame4, timer, frame2))
            st2 = np.hstack((temp, frame3, temp))
            x, y = int(fWin.shape[0] / 2) + 50, int(fWin.shape[1] / 2) - 80
            cv2.putText(fWin, 'Green Window for Lane 1:', (x - 50, y - 50), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0))
            cv2.putText(fWin, str(predected_time), (x + 10, y), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255))

            fWin = np.vstack((st0, st1, st2))
            cv2.imshow("frame", fWin)
            cv2.waitKey(1)
            rem_time = predected_time - (time.time() - t0)
            if int(rem_time) == 5:
                print("processing frame3")
                next_predected_time = process(frame3)
            print(rem_time)

        predected_time=next_predected_time


        #For Frame3#
        t0 = time.clock()
        t0 = time.time()
        next_predected_time = 0
        while (time.time() - t0 <= predected_time):
            print("frame 3")
            ret2, frame3 = vid3.read()
            st0 = np.hstack((temp, frame1, temp))
            st1 = np.hstack((frame4, timer, frame2))
            st2 = np.hstack((temp, frame3, temp))
            x, y = int(fWin.shape[0] / 2) + 50, int(fWin.shape[1] / 2) - 80
            cv2.putText(fWin, 'Green Window for Lane 1:', (x - 50, y - 50), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0))
            cv2.putText(fWin, str(predected_time), (x + 10, y), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255))

            fWin = np.vstack((st0, st1, st2))
            cv2.imshow("frame", fWin)
            cv2.waitKey(1)
            rem_time = predected_time - (time.time() - t0)
            if int(rem_time) == 5:
                print("processing frame4")
                next_predected_time = process(frame4)
            print(rem_time)

        predected_time=next_predected_time//2

        #For Frame4#
        t0 = time.clock()
        t0 = time.time()
        next_predected_time = 0
        while (time.time() - t0 <= predected_time):
            print("frame 4")
            ret2, frame4 = vid4.read()
            st0 = np.hstack((temp, frame1, temp))
            st1 = np.hstack((frame4, timer, frame2))
            st2 = np.hstack((temp, frame3, temp))
            x, y = int(fWin.shape[0] / 2) + 50, int(fWin.shape[1] / 2) - 80
            cv2.putText(fWin, 'Green Window for Lane 1:', (x - 50, y - 50), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0))
            cv2.putText(fWin, str(predected_time), (x + 10, y), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255))

            fWin = np.vstack((st0, st1, st2))
            cv2.imshow("frame", fWin)
            cv2.waitKey(1)
            rem_time = predected_time - (time.time() - t0)
            if int(rem_time) == 5:
                print("processing frame 1")
                next_predected_time = process(frame1)
            print(rem_time)