import time
import sys
import cv2


video_path = 'video2.h264'
frame_rate = 0

if len(sys.argv) > 1: 
    video_path = sys.argv[1]

if len(sys.argv) > 2: 
    frame_rate = int(sys.argv[2])



if __name__ == '__main__': 
    cap = cv2.VideoCapture(video_path)
    while (cap.isOpened()): 
        ret, frame = cap.read()
        try:
            cv2.imshow('video', frame)
        except cv2.error as e:
            print(e)
            print('done!')
            break
        if (cv2.waitKey(1000//frame_rate if frame_rate > 0 else 1) & 0xFF) == ord('q'): 
            break
    cap.release()
    cv2.destroyAllWindows()

