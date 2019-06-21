# import time
import sys
import cv2


video_path = 'video2.h264'
frame_rate = 0
single_frame = 0

if len(sys.argv) > 1:
    video_path = sys.argv[1]

if len(sys.argv) > 2:
    frame_rate = int(sys.argv[2])


def process_key() -> bool:
    """
    process keyboard input
    """
    global single_frame
    wait_time = 1 - single_frame
    key = cv2.waitKey(wait_time) & 0xFF
    if key == ord('q'):
        return True
    elif key == ord(' '):
        single_frame = 1 - single_frame
        return False
    elif key == ord('s'):
        return False
    return False


if __name__ == '__main__':
    cap = cv2.VideoCapture(video_path)
    _, frame = cap.read()
    print(frame.shape)
    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        try:
            cv2.imshow('video', frame)
        except cv2.error as e:
            print(e)
            break
        if process_key():
            break
    cap.release()
    cv2.destroyAllWindows()
