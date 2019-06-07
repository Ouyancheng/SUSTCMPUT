# import time
import sys
import cv2


video_path = 'video2.h264'
frame_rate = 0

if len(sys.argv) > 1:
    video_path = sys.argv[1]

if len(sys.argv) > 2:
    frame_rate = int(sys.argv[2])

def process_key() -> bool:
    """
    process keyboard input
    """
    force_quit = False
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        force_quit = True
    elif key == ord('p') or key == ord(' '):
        while 1:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('r') or key == ord(' '):
                break
            if key == ord('q'):
                force_quit = True
                break
    return force_quit


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

