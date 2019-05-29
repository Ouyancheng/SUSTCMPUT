"""
image detector
"""
import argparse
import cv2
import numpy as np
import imutils

binary_thresh = 20
video_path = 'video2.h264'
dilate_kernel = np.ones((6, 6), np.uint8)
erode_kernel = np.ones((3, 3), np.uint8)
img_width = 240 # 480
min_contour_area = 1200
max_contour_area = 6800

class ImgSeq:
    """
    The image sequence, used for computing moving average
    """
    def __init__(self, maxlength, initial_img):
        self.seq = [initial_img] * maxlength
        self.next_index = 0
        self.avg = initial_img
        self.maxlength = maxlength

    def add_img(self, img, update_avg=True):
        """
        Add an image to the buffer while kicking out the oldest one.
        :param img -- the image to add
        :return None
        """
        if update_avg:
            self.avg -= self.seq[self.next_index] // self.maxlength
        self.seq[self.next_index] = img
        if update_avg:
            self.avg += self.seq[self.next_index] // self.maxlength
        self.next_index = (self.next_index + 1) % self.maxlength


def capture_single_frame(cap, resize_width=0) -> np.ndarray:
    ret, frame = cap.read()
    if not ret:
        return None
    if resize_width:
        frame = imutils.resize(frame, width=resize_width)
    return frame

def two_img_diff(frame) -> np.ndarray:
    grayframe = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    grayframe = cv2.GaussianBlur(grayframe, (21, 21), 0)
    frame_diff = cv2.absdiff(grayframe, two_img_diff.last_frame)
    two_img_diff.last_frame = grayframe
    _, frame_thresh = cv2.threshold(frame_diff, binary_thresh, 255, cv2.THRESH_BINARY)
    return frame_thresh


def moving_avg_diff(frame) -> np.ndarray:
    grayframe = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    grayframe = cv2.GaussianBlur(grayframe, (21, 21), 0)
    frame_diff = cv2.absdiff(grayframe, moving_avg_diff.seq.avg)
    moving_avg_diff.seq.add_img(grayframe)
    _, frame_thresh = cv2.threshold(frame_diff, binary_thresh, 255, cv2.THRESH_BINARY)
    return frame_thresh

def three_img_diff(frame) -> np.ndarray:
    grayframe = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    grayframe = cv2.GaussianBlur(grayframe, (21, 21), 0)
    frame_diff = cv2.absdiff(grayframe, three_img_diff.last_frame)
    last_frame_diff = cv2.absdiff(three_img_diff.last_frame, three_img_diff.last_last_frame)
    three_img_diff.last_last_frame = three_img_diff.last_frame
    three_img_diff.last_frame = grayframe
    _, last_frame_thresh = cv2.threshold(last_frame_diff, binary_thresh, 255, cv2.THRESH_BINARY)
    _, frame_thresh = cv2.threshold(frame_diff, binary_thresh, 255, cv2.THRESH_BINARY)
    frame_thresh = cv2.bitwise_and(last_frame_thresh, frame_thresh)
    return frame_thresh

def img_subtract_mog2(frame) -> np.ndarray:
    frame_thresh = img_subtract_mog2.mog.apply(frame, learningRate=0.01)
    _, frame_thresh = cv2.threshold(frame_thresh, 128, 255, cv2.THRESH_BINARY)
    frame_thresh = cv2.erode(frame_thresh, kernel=erode_kernel, iterations=3)
    return frame_thresh



two_img_diff.last_frame = None
moving_avg_diff.seq = None
three_img_diff.last_frame = None
three_img_diff.last_last_frame = None
img_subtract_mog2.mog = None

def img_diff_init(ref_frame, last_ref_frame):
    ref_frame = cv2.cvtColor(ref_frame, cv2.COLOR_RGB2GRAY)
    ref_frame = cv2.GaussianBlur(ref_frame, (21, 21), 0)
    last_ref_frame = cv2.cvtColor(last_ref_frame, cv2.COLOR_RGB2GRAY)
    last_ref_frame = cv2.GaussianBlur(last_ref_frame, (21, 21), 0)
    two_img_diff.last_frame = ref_frame
    moving_avg_diff.seq = ImgSeq(3, ref_frame)
    three_img_diff.last_frame = ref_frame
    three_img_diff.last_last_frame = last_ref_frame
    img_subtract_mog2.mog = cv2.createBackgroundSubtractorMOG2(detectShadows=True)


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



def main():
    """
    main function
    :return None
    """
    global video_path, binary_thresh, dilate_kernel, erode_kernel, img_width
    cap = cv2.VideoCapture(video_path)
    last_ref_frame = capture_single_frame(cap, resize_width=img_width)
    ref_frame = capture_single_frame(cap, resize_width=img_width)
    print(ref_frame.shape)
    img_diff_init(ref_frame, last_ref_frame)
    # roi_x, roi_y, roi_w, roi_h = cv2.selectROI("ROI", ref_frame)
    roi_x, roi_y, roi_w, roi_h = (118, 0, 122, 30)
    roi_counter = 0
    force_quit = False
    while cap.isOpened():
        frame = capture_single_frame(cap, resize_width=img_width)
        if frame is None:
            break

        frame_thresh = two_img_diff(frame)
        # frame_thresh = cv2.erode(frame_thresh, erode_kernel, iterations=1)
        # frame_thresh = cv2.dilate(frame_thresh, dilate_kernel, iterations=3)
        frame_thresh3 = three_img_diff(frame)
        frame_thresh_avg = moving_avg_diff(frame)
        frame_thresh_mog = img_subtract_mog2(frame)

        cv2.imshow('two', frame_thresh)
        cv2.imshow('three', frame_thresh3)
        cv2.imshow('avg', frame_thresh_avg)
        cv2.imshow('mog', frame_thresh_mog)
        cnts, _ = cv2.findContours(frame_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x+roi_w, roi_y+roi_h), (0, 0, 255), 2)
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            if min_contour_area < w * h < max_contour_area:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                x_centroid = (x+x+w)//2
                y_centroid = (y+y+h)//2
                cv2.circle(frame, (x_centroid, y_centroid), 1, (255,0,255), 5)
                if roi_x < x_centroid < roi_x+roi_w and roi_y < y_centroid < roi_y + roi_h:
                    # in ROI
                    roi_counter += 1
                    print(roi_counter)
                    pass

        cv2.imshow('original', frame)

        force_quit = process_key()
        if force_quit:
            break

    cv2.destroyAllWindows()
    cap.release()



if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-v", "--video", type=str)
    args = arg_parser.parse_args()
    if args.video:
        video_path = args.video
    main()
