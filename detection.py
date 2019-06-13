import cv2
import numpy as np



binary_thresh = 20
dilate_kernel = np.ones((5, 5), np.uint8)
erode_kernel = np.ones((5, 5), np.uint8)
open_kernel = np.ones((5, 5), np.uint8)
close_kernel = np.ones((25, 25), np.uint8)
close_kernel2 = np.ones((3, 3), np.uint8)
img_width = 480 # 480
img_height = img_width // 1.5  # will be updated according to the real frame
min_contour_area = int(2000 * (img_width/240)**2)
max_contour_area = int(15000 * (img_width/240)**2)

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
        :param update_avg -- whether to update the average or not (default=True)
        :return None
        """
        if update_avg:
            self.avg -= self.seq[self.next_index] // self.maxlength
        self.seq[self.next_index] = img
        if update_avg:
            self.avg += self.seq[self.next_index] // self.maxlength
        self.next_index = (self.next_index + 1) % self.maxlength



def two_img_diff(frame) -> np.ndarray:
    grayframe = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    grayframe = cv2.GaussianBlur(grayframe, (21, 21), 0)
    frame_diff = cv2.absdiff(grayframe, two_img_diff.last_frame)
    two_img_diff.last_frame = grayframe
    _, frame_thresh = cv2.threshold(frame_diff, binary_thresh, 255, cv2.THRESH_BINARY)
    frame_thresh = cv2.dilate(frame_thresh, dilate_kernel, iterations=3)
    return frame_thresh


def moving_avg_diff(frame) -> np.ndarray:
    grayframe = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    grayframe = cv2.GaussianBlur(grayframe, (21, 21), 0)
    frame_diff = cv2.absdiff(grayframe, moving_avg_diff.seq.avg)
    moving_avg_diff.seq.add_img(grayframe)
    _, frame_thresh = cv2.threshold(frame_diff, binary_thresh, 255, cv2.THRESH_BINARY)
    frame_thresh = cv2.dilate(frame_thresh, dilate_kernel, iterations=3)
    return frame_thresh

def three_img_diff(frame) -> np.ndarray:
    """
    |grayframe - last_frame| & |last_frame - last_last_frame|
    """
    grayframe = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    grayframe = cv2.GaussianBlur(grayframe, (21, 21), 0)
    frame_diff = cv2.absdiff(grayframe, three_img_diff.last_frame)
    last_frame_diff = cv2.absdiff(three_img_diff.last_frame, three_img_diff.last_last_frame)
    three_img_diff.last_last_frame = three_img_diff.last_frame
    three_img_diff.last_frame = grayframe
    _, last_frame_thresh = cv2.threshold(last_frame_diff, binary_thresh, 255, cv2.THRESH_BINARY)
    _, frame_thresh = cv2.threshold(frame_diff, binary_thresh, 255, cv2.THRESH_BINARY)
    frame_thresh = cv2.bitwise_and(last_frame_thresh, frame_thresh)
    frame_thresh = cv2.dilate(frame_thresh, dilate_kernel, iterations=3)
    return frame_thresh

def img_subtract_mog2(frame) -> np.ndarray:
    frame_thresh = img_subtract_mog2.mog.apply(frame, learningRate=0.01)
    # cv2.imshow('shadow', frame_thresh)
    _, frame_thresh = cv2.threshold(frame_thresh, 200, 255, cv2.THRESH_BINARY)
    frame_thresh = cv2.morphologyEx(frame_thresh, cv2.MORPH_OPEN, open_kernel)
    frame_thresh = cv2.morphologyEx(frame_thresh, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    # frame_thresh = cv2.erode(frame_thresh, kernel=erode_kernel, iterations=1)
    return frame_thresh

def img_subtract_knn(frame) -> np.ndarray:
    frame_thresh = img_subtract_knn.knn.apply(frame)
    _, frame_thresh = cv2.threshold(frame_thresh, 200, 255, cv2.THRESH_BINARY)
    # frame_thresh = cv2.morphologyEx(frame_thresh, cv2.MORPH_OPEN, open_kernel)
    # frame_thresh = cv2.morphologyEx(frame_thresh, cv2.MORPH_CLOSE, close_kernel)
    return frame_thresh

def img_subtract_gmg(frame) -> np.ndarray:
    frame_thresh = img_subtract_gmg.gmg.apply(frame)
    _, frame_thresh = cv2.threshold(frame_thresh, 200, 255, cv2.THRESH_BINARY)
    # frame_thresh = cv2.morphologyEx(frame_thresh, cv2.MORPH_OPEN, open_kernel)
    # frame_thresh = cv2.morphologyEx(frame_thresh, cv2.MORPH_CLOSE, close_kernel)
    return frame_thresh

def img_subtract_cnt(frame) -> np.ndarray:
    frame_thresh = img_subtract_cnt.cnt.apply(frame)
    _, frame_thresh = cv2.threshold(frame_thresh, 200, 255, cv2.THRESH_BINARY)
    # frame_thresh = cv2.morphologyEx(frame_thresh, cv2.MORPH_OPEN, open_kernel)
    # frame_thresh = cv2.morphologyEx(frame_thresh, cv2.MORPH_CLOSE, close_kernel)
    return frame_thresh

def img_subtract_gsoc(frame) -> np.ndarray:
    frame_thresh = img_subtract_gsoc.gsoc.apply(frame)
    _, frame_thresh = cv2.threshold(frame_thresh, 200, 255, cv2.THRESH_BINARY)
    # frame_thresh = cv2.morphologyEx(frame_thresh, cv2.MORPH_OPEN, open_kernel)
    # frame_thresh = cv2.morphologyEx(frame_thresh, cv2.MORPH_CLOSE, close_kernel)
    return frame_thresh


two_img_diff.last_frame = None
moving_avg_diff.seq = None
three_img_diff.last_frame = None
three_img_diff.last_last_frame = None
img_subtract_mog2.mog = None
img_subtract_knn.knn = None
img_subtract_gmg.gmg = None
img_subtract_cnt.cnt = None
img_subtract_gsoc.gsoc = None

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
    img_subtract_knn.knn = cv2.createBackgroundSubtractorKNN(detectShadows=True)
    img_subtract_gmg.gmg = cv2.bgsegm.createBackgroundSubtractorGMG()
    img_subtract_cnt.cnt = cv2.bgsegm.createBackgroundSubtractorCNT()
    img_subtract_gsoc.gsoc = cv2.bgsegm.createBackgroundSubtractorGSOC()

def frame_subtraction(frame) -> np.array:
    # frame_thresh2 = two_img_diff(frame)
    # frame_thresh3 = three_img_diff(frame)
    # frame_thresh_avg = moving_avg_diff(frame)
    frame_thresh_mog2 = img_subtract_mog2(frame)
    # frame_thresh_knn = img_subtract_knn(frame)
    # frame_thresh_gmg = img_subtract_gmg(frame)
    # frame_thresh_cnt = img_subtract_cnt(frame)
    # frame_thresh_gsoc = img_subtract_gsoc(frame)

    # frame_thresh = cv2.erode(frame_thresh, erode_kernel, iterations=1)
    # frame_thresh = cv2.dilate(frame_thresh, dilate_kernel, iterations=3)
    # return frame_thresh_avg

    # cv2.imshow('two', frame_thresh2)
    # cv2.imshow('three', frame_thresh3)
    # cv2.imshow('avg', frame_thresh_avg)
    # cv2.imshow('mog', frame_thresh_mog)
    return frame_thresh_mog2
    # return frame_thresh_knn
    # return frame_thresh_gmg
    # return frame_thresh_cnt
    # return frame_thresh_gsoc



def detect_objects(frame) -> ('centroids: [(x: int, y: int)]', 'bboxes: [(x: int, y: int, w: int, h: int)]', 'mask: np.array'):
    frame_thresh = frame_subtraction(frame)
    cnts, _ = cv2.findContours(frame_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centroids = []
    bboxes = []
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        # print('rect area:', w*h)
        # print(type(x), type(y), type(w), type(h))
        if (min_contour_area < w * h):
            x_centroid = (x + (x+w)) // 2
            y_centroid = (y + (y+h)) // 2
            bboxes.append((x, y, w, h))
            centroids.append((x_centroid, y_centroid))
        # elif (w * h > max_contour_area):
        #     bboxes.append((x, y, w, h//2))
        #     centroids.append(((x+x+w)//2, (y+y+h//2)//2))
        #     bboxes.append((x, y+h//2, w, h//2))
        #     centroids.append(((x+x+w)//2, (y+h//2+y+h//2+h//2)//2))
        #     pass
        else:
            # cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
            pass

    return centroids, bboxes, frame_thresh

