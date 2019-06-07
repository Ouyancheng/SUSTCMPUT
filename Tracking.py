"""
image detector
"""
import argparse
import cv2
import numpy as np
import imutils

binary_thresh = 20
video_path = 'video2.h264'
dilate_kernel = np.ones((5, 5), np.uint8)
erode_kernel = np.ones((3, 3), np.uint8)
open_kernel = np.ones((5, 5), np.uint8)
close_kernel = np.ones((25, 25), np.uint8)
img_width = 240 # 480
img_height = 160
min_contour_area = 1500 * (img_width//240) * (img_width//240)
max_contour_area = 8000 * (img_width//240) * (img_width//240)

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
    _, frame_thresh = cv2.threshold(frame_thresh, 128, 255, cv2.THRESH_BINARY)
    # frame_thresh = cv2.erode(frame_thresh, kernel=erode_kernel, iterations=3)
    frame_thresh = cv2.morphologyEx(frame_thresh, cv2.MORPH_OPEN, open_kernel)
    frame_thresh = cv2.morphologyEx(frame_thresh, cv2.MORPH_CLOSE, close_kernel)
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

def frame_subtraction(frame) -> np.array:
    # frame_thresh2 = two_img_diff(frame)
    # frame_thresh3 = three_img_diff(frame)
    # frame_thresh_avg = moving_avg_diff(frame)
    frame_thresh_mog = img_subtract_mog2(frame)

    # frame_thresh = cv2.erode(frame_thresh, erode_kernel, iterations=1)
    # frame_thresh = cv2.dilate(frame_thresh, dilate_kernel, iterations=3)

    # cv2.imshow('two', frame_thresh2)
    # cv2.imshow('three', frame_thresh3)
    # cv2.imshow('avg', frame_thresh_avg)
    # cv2.imshow('mog', frame_thresh_mog)
    return frame_thresh_mog



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

class TrackedObj:
    def __init__(self, x_centroid, y_centroid, age=3):
        self.x_centroid = x_centroid
        self.y_centroid = y_centroid
        self.age = age

def rect_distance(x1, y1, x2, y2):
    return (x2-x1)+(y2-y1)


def check_entrance_line_crossing(y, entrance_y, exit_y) -> bool:
    abs_distance = abs(y - entrance_y)
    if abs_distance <= 2 and y < exit_y:
        return True
    else:
        return False

def check_exit_line_crossing(y, entrance_y, exit_y) -> bool:
    abs_distance = abs(y - exit_y)
    if abs_distance <= 2 and y > entrance_y:
        return True
    else:
        return False


def detect_objects(frame) -> ('centroids: [(x: int, y: int)]', 'bboxes: [(x: int, y: int, w: int, h: int)]', 'mask: np.array'):
    frame_thresh = frame_subtraction(frame)
    cnts, _ = cv2.findContours(frame_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centroids = []
    bboxes = []
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        if (min_contour_area < w * h < max_contour_area):
            x_centroid = (x + (x+w)) // 2
            y_centroid = (y + (y+h)) // 2
            bboxes.append((x, y, w, h))
            centroids.append((x_centroid, y_centroid))
        else:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
            pass

    return centroids, bboxes, frame_thresh


class Track:
    def __init__(self, identity, centroid, bbox, kalman):
        self.id = identity
        self.bbox = bbox
        self.kalman = kalman
        self.age = 1
        self.total_visible_count = 1
        self.consecutive_invisible_count = 0
        self.predicted_centroid = centroid
        self.centroid = centroid
        self.init_centroid = centroid
        pass

tracks = []

def predict_new_location_of_all_tracks():
    for t in tracks:
        bbox = t.bbox
        predicted_centroid = t.kalman.predict()
        predicted_centroid = predicted_centroid.astype(np.int32)
        t.predicted_centroid = (predicted_centroid[0], predicted_centroid[1])
        t.centroid = t.predicted_centroid
        box_upper_left = (predicted_centroid[0]-bbox[2]//2, predicted_centroid[1]-bbox[3]//2)
        t.bbox = (box_upper_left[0], box_upper_left[1], bbox[2], bbox[3])

def detection_to_track_assignment(centroids, bboxes):
    # TODO: implement a "real" assignment algorithm
    assignments = []
    unassigned_tracks = []
    unassigned_detections = []
    for i in range(max(len(centroids), len(tracks))):
        if i >= len(centroids):
            unassigned_tracks.append(i)
        elif i >= len(tracks):
            unassigned_detections.append(i)
        else:
            assignments.append((i, i))
    return assignments, unassigned_tracks, unassigned_detections


def update_assigned_tracks(centroids, bboxes, assignments: [('track_index', 'detection_index')]):
    for assignment in assignments:
        track_index = assignment[0]
        detection_index = assignment[1]
        centroid = centroids[detection_index]
        bbox = bboxes[detection_index]
        tracks[track_index].kalman.correct(np.array(
            [[np.float32(centroid[0])],
             [np.float32(centroid[1])]]))
        tracks[track_index].centroid = centroid
        tracks[track_index].bbox = bbox
        tracks[track_index].age += 1
        tracks[track_index].total_visible_count += 1
        tracks[track_index].consecutive_invisible_count = 0

def update_unassigned_tracks(unassigned_tracks: [int]):
    for unassignment in unassigned_tracks:
        if unassignment < len(tracks):
            tracks[unassignment].age += 1
            tracks[unassignment].consecutive_invisible_count += 1

down_count = 0
up_count = 0

def delete_lost_tracks():
    global tracks, down_count, up_count
    new_tracks = []
    if len(tracks) == 0:
        return
    invisible_for_too_long = 20
    age_threshold = 8
    border_area = 40
    for t in tracks:
        visibility = t.total_visible_count / t.age
        if (t.age < age_threshold and visibility < 0.6):
            print('tracker deleted')
        elif (t.consecutive_invisible_count >= invisible_for_too_long):
            print('tracker deleted')
            if t.init_centroid[1] < border_area and t.predicted_centroid[1] > img_height - border_area:
                print('down')
                down_count += 1
            elif t.init_centroid[1] > img_height - border_area and t.predicted_centroid[1] < border_area:
                print('up')
                up_count += 1
            pass
        else:
            new_tracks.append(t)
    tracks = new_tracks

next_track_id = 1

def create_new_tracks(centroids, bboxes, unassigned_detections):
    global next_track_id
    for i in unassigned_detections:
        print('new track created')
        centroid = centroids[i]
        bbox = bboxes[i]
        # x, y, dx, dy
        kalman = cv2.KalmanFilter(4, 2)
        # we can measure x, y, and they are identity mapping
        kalman.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]],
            np.float32)
        # x' = x + dx, y' = y + dy, dx' = dx, dy' = dy
        kalman.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]],
            np.float32)
        # the covariance matrix of the measurement
        kalman.processNoiseCov = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1]],
            np.float32) * 0.1
        kalman.correct(np.array(
            [[np.float32(centroid[0])],
             [np.float32(centroid[1])]]))
        new_track = Track(next_track_id, centroid, bbox, kalman)
        next_track_id = (next_track_id + 1) % 1024
        tracks.append(new_track)

def display_tracking_results(frame, mask, centroids, bboxes):
    min_visible_count = 6
    for t in tracks:
        # reliable tracks
        if t.total_visible_count >= min_visible_count:
            # (x, y, w, h) = t.bbox
            # cv2.rectangle(mask, (x, y), (x+w, y+h), (255, 255, 255), 2)
            # cv2.circle(frame, ((x+x+w)//2, (y+y+h)//2), 3, (0,0,0), 5)
            (x, y) = t.predicted_centroid
            cv2.circle(frame, (x, y), 3, (120,120,0), 5)
            pass

    for i in range(len(centroids)):
        (cx, cy) = centroids[i]
        (x, y, w, h) = bboxes[i]
        cv2.circle(frame, (cx, cy), 1, (255,0,255), 5)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        pass

    cv2.putText(frame, 'down:'+str(down_count), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255))
    cv2.putText(frame, 'up:'+str(up_count), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0))

    cv2.imshow('original', frame)
    cv2.imshow('mask', mask)

    pass



def main():
    """
    main function
    :return None
    """
    global video_path, binary_thresh, dilate_kernel, erode_kernel, img_width, img_height
    cap = cv2.VideoCapture(video_path)
    last_ref_frame = capture_single_frame(cap, resize_width=img_width)
    ref_frame = capture_single_frame(cap, resize_width=img_width)
    print(ref_frame.shape)
    img_height = ref_frame.shape[0]
    img_diff_init(ref_frame, last_ref_frame)

    while cap.isOpened():
        frame = capture_single_frame(cap, resize_width=img_width)
        if frame is None:
            break
        centroids, bboxes, mask = detect_objects(frame)
        predict_new_location_of_all_tracks()
        assignments, unassigned_tracks, unassigned_detections = detection_to_track_assignment(centroids, bboxes)
        # print(assignments, unassigned_tracks, unassigned_detections)
        update_assigned_tracks(centroids, bboxes, assignments)
        update_unassigned_tracks(unassigned_tracks)
        delete_lost_tracks()
        create_new_tracks(centroids, bboxes, unassigned_detections)
        display_tracking_results(frame, mask, centroids, bboxes)

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
