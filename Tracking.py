"""
image detector
"""
import argparse
import cv2
import numpy as np
import imutils
import Detection
import math

video_path = 'video3.h264'
single_frame = 0

height_border_ratio = 0.3
width_border_ratio = 0.3


def capture_single_frame(cap, resize_width=0) -> np.ndarray:
    ret, frame = cap.read()
    if not ret:
        return None
    if resize_width:
        frame = imutils.resize(frame, width=resize_width)
    return frame


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

class TrackedObj:
    def __init__(self, x_centroid, y_centroid, age=3):
        self.x_centroid = x_centroid
        self.y_centroid = y_centroid
        self.age = age

def rect_distance(p1: (int, int), p2: (int, int)) -> int:
    # print(type(p1[0]), type(p2[1]))
    return abs(p1[0]-p2[0])+abs(p1[1]-p2[1])

def euclidean_squared_distance(p1: (int, int), p2: (int, int)) -> int:
    return (p1[0]-p2[0])*(p1[0]-p2[0])+(p1[1]-p2[1])*(p1[1]-p2[1])

def euclidean_distance(p1: (int, int), p2: (int, int)) -> float:
    return math.sqrt(euclidean_squared_distance(p1, p2))


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
        self.previous_centroid = centroid
        self.init_centroid = centroid
        pass

tracks = []

def predict_new_location_of_all_tracks():
    for t in tracks:
        bbox = t.bbox
        predicted_centroid = t.kalman.predict()
        predicted_centroid = predicted_centroid.astype(np.int32)
        t.predicted_centroid = (int(predicted_centroid[0][0]), int(predicted_centroid[1][0]))
        # t.centroid = t.predicted_centroid
        box_upper_left = (predicted_centroid[0]-bbox[2]//2, predicted_centroid[1]-bbox[3]//2)
        t.bbox = (box_upper_left[0], box_upper_left[1], bbox[2], bbox[3])

def detection_to_track_assignment(centroids, bboxes):
    # TODO: implement a "real" assignment algorithm
    assignments = []
    unassigned_tracks = []
    unassigned_detections = []
    assigned_centroids = set()
    if centroids and tracks:
        print('centroids:', centroids)
        print('tracks:', [t.centroid for t in tracks])

    furthest_dist = 86.0

    for i in range(len(tracks)):
        min_dist = 2147483647
        closest_centroid = None
        distances = []
        for j in range(len(centroids)):
            current_dist = euclidean_distance(tracks[i].centroid, centroids[j])
            distances.append(current_dist)
            if j in assigned_centroids:
                continue
            if current_dist > furthest_dist:
                # print('too far')
                continue
            if current_dist < min_dist:
                min_dist = current_dist
                closest_centroid = j
        print('track:',tracks[i].id, 'distances:', distances)
        if closest_centroid is not None:
            assignments.append((i, closest_centroid))
            assigned_centroids.add(closest_centroid)
        else:
            print('tracker {} no assign'.format(tracks[i].id))
            unassigned_tracks.append(i)
    unassigned_detections = list(set(range(len(centroids))).difference(assigned_centroids))

    return assignments, unassigned_tracks, unassigned_detections


def detection_to_track_identity_mapping(centroids, bboxes):
    '''
    Identity mapping...
    '''
    assignments = []
    unassigned_tracks = []
    unassigned_detections = []
    if assignments:
        print(assignments)

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
        tracks[track_index].previous_centroid = tracks[track_index].centroid
        # print('uatcentroid:', tracks[track_index].previous_centroid)
        tracks[track_index].centroid = centroid
        # print('updatedcentroid:', centroid)
        tracks[track_index].bbox = bbox
        tracks[track_index].age += 1
        tracks[track_index].total_visible_count += 1
        tracks[track_index].consecutive_invisible_count = 0

def update_unassigned_tracks(unassigned_tracks: [int]):
    for unassignment in unassigned_tracks:
        if unassignment < len(tracks):
            tracks[unassignment].age += 1
            tracks[unassignment].consecutive_invisible_count += 1
            pdc = tracks[unassignment].predicted_centroid
            tracks[unassignment].previous_centroid = (pdc[0], pdc[1])

down_count = 0
up_count = 0
left_count = 0
right_count = 0

def delete_lost_tracks():
    global tracks, down_count, up_count, left_count, right_count
    new_tracks = []
    if len(tracks) == 0:
        return
    invisible_for_too_long = 10
    age_threshold = 8
    height_border_area = int(Detection.img_height * height_border_ratio)
    width_border_area = int(Detection.img_width * width_border_ratio)
    for t in tracks:
        visibility = t.total_visible_count / t.age
        if (t.age < age_threshold and visibility < 0.6):
            print('tracker {} deleted'.format(t.id))
        elif (t.consecutive_invisible_count >= invisible_for_too_long):
            print('tracker {} deleted'.format(t.id))
            if t.init_centroid[1] < height_border_area and t.predicted_centroid[1] > Detection.img_height - height_border_area:
                print('down')
                down_count += 1
            elif t.init_centroid[1] > Detection.img_height - height_border_area and t.predicted_centroid[1] < height_border_area:
                print('up')
                up_count += 1
            if t.init_centroid[0] < width_border_area and t.predicted_centroid[0] > Detection.img_width - width_border_area:
                print('right')
                right_count += 1
            elif t.init_centroid[0] > Detection.img_width - width_border_area and t.predicted_centroid[0] < width_border_area:
                print('left')
                left_count += 1
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
        correct_array = np.array(
            [[np.float32(centroid[0])],
             [np.float32(centroid[1])]])
        # for _ in range(10):
        kalman.predict()
        kalman.correct(correct_array)
        new_track = Track(next_track_id, centroid, bbox, kalman)
        next_track_id = (next_track_id + 1) % 16384
        tracks.append(new_track)
        print('new track centroid', new_track.centroid)


def display_tracking_results(frame, mask, centroids, bboxes):
    min_visible_count = 6
    for t in tracks:
        # reliable tracks
        if t.total_visible_count >= min_visible_count:
            # (x, y, w, h) = t.bbox
            # cv2.rectangle(mask, (x, y), (x+w, y+h), (255, 255, 255), 2)
            # cv2.circle(frame, ((x+x+w)//2, (y+y+h)//2), 3, (0,0,0), 5)
            (x, y) = t.predicted_centroid
            cv2.circle(frame, (x, y), 3, (0,255,255), 5)
            pass

    for i in range(len(centroids)):
        (cx, cy) = centroids[i]
        (x, y, w, h) = bboxes[i]
        cv2.circle(frame, (cx, cy), 1, (255,0,255), 5)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        pass

    cv2.putText(frame, 'down:'+str(down_count), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), thickness=2)
    cv2.putText(frame, 'up:'+str(up_count), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), thickness=2)
    cv2.putText(frame, 'left:'+str(left_count), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), thickness=2)
    cv2.putText(frame, 'right:'+str(right_count), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), thickness=2)


    cv2.imshow('original', frame)
    cv2.imshow('mask', mask)

    pass



def main():
    """
    main function
    :return None
    """
    global video_path, \
        binary_thresh, \
        dilate_kernel, \
        erode_kernel, \
        min_contour_area, \
        max_contour_area
    cap = cv2.VideoCapture(video_path)
    last_ref_frame = capture_single_frame(cap, resize_width=Detection.img_width)
    ref_frame = capture_single_frame(cap, resize_width=Detection.img_width)
    print(ref_frame.shape)
    Detection.img_height = ref_frame.shape[0]
    Detection.img_diff_init(ref_frame, last_ref_frame)
    frame_cnt = 1
    while cap.isOpened():
        if single_frame:
            print('FRAME {}'.format(frame_cnt))
        frame = capture_single_frame(cap, resize_width=Detection.img_width)
        if frame is None:
            break
        centroids, bboxes, mask = Detection.detect_objects(frame)
        # print('centroids:', centroids)
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
        frame_cnt += 1

    cv2.destroyAllWindows()
    cap.release()



if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-v", "--video", type=str)
    # arg_parser.add_argument("-s", "--single", type=bool, nargs='?', const=True)
    args = arg_parser.parse_args()
    if args.video:
        video_path = args.video
    # if args.single:
    #     single_frame = 1
    main()