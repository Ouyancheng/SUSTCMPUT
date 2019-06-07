import cv2
import numpy as np


# x, y, dx, dy
tracker = cv2.KalmanFilter(4, 2)
# we can measure x, y, and they are identity mapping
tracker.measurementMatrix = np.array(
    [[1, 0, 0, 0],
     [0, 1, 0, 0]],
    np.float32)*1
# x' = x + dx, y' = y + dy, dx' = dx, dy' = dy
tracker.transitionMatrix = np.array(
    [[1, 0, 1, 0],
     [0, 1, 0, 1],
     [0, 0, 1, 0],
     [0, 0, 0, 1]],
    np.float32)
# the covariance matrix of the measurement
tracker.processNoiseCov = np.array(
    [[1, 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, 1, 0],
     [0, 0, 0, 1]],
    np.float32) * 0.1

last_measurement = np.array((2, 1), np.float32)
last_prediction = np.zeros((2, 1), np.float32)
current_measurement = np.array((2, 1), np.float32)
current_prediction = np.zeros((2, 1), np.float32)
frame = np.zeros((600, 800, 3), np.uint8)

def mouse_move(event, x, y, s, p):
    global frame, \
        last_measurement, \
        last_prediction, \
        current_prediction, \
        current_measurement, \
        tracker
    last_prediction = current_prediction
    last_measurement = current_measurement
    current_measurement = np.array(
        [[np.float32(x)],
         [np.float32(y)]]
    )
    tracker.correct(current_measurement)
    current_prediction = tracker.predict()
    last_measurement_coord = last_measurement[0], last_measurement[1]
    current_measurement_coord = current_measurement[0], current_measurement[1]
    last_prediction_coord = last_prediction[0], last_prediction[1]
    current_prediction_coord = current_prediction[0], current_prediction[1]
    cv2.line(frame, last_measurement_coord, current_measurement_coord, (255, 0, 0))
    cv2.line(frame, last_prediction_coord, current_prediction_coord, (0, 255, 0))
    pass

if __name__ == '__main__':
    cv2.namedWindow('mouse_tracker')
    cv2.setMouseCallback('mouse_tracker', mouse_move)
    while 1:
        cv2.imshow('mouse_tracker', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
    cv2.destroyAllWindows()




