import cv2
import numpy as np
import model.cam as camera
import model.detector as Detector
import model.tracker as Tracker
import model.serial as Serial
import time

def nothing(x):
    pass

def init_board():
    cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Camera', 1280, 720)
    cv2.namedWindow('Mask', cv2.WINDOW_NORMAL)
    cv2.namedWindow('board', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
    
    cv2.namedWindow('Controls')
    cv2.createTrackbar('H Min', 'Controls', 129, 179, nothing)
    cv2.createTrackbar('H Max', 'Controls', 179, 179, nothing)
    cv2.createTrackbar('S Min', 'Controls', 47, 255, nothing)
    cv2.createTrackbar('S Max', 'Controls', 255, 255, nothing)
    cv2.createTrackbar('V Min', 'Controls', 81, 255, nothing)
    cv2.createTrackbar('V Max', 'Controls', 255, 255, nothing)
    cv2.createTrackbar('light_area', 'Controls', 5, 200, nothing)
    cv2.createTrackbar('board_min_area', 'Controls', 81000, 200000, nothing)
    cv2.createTrackbar('bin_thresh', 'Controls', 50, 255, nothing)
    cv2.createTrackbar('canny_min', 'Controls', 50, 255, nothing)
    cv2.createTrackbar('canny_max', 'Controls', 150, 255, nothing)
    cv2.createTrackbar('kernel_x', 'Controls', 3, 10, nothing)
    cv2.createTrackbar('kernel_y', 'Controls', 3, 10, nothing)

def update_hsv():
    h_min = cv2.getTrackbarPos('H Min', 'Controls')
    h_max = cv2.getTrackbarPos('H Max', 'Controls')
    s_min = cv2.getTrackbarPos('S Min', 'Controls')
    s_max = cv2.getTrackbarPos('S Max', 'Controls')
    v_min = cv2.getTrackbarPos('V Min', 'Controls')
    v_max = cv2.getTrackbarPos('V Max', 'Controls')
    light_area = cv2.getTrackbarPos('light_area', 'Controls')
    board_min_area = cv2.getTrackbarPos('board_min_area', 'Controls')
    bin_thresh = cv2.getTrackbarPos('bin_thresh', 'Controls')
    canny_min = cv2.getTrackbarPos('canny_min', 'Controls')
    canny_max = cv2.getTrackbarPos('canny_max', 'Controls')
    kernel_x = cv2.getTrackbarPos('kernel_x', 'Controls')
    kernel_y = cv2.getTrackbarPos('kernel_y', 'Controls')

    detector.bgr_lower = (h_min, s_min, v_min)
    detector.bgr_upper = (h_max, s_max, v_max)
    detector.light_min_area = light_area
    detector.board_min_area = board_min_area
    detector.bin_val = bin_thresh
    detector.canny_min = canny_min
    detector.canny_max = canny_max
    detector.kernel_x = kernel_x
    detector.kernel_y = kernel_y

def main():
    init_board()
    
    last_time = time.time()
    frame_count = 0
    fps = 0

    try:
        while True:
            ret, frame = cam.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue

            update_hsv()
            
            position = detector.detect(frame)
            yaw, pitch = tracker.track(position, dt=1/120)
            
            # 死区
            if abs(yaw) < 0.5 and abs(pitch) < 0.5:
                yaw = pitch = 0.0

            result = detector.display(frame)
            
            # serial.send_data(-yaw, pitch)   # 需要时再打开

            cv2.imshow('Camera', frame)
            cv2.imshow('Mask', detector.mask)
            cv2.imshow('board', detector.binary)
            cv2.imshow('Result', result)

            # FPS 计算
            current_time = time.time()
            frame_count += 1
            elapsed_time = current_time - last_time
            if elapsed_time >= 1.0:
                fps = frame_count / elapsed_time
                frame_count = 0
                last_time = current_time
                print(f"FPS: {fps:.2f}")

            if cv2.waitKey(1) == ord('q'):
                break

    finally:
        cam.release()           # 正确释放线程和摄像头
        cv2.destroyAllWindows()

# ==================== 初始化部分 ====================
cam = camera.Camera(index=0, width=1280, height=720, fps=60)   # 先保持60

detector = Detector.Detector(
    color=[(13, 255, 152), (0, 51, 110)],
    light_min_area=5,
    board_min_area=81000,
    bin_val=200,
    canny_min=50,
    canny_max=150,
    kernel_x=3,
    kernel_y=3
)

tracker = Tracker.Tracker(frame_add=5)
# serial = Serial.Serial(...)   # 需要时再取消注释

if __name__ == "__main__":
    main()