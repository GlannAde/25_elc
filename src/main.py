'''import cv2
import time
import math

# 导入我们优化过的全套模块
from model.cam import Camera
from model.detector import Detector
from model.Kalman import KalmanFilter2D
# from pid import PIDController
# from serial import Serial
# from status import GPIN

# 定义状态机常量
class State:
    LOST = 0
    TRACKING = 1
    PREDICTING = 2

def main():
    print("正在启动 120FPS 视觉云台追踪系统...")

    # --- 1. 初始化硬件接口 ---
    # 相机：记得核对 index，如果是外部 USB 摄像头通常是 0, 1, 4 等
    cam = Camera(index=0, width=640, height=480, format='MJPG', fps=120)
    center_x, center_y = cam.width / 2, cam.height / 2

    # 物理标定参数：务必填入你用 get_pixel_h.py 测出来的值！
    f_pixel_h = 725.6

    # 串口：核对单片机的端口号
    ser = Serial(port='/dev/ttyACM0', baudrate=115200)

    # GPIO：激光与心跳
    lazer = GPIN(pin=16, mode=1)
    heart_beat = GPIN(pin=18, mode=1)

    # --- 2. 初始化算法大脑 ---
    detector = Detector(min_area=5000, max_area=500000, use_otsu=True)
    kf = KalmanFilter2D(q_scale=0.01, r_scale=0.5, default_dt=1/120.0)

    # PID 控制器 (Kp, Ki, Kd 需要上真机慢慢调，先给个保守值)
    pid_yaw = PIDController(Kp=1.2, Ki=0.01, Kd=0.5)
    pid_pitch = PIDController(Kp=1.2, Ki=0.01, Kd=0.5)

    # --- 3. 运行变量 ---
    current_state = State.LOST
    lost_frames_count = 0
    MAX_PREDICT_FRAMES = 15  # 如果连丢 15 帧（约 0.1 秒），彻底判定丢失

    prev_time = time.time()

    try:
        while True:
            # A. 维持心跳 (防卡死看门狗)
            heart_beat.flash()

            # B. 极速读取最新帧
            ret, frame = cam.read()
            if not ret or frame is None:
                continue

            # C. 耗时与 FPS 计算
            curr_time = time.time()
            dt = curr_time - prev_time
            fps = 1.0 / (dt + 1e-6)
            prev_time = curr_time

            # D. 视觉识别 (核心算力层)
            target_board = detector.detect(frame, debug=False)

            # E. 状态机与控制解算
            target_x, target_y = center_x, center_y

            if target_board and target_board.center:
                # 【状态 1：正常捕获】
                current_state = State.TRACKING
                lost_frames_count = 0
                lazer.set_value(1) # 开火
                # 将实际坐标喂给卡尔曼，获取平滑后的最优坐标
                target_x, target_y = kf.update(target_board.center)

            else:
                # 【状态 2/3：目标消失】
                lost_frames_count += 1
                if lost_frames_count < MAX_PREDICT_FRAMES and current_state != State.LOST:
                    # 刚丢不久，依靠卡尔曼的惯性进行预测
                    current_state = State.PREDICTING
                    lazer.set_value(1)
                    target_x, target_y = kf.predict(dt=dt)
                else:
                    # 彻底丢失，系统归零
                    current_state = State.LOST
                    lazer.set_value(0) # 停火
                    kf.reset()         # 清空历史惯性
                    pid_yaw.reset()    # 清空积分，防止疯转
                    pid_pitch.reset()

            # F. 计算电机控制量并下发
            if current_state in [State.TRACKING, State.PREDICTING]:
                # 1. 像素偏差
                error_x = target_x - center_x
                error_y = target_y - center_y

                # 2. 转换为真实物理偏角 (单位：度)
                yaw_angle_err = math.atan(error_x / f_pixel_h) * (180 / math.pi)
                pitch_angle_err = math.atan(error_y / f_pixel_h) * (180 / math.pi)

                # 3. PID 计算增量/速度
                yaw_output = pid_yaw.compute(yaw_angle_err)
                pitch_output = pid_pitch.compute(pitch_angle_err)

                # 4. 串口发送！
                # 警告：根据你电机的接线方向，这里的正负号可能需要取反，例如 yaw=-yaw_output
                ser.send_data(yaw=yaw_output, pitch=pitch_output)

                # 可视化：在预测或追踪位置画个绿色大十字准星
                cv2.drawMarker(frame, (int(target_x), int(target_y)), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)

            # G. 可视化渲染
            # 画中心基准十字 (红色)
            cv2.drawMarker(frame, (int(center_x), int(center_y)), (0, 0, 255), cv2.MARKER_CROSS, 20, 1)

            # 屏幕打印状态
            state_str = ["LOST", "TRACKING", "PREDICTING"][current_state]
            color = (0, 0, 255) if current_state == State.LOST else (0, 255, 0)
            cv2.putText(frame, f"FPS: {fps:.1f} | {state_str}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # 你也可以叠加 detector 的绘制画面
            frame = detector.draw(frame)
            cv2.imshow("Main Tracking", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n收到退出指令...")
    except Exception as e:
        print(f"\n主循环异常: {e}")
    finally:
        print("正在安全释放所有资源...")
        cam.release()
        ser.close()
        lazer.cleanup()
        heart_beat.cleanup()
        cv2.destroyAllWindows()
        print("系统已彻底关闭。")

if __name__ == '__main__':
    main()
'''
