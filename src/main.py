import time
import cv2

# --- 导入视觉与控制模块 ---
from models.cam import Camera
from models.detector import Detector
from models.pid import PIDController
from models.status import GPIN

# --- 导入硬件驱动模块 ---
from models.stepper import EmmMotor
from models.tracker import Status, Tracker

# ==================== 系统参数配置区 ====================
CAMERA_INDEX = 0  # 摄像头索引 (外接通常为0或1)
YAW_PORT = "/dev/ttyACM0"  # Yaw轴电机串口 (Linux下可能是 /dev/ttyACM0)
PITCH_PORT = "/dev/ttyACM1"  # Pitch轴电机串口 (根据实际情况修改)

USE_KF = True  # 是否启用卡尔曼滤波预测
SHOW_WINDOWS = True  # 是否显示调试画面和参数控制台

# 齿轮传动比配置 (电机转动角度 / 云台实际转动角度)
# Pitch轴: 13:120 -> 电机转120度云台转13度 -> 传动比 = 120/13 ≈ 9.23
# Yaw轴: 11:56 -> 电机转56度云台转11度 -> 传动比 = 56/11 ≈ 5.09
GEAR_RATIO_YAW = 5.09     # Yaw轴传动比 (11:56)
GEAR_RATIO_PITCH = 9.23   # Pitch轴传动比 (13:120)
# ========================================================

# ==================== 模块初始化 ====================
camera = Camera(index=CAMERA_INDEX, width=640, height=480)
detector = Detector(min_area=5000, max_area=500000)
tracker = Tracker(f_pixel_h=725.6, real_height=17.5, use_kf=USE_KF)

# 电机初始化 (注意检查你的两电机 ID 是否分别设置为了 1 和 2)
stepper_yaw = EmmMotor(port=YAW_PORT, baudrate=115200, timeout=1, motor_id=1)
stepper_pitch = EmmMotor(port=PITCH_PORT, baudrate=115200, timeout=1, motor_id=2)

# PID 初始化 (初始值会被滑动条覆盖)
pid_yaw = PIDController(Kp=0.0, Ki=0.0, Kd=0.0, dt=1 / 30)
pid_pitch = PIDController(Kp=0.0, Ki=0.0, Kd=0.0, dt=1 / 30)

# GPIO 外设
lazer = GPIN(pin=16, mode=1)  # 激光笔控制
heart_beat = GPIN(pin=18, mode=1)  # 系统心跳灯
# ========================================================

def nothing(x):
    pass


def init_board():
    """初始化调试窗口和 PID 动态调参滑动条"""
    cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Controls", 400, 350)
    cv2.namedWindow("Tracker", cv2.WINDOW_FREERATIO)

    # Yaw 轴参数 (考虑到滑动条只能是整数，通过除以倍数来获得浮点数)
    cv2.createTrackbar("yaw_kp", "Controls", 20, 1000, nothing)  # 真实值 = val / 1000
    cv2.createTrackbar("yaw_ki", "Controls", 0, 1000, nothing)
    cv2.createTrackbar("yaw_kd", "Controls", 10, 1000, nothing)

    # Pitch 轴参数
    cv2.createTrackbar("pitch_kp", "Controls", 20, 1000, nothing)
    cv2.createTrackbar("pitch_ki", "Controls", 0, 1000, nothing)
    cv2.createTrackbar("pitch_kd", "Controls", 10, 1000, nothing)

    # 电机运动参数
    cv2.createTrackbar("vel_rpm", "Controls", 500, 5000, nothing)
    cv2.createTrackbar("acc", "Controls", 100, 255, nothing)


def update_params():
    """读取滑块参数并实时更新给算法层"""
    # 获取并换算 PID 参数
    yaw_kp = cv2.getTrackbarPos("yaw_kp", "Controls") / 1000.0
    yaw_ki = cv2.getTrackbarPos("yaw_ki", "Controls") / 100000.0
    yaw_kd = cv2.getTrackbarPos("yaw_kd", "Controls") / 100000.0

    pitch_kp = cv2.getTrackbarPos("pitch_kp", "Controls") / 1000.0
    pitch_ki = cv2.getTrackbarPos("pitch_ki", "Controls") / 100000.0
    pitch_kd = cv2.getTrackbarPos("pitch_kd", "Controls") / 100000.0

    vel_rpm = max(1, cv2.getTrackbarPos("vel_rpm", "Controls"))  # 速度不能为0
    acc = max(1, cv2.getTrackbarPos("acc", "Controls"))

    # 动态赋值给 PID 模块
    pid_yaw.set_Kp(yaw_kp)
    pid_yaw.set_Ki(yaw_ki)
    pid_yaw.set_Kd(yaw_kd)

    pid_pitch.set_Kp(pitch_kp)
    pid_pitch.set_Ki(pitch_ki)
    pid_pitch.set_Kd(pitch_kd)

    return vel_rpm, acc


def main():
    print("\n 视觉追踪系统已启动！\n   [按 'q' 键退出]")

    # 1. 硬件上电准备
    try:
        stepper_yaw.emm_v5_en_control(state=True)
        stepper_pitch.emm_v5_en_control(state=True)
    except Exception as e:
        print(f"电机使能失败: {e}。程序将继续运行以测试视觉。")

    if SHOW_WINDOWS:
        init_board()

    prev_time = time.time()

    try:
        while True:
            # 1. 刷新硬件心跳狗
            heart_beat.flash()

            # 2. 极速读帧
            ret, frame = camera.read()
            if not ret or frame is None:
                print("无法获取画面，重试中...")
                time.sleep(0.01)
                continue

            # 3. 动态获取 GUI 参数
            if SHOW_WINDOWS:
                vel_rpm, acc = update_params()
            else:
                vel_rpm, acc = 500, 100  # 无界面时的默认保守值

            # 4. 视觉识别与滤波解算
            target = detector.detect(frame)
            yaw_err, pitch_err, dist, status, laser_pos, smooth_center = tracker.track(
                target
            )

            # 5. FPS 计算
            curr_time = time.time()
            fps = 1.0 / max(curr_time - prev_time, 1e-6)
            prev_time = curr_time

            # 6. 开火决策
            if tracker.onfire:
                lazer.set_value(1)
            else:
                lazer.set_value(0)

            # 7. 电机闭环控制逻辑
            if status in (Status.TRACK, Status.TMP_LOST):
                # 解算 PID 输出量 (即当前帧需要增减的角度)
                correction_yaw = pid_yaw.compute(yaw_err)
                correction_pitch = pid_pitch.compute(pitch_err)

                try:
                    # 关键动作：abs_mode=False 代表"相对运动"。
                    # 即让电机在当前位置的基础上，再转动 correction 的度数去追赶目标。
                    # 应用齿轮传动比补偿：电机需要多转 GEAR_RATIO 倍才能达到期望的云台角度
                    stepper_yaw.emm_v5_move_to_angle(
                        angle_deg=-correction_yaw * GEAR_RATIO_YAW,
                        vel_rpm=vel_rpm,
                        acc=acc,
                        abs_mode=False,
                    )

                    stepper_pitch.emm_v5_move_to_angle(
                        angle_deg=-correction_pitch * GEAR_RATIO_PITCH,
                        vel_rpm=vel_rpm,
                        acc=acc,
                        abs_mode=False,
                    )
                except Exception as e:
                    print(f"电机控制指令异常: {e}")

            elif status == Status.LOST:
                # 目标丢失，清空积分池，防止云台再次捕获时发生剧烈甩头
                pid_yaw.reset()
                pid_pitch.reset()

                # 可选：让电机立即停止转动
                # stepper_yaw.emm_v5_stop_now()
                # stepper_pitch.emm_v5_stop_now()

            # 8. 终端状态打印
            status_map = {
                Status.TRACK: "TRACKING",
                Status.TMP_LOST: "PREDICTING",
                Status.LOST: "LOST",
            }
            print(
                f"FPS: {fps:.1f} | 状态: {status_map[status]} | Yaw_err: {yaw_err:.2f} | Pitch_err: {pitch_err:.2f}"
            )

            # 9. 画面绘制与显示 (完全解耦，纯 main 逻辑)
            if SHOW_WINDOWS:
                vis_trk = frame.copy()

                # 如果正在追踪，画出经过 3D 卡尔曼平滑后的丝滑准星
                if status != Status.LOST and smooth_center:
                    cv2.drawMarker(
                        vis_trk,
                        smooth_center,
                        (0, 255, 0),  # 绿色丝滑准星
                        cv2.MARKER_CROSS,
                        20,
                        2,
                    )

                # 如果解算出了激光落点，画个红点
                if laser_pos:
                    cv2.circle(vis_trk, laser_pos, 4, (0, 0, 255), -1)

                # 将调试数据直接写在画面上
                cv2.putText(
                    vis_trk,
                    f"FPS: {fps:.1f} | {status_map[status]}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("Tracker", vis_trk)

            # 10. 退出指令
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except Exception as e:
        print(f"\n 主循环发生异常: {str(e)}")
    except KeyboardInterrupt:
        print("\n 收到键盘终止信号...")
    finally:
        print("\n 正在安全释放系统资源...")
        camera.release()
        try:
            # 安全第一：退出时务必让电机急停并去使能，防止程序关了电机还在疯转
            stepper_yaw.emm_v5_stop_now()
            stepper_pitch.emm_v5_stop_now()
            stepper_yaw.emm_v5_en_control(state=False)
            stepper_pitch.emm_v5_en_control(state=False)
            stepper_yaw.close()
            stepper_pitch.close()
        except Exception as e:
            print(f"电机关闭异常: {e}")

        lazer.cleanup()
        heart_beat.cleanup()
        cv2.destroyAllWindows()
        print("系统已安全彻底关闭。")


if __name__ == "__main__":
    main()
