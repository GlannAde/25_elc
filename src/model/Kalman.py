import cv2
import numpy as np

class KalmanFilter2D:
    """
    2D 卡尔曼滤波器
    将 X 和 Y 轴合并为一个滤波器，减半底层的 API 调用开销。
    状态量: [x, y, vx, vy] (位置和速度)
    观测量: [x, y] (仅位置)
    """
    def __init__(self, q_scale=0.01, r_scale=0.5, default_dt=1/120.0):
        self.q_scale = q_scale
        self.r_scale = r_scale
        self.dt = default_dt

        # 状态维度4 (x, y, vx, vy)，观测维度2 (x, y)
        self.kf = cv2.KalmanFilter(4, 2)

        # 状态转移矩阵 (F)
        # x' = x + vx * dt | y' = y + vy * dt
        self.kf.transitionMatrix = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)

        # 观测矩阵 (H)
        # 我们只能观测到 x 和 y
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], np.float32)

        # 过程噪声协方差 (Q) - 信任预测模型的程度
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * q_scale

        # 测量噪声协方差 (R) - 信任传感器的程度
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * r_scale

        # 初始化协方差矩阵
        self.reset()

    def predict(self, dt=None):
        """预测目标的下一步位置"""
        # 如果相机掉帧或帧率波动，动态修正时间步长 dt
        if dt is not None and dt > 0:
            self.dt = dt
            self.kf.transitionMatrix[0, 2] = dt
            self.kf.transitionMatrix[1, 3] = dt

        prediction = self.kf.predict()
        return prediction[0, 0], prediction[1, 0] # 返回预测的 (x, y)

    def update(self, center):
        """利用视觉识别到的真实中心点 (x, y) 修正模型"""
        measure = np.array([[np.float32(center[0])],
                            [np.float32(center[1])]])
        estimate = self.kf.correct(measure)
        return estimate[0, 0], estimate[1, 0] # 返回滤波后的最优平滑坐标 (x, y)

    def reset(self):
        """目标丢失后重新捕获时调用，清空历史惯性"""
        self.kf.statePost = np.zeros((4, 1), np.float32)
        # 将误差协方差设为极大值 (1000)，强制滤波器在第一帧瞬间 100% 相信观测值
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 1000.0

    def get_state(self):
        """获取当前滤波器的最优位置"""
        return self.kf.statePost[0, 0], self.kf.statePost[1, 0]
