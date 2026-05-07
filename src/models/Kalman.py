import cv2
import numpy as np


class KalmanFilter3D:
    """
    高频追踪优化的 3D 卡尔曼滤波器
    3D 滤波器，用于处理 Z轴深度信息
    状态量: [x, y, dist, vx, vy, v_dist] (6维)
    观测量: [x, y, dist] (3维)
    """

    def __init__(
        self, q_scale=0.35, r_xy_scale=0.1, r_dist_scale=2.0, default_dt=1 / 120.0
    ):
        self.dt = default_dt
        self.kf = cv2.KalmanFilter(6, 3)

        # 状态转移矩阵 (F)
        self.kf.transitionMatrix = np.array(
            [
                [1, 0, 0, self.dt, 0, 0],
                [0, 1, 0, 0, self.dt, 0],
                [0, 0, 1, 0, 0, self.dt],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ],
            np.float32,
        )

        # 观测矩阵 (H)
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]], np.float32
        )

        # 过程噪声 (Q) - 信任预测模型的程度
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * q_scale

        # 测量噪声 (R) - 信任传感器的程度 (区分 XY平移 与 Z轴深度 的不同置信度)
        R = np.eye(3, dtype=np.float32)
        R[0, 0] = r_xy_scale
        R[1, 1] = r_xy_scale
        R[2, 2] = r_dist_scale
        self.kf.measurementNoiseCov = R

        self.reset()

        # 保存基准的 Q 矩阵
        self.base_q = np.eye(6, dtype=np.float32) * q_scale
        self.kf.processNoiseCov = self.base_q.copy()

    def predict(self, dt=None):
        """预测目标的下一步位置"""
        if dt is not None and dt > 0:
            self.dt = dt
            self.kf.transitionMatrix[0, 3] = dt
            self.kf.transitionMatrix[1, 4] = dt
            self.kf.transitionMatrix[2, 5] = dt

            # 动态调整过程噪声: 如果这帧等了很久(比如系统卡顿)，预测的不确定性就变大，放大 Q
            scale = dt / (1 / 120.0)
            self.kf.processNoiseCov = self.base_q * scale

        prediction = self.kf.predict()
        return prediction[0, 0], prediction[1, 0], prediction[2, 0]

    def update(self, center_x, center_y, dist):
        """利用视觉识别到的真实 (x, y, dist) 修正模型"""
        measure = np.array(
            [[np.float32(center_x)], [np.float32(center_y)], [np.float32(dist)]]
        )
        estimate = self.kf.correct(measure)
        return estimate[0, 0], estimate[1, 0], estimate[2, 0]

    def reset(self):
        """目标丢失后重新捕获时调用，清空历史惯性"""
        self.kf.statePost = np.zeros((6, 1), np.float32)
        self.kf.errorCovPost = np.eye(6, dtype=np.float32) * 1000.0
