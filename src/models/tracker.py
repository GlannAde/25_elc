import math
import time

# 枚举库
from enum import IntEnum

import numpy as np

# 引入 3D 卡尔曼滤波器
from Kalman import KalmanFilter3D


class Status(IntEnum):
    LOST = 0
    TMP_LOST = 2
    TRACK = 3


class Tracker:
    def __init__(
        self,
        img_width=640,
        img_height=480,
        f_pixel_h=725.6,
        real_height=17.5,
        use_kf=True,
    ):
        self.img_width = img_width
        self.img_height = img_height

        self.f_pixel_h = f_pixel_h
        self.real_height = real_height

        self.ref_point = np.array([-0.15, 1.35, 0.0])
        self.yaw_bias = -2.3
        self.pitch_bias = -0.2

        self.status = Status.LOST
        self.lost_count = 0
        self.frame_lost_tol = 8
        self.last_time = None
        self.laser_pos = None

        self.onfire = False
        self.fire_deadzone = 1.5

        self.use_kf = use_kf
        if self.use_kf:
            # 3D 滤波器的初始化
            self.kf = KalmanFilter3D(
                q_scale=0.35, r_xy_scale=0.1, r_dist_scale=2.0, default_dt=1 / 120.0
            )

    def time_diff(self):
        current_time = time.time_ns()
        if self.last_time is None:
            self.last_time = current_time
            return 1 / 120.0
        diff = (current_time - self.last_time) / 1e9
        self.last_time = current_time
        return min(diff, 0.1)

    def get_dist(self, board):
        pts = board.points
        h_left = math.sqrt((pts[0][0] - pts[1][0]) ** 2 + (pts[0][1] - pts[1][1]) ** 2)
        h_right = math.sqrt((pts[3][0] - pts[2][0]) ** 2 + (pts[3][1] - pts[2][1]) ** 2)
        avg_h_px = (h_left + h_right) / 2.0

        if avg_h_px <= 1e-3:
            return 1000.0

        dist = (self.real_height * self.f_pixel_h) / avg_h_px
        return dist

    def filter_and_predict(self, target):
        dt = self.time_diff()

        if not self.use_kf:
            if target and target.center:
                self.status = Status.TRACK
                return target.center[0], target.center[1], self.get_dist(target)
            else:
                self.status = Status.LOST
                return self.img_width / 2, self.img_height / 2, 0.0

        if target and target.center:
            # 捕捉到目标
            if self.status == Status.LOST:
                self.kf.reset()
            self.status = Status.TRACK
            self.lost_count = 0

            # 先预测时间步长，再用真实值更新
            self.kf.predict(dt=dt)
            return self.kf.update(
                target.center[0], target.center[1], self.get_dist(target)
            )

        else:
            # 目标丢失
            self.lost_count += 1
            if self.lost_count <= self.frame_lost_tol:
                self.status = Status.TMP_LOST
                # 仅靠惯性预测
                return self.kf.predict(dt=dt)
            else:
                self.status = Status.LOST
                self.kf.reset()
                return self.img_width / 2, self.img_height / 2, 0.0

    def solve(self, cx, cy, dist):
        if dist <= 0:
            dist = 0.1

        offset_x = cx - self.img_width / 2
        offset_y = cy - self.img_height / 2

        dx = offset_x + (self.ref_point[0] * self.f_pixel_h / dist)
        dy = offset_y + (self.ref_point[1] * self.f_pixel_h / dist)

        self.laser_pos = (int(self.img_width / 2 + dx), int(self.img_height / 2 + dy))

        yaw = -math.degrees(math.atan2(dx, self.f_pixel_h))
        pitch = math.degrees(math.atan2(dy, self.f_pixel_h))

        yaw += self.yaw_bias
        pitch += self.pitch_bias

        return yaw, pitch, dist

    def track(self, board):
        filtered_cx, filtered_cy, filtered_dist = self.filter_and_predict(board)

        if self.status != Status.LOST:
            yaw_err, pitch_err, dist = self.solve(
                filtered_cx, filtered_cy, filtered_dist
            )

            # --- 动态开火决策 ---
            # 只有当视觉完全捕获目标 (TRACK) 且角度误差都在允许死区内时，才进行击发。
            # 如果是 TMP_LOST (预测状态)，为了安全起见，停止击发。
            if (
                self.status == Status.TRACK
                and abs(yaw_err) < self.fire_deadzone
                and abs(pitch_err) < self.fire_deadzone
            ):
                self.onfire = True
            else:
                self.onfire = False

            # --- [核心新增] 提取 3D 滤波后的 XY 坐标，用于画面丝滑显示 ---
            smooth_center = (int(filtered_cx), int(filtered_cy))

            return yaw_err, pitch_err, dist, self.status, self.laser_pos, smooth_center
        else:
            self.laser_pos = None
            self.onfire = False  # 完全丢失目标时，强制断开激光
            return 0.0, 0.0, 0.0, self.status, None, None
