import time

import cv2
import numpy as np


class Board:
    def __init__(self):
        self.points = []  # 四个端点坐标
        self.center = None  # 对角线交点坐标 (x, y)
        self.area = 0.0


class Detector:
    def __init__(self, min_area=3000, max_area=500000, use_otsu=True):
        self.board_min_area = min_area
        self.board_max_area = max_area
        self.use_otsu = use_otsu
        self.manual_threshold = 127  # 如果关闭 OTSU，则使用此手动阈值
        self.boards = []
        self.raw = None
        self.binary = None

        # [优化引入] 预留标准的正方形坐标，用于透视变换
        self.std_square = np.float32([[0, 0], [0, 100], [100, 100], [100, 0]])

    def process_image(self, frame):
        self.raw = frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # [优化引入] 性能开关：算力不够时可随时关闭 OTSU 换取高帧率
        if self.use_otsu:
            _, binary = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
            )
        else:
            _, binary = cv2.threshold(
                gray, self.manual_threshold, 255, cv2.THRESH_BINARY_INV
            )

        self.binary = binary
        return binary

    def find_board(self, binary):
        """核心：保留第二套强大的 RETR_CCOMP 拓扑查找逻辑"""
        boards = []
        contours, hierarchy = cv2.findContours(
            binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )

        if hierarchy is None:
            self.boards = []
            return []

        # 优先寻内轮廓，没有则寻外轮廓
        inner_contours = [
            (i, c) for i, c in enumerate(contours) if hierarchy[0][i][3] != -1
        ]
        target_contours = (
            inner_contours
            if inner_contours
            else [(i, c) for i, c in enumerate(contours) if hierarchy[0][i][3] == -1]
        )

        for i, contour in target_contours:
            area = cv2.contourArea(contour)
            if self.board_min_area < area < self.board_max_area:
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

                if len(approx) == 4:
                    points = approx.reshape(4, 2)

                    sum_xy = points.sum(axis=1)
                    diff_xy = points[:, 0] - points[:, 1]
                    sorted_points = [
                        points[np.argmin(sum_xy)],  # 左上
                        points[np.argmax(diff_xy)],  # 左下
                        points[np.argmax(sum_xy)],  # 右下
                        points[np.argmin(diff_xy)],  # 右上
                    ]

                    if len(set(tuple(pt) for pt in sorted_points)) < 4:
                        sorted_points = [
                            points[np.argmin(points[:, 0])],
                            points[np.argmin(points[:, 1])],
                            points[np.argmax(points[:, 0])],
                            points[np.argmax(points[:, 1])],
                        ]

                    board = Board()
                    board.points = [tuple(map(int, pt)) for pt in sorted_points]
                    board.area = area
                    board.center = self._calculate_intersection(board.points)

                    if board.center is not None:
                        boards.append(board)

        if boards:
            boards.sort(key=lambda b: b.area, reverse=True)
            self.boards = boards
        else:
            self.boards = []

        return boards

    def get_perspective_offset(self, board, target_ratio_x=0.5, target_ratio_y=0.5):
        """
        [优化引入] 第一套的透视变换精华
        不再仅仅局限于打中心点！
        例如：传入 target_ratio_x=0.2, target_ratio_y=0.2 就可以在倾斜靶纸上精准定位到左上角 20% 处。
        """
        if len(board.points) != 4:
            return board.center

        dst_pts = np.float32(board.points)
        M = cv2.getPerspectiveTransform(self.std_square, dst_pts)

        # 目标点在 100x100 标准正方形里的坐标
        target_pt = np.float32([[[100 * target_ratio_x, 100 * target_ratio_y]]])

        # 映射回现实畸变的画面中
        real_pt = cv2.perspectiveTransform(target_pt, M)
        return (int(real_pt[0][0][0]), int(real_pt[0][0][1]))

    def _calculate_intersection(self, points):
        """两点式求交点 (保留原版)"""
        x1, y1 = points[0]
        x2, y2 = points[2]
        x3, y3 = points[1]
        x4, y4 = points[3]
        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denominator == 0:
            return None
        px = (
            (x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)
        ) / denominator
        py = (
            (x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)
        ) / denominator
        return (int(px), int(py))

    def draw(self, image):
        if not self.boards or image is None:
            return image

        board = self.boards[0]
        if not board.points or board.center is None:
            return image

        # [优化引入] 使用 polylines 提升 4 倍绘线效率
        pts = np.array(board.points, np.int32)
        cv2.polylines(image, [pts], True, (0, 255, 0), 2)

        # 蓝色画出对角线
        cv2.line(image, board.points[0], board.points[2], (255, 0, 0), 2)
        cv2.line(image, board.points[1], board.points[3], (255, 0, 0), 2)

        # 绿色画出交点
        cv2.circle(image, board.center, 5, (0, 255, 0), -1)

        # 画出相机光轴中心
        h, w = image.shape[:2]
        cv2.circle(image, (w // 2, h // 2), 5, (0, 165, 255), -1)

        return image

    def detect(self, frame, debug=False):
        """对外接口，加入了耗时探针"""
        start = time.time()

        bin_img = self.process_image(frame)
        t_process = time.time()

        boards = self.find_board(bin_img)
        t_find = time.time()

        if debug:
            print(
                f"Vision Cost - OTSU/Bin: {(t_process - start) * 1000:.1f}ms | Find Board: {(t_find - t_process) * 1000:.1f}ms"
            )

        return boards[0] if boards else None

    def display(self, dis):
        if self.raw is None:
            return None, self.binary
        vis = self.raw.copy()
        if dis == 1:
            res = self.draw(vis)
            return res, self.binary
        return vis, self.binary
