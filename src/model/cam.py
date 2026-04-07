import cv2
import threading
import time
import queue

class Camera:
    def __init__(self, index=0, width=640, height=480, fps=60):
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        
        self.frame_queue = queue.Queue(maxsize=8)
        self.running = False
        self.thread = None
        self.latest_frame = None
        self.lock = threading.Lock()

        self.open_camera()
        self.start_capture_thread()

    def open_camera(self):
        print(f"[Camera] 正在打开摄像头 /dev/video{self.index} ...")
        
        # 强制使用 V4L2 后端
        self.cam = cv2.VideoCapture(self.index, cv2.CAP_V4L2)
        
        # 严格顺序强制设置
        self.cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cam.set(cv2.CAP_PROP_FPS, self.fps)
        self.cam.set(cv2.CAP_PROP_BUFFERSIZE, 4)   # 增加缓冲区

        # 打印实际参数
        w = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f = self.cam.get(cv2.CAP_PROP_FPS)
        
        print(f"[Camera] 实际分辨率: {w} x {h}")
        print(f"[Camera] 实际FPS设置: {f:.1f} (目标: {self.fps})")

    def capture_thread(self):
        """独立线程持续读取帧"""
        while self.running:
            ret, frame = self.cam.read()
            if ret and frame is not None:
                with self.lock:
                    self.latest_frame = frame.copy()
                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    pass
            else:
                time.sleep(0.005)

    def start_capture_thread(self):
        self.running = True
        self.thread = threading.Thread(target=self.capture_thread, daemon=True)
        self.thread.start()
        print("[Camera] 捕获线程已启动")

    def read(self):
        """主线程获取最新帧"""
        with self.lock:
            if self.latest_frame is not None:
                return True, self.latest_frame.copy()
        return False, None

    def release(self):
        """正确释放资源"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if hasattr(self, 'cam') and self.cam.isOpened():
            self.cam.release()
        print("[Camera] 摄像头已释放")