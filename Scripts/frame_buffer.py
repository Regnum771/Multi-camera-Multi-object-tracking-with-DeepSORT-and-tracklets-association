import threading


class FrameBuffer:
    """
    Thread-safe latest-frame buffer per camera.
    """

    def __init__(self):
        self._frames = {}
        self._lock = threading.Lock()

    def update(self, camera_name, frame):
        with self._lock:
            self._frames[camera_name] = frame

    def get(self, camera_name):
        with self._lock:
            return self._frames.get(camera_name)
