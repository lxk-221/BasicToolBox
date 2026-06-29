from camera.camera import Camera
import pyrealsense2 as rs

class RealSenseCamera(Camera):
    def __init__(self):
        super().__init__()

    def init(self):
        '''
        should return handle, K, dist
        '''
        raise NotImplementedError
    
    def get_frame(self):
        raise NotImplementedError
    
    def release(self):
        raise NotImplementedError