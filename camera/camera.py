
class Camera:
    def __init__(self):
        pass

    def init(self):
        '''
        should return handle, K, dist
        '''
        raise NotImplementedError
    
    def get_frame(self):
        raise NotImplementedError

    def release(self):
        raise NotImplementedError