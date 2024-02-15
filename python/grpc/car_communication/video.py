import numpy as np
import cv2


def convert_bytes_to_frame(frame_bytes:bytes) -> np.ndarray | None:
    try:
        frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        return frame
    except:
        return None

class VideoPlayer:
    _frames = []

    @classmethod
    def add_frame(cls,frame: np.ndarray) -> None:
        cls._frames.append(frame)

    @classmethod
    def add_image(cls,image:bytes) -> None:
        frame = convert_bytes_to_frame(image)
        if not(frame): return
        cls.add_frame(frame)
    
    @classmethod
    def display_video(cls) -> None:
        while True:
            if not(cls._frames): continue
            frame = cls._frames.pop(0)
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        cv2.destroyAllWindows()
