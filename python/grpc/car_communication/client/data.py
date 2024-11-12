from dataclasses import dataclass
import numpy as np

from units import *


@dataclass
class CarParameters:
    _data = ["steer", "forward", "backward"]

    steer: float
    forward: float
    backward: float

    def to_list(self) -> list[float]:
        data = [float(self.__getattribute__(i)) for i in self._data]
        return data

    def __repr__(self) -> str:
        d = []
        for name in self._data:
            value = self.__getattribute__(name)
            d.append(f"- {name}: {value}")
        return "\n".join(d)


@dataclass
class DistanceSensorData:
    direction: str
    distance: Meter

    def __repr__(self) -> str:
        d = f"DistanceSensor[{self.direction}]: {self.distance} meter"
        return d


@dataclass
class RouterData:
    id: str
    rssi: Rssi

    def __repr__(self) -> str:
        d = f"Router[{self.id}]: {self.rssi} dBm"
        return d


class CameraImage:
    def __init__(self, frame: np.ndarray) -> None:
        self.frame = frame

    @property
    def bytes_count(self) -> int:
        return len(self.frame.tobytes())

    def __repr__(self) -> str:
        d = f"CameraImage: {self.bytes_count} bytes"
        return d


@dataclass
class GrpcClientData:
    car_id: str
    car_speed: float
    car_parameters: CarParameters
    camera_image: CameraImage | None
    distance_sensors: list[DistanceSensorData]
    routers: list[RouterData]
    boxes_in_camera_view: bool
    car_collision_data: bool
    qr_code_metadata: str

    @property
    def _sorted_routers(self) -> list[RouterData]:
        key = lambda x: x[0]
        routers = [i[1] for i in sorted([(r.id, r) for r in self.routers], key=key)]
        return routers

    def __repr__(self) -> str:
        total = "== GrpcClientData ==\n"
        total += f"CarId: {self.car_id}\n"
        total += f"CarSpeed: {self.car_speed}\n"
        total += f"CarParameters: \n{self.car_parameters}\n"
        total += f"CameraImage: {self.camera_image}\n"
        total += "DistanceSensors:\n"
        for i in self.distance_sensors:
            total += f"- {i}\n"
        total += "Routers:\n"
        for i in self._sorted_routers:
            total += f"- {i}\n"
        total += f"BoxesInView: {self.boxes_in_camera_view}\n"
        total += f"CarCollision: {self.car_collision_data}\n"
        total += f"QrCodeMetadata: {self.qr_code_metadata}\n"
        return total
