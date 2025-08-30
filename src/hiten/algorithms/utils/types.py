from enum import IntEnum

from abc import ABC, abstractmethod


class State(ABC):
    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, key: int) -> float:
        pass

    

class SynodicState(IntEnum):
    X=0
    Y=1
    Z=2
    VX=3
    VY=4
    VZ=5

class CenterManifoldState(IntEnum):
    q1=0
    q2=1
    q3=2
    p1=3
    p2=4
    p3=5

class RestrictedCenterManifoldState(IntEnum):
    q2=0
    p2=1
    q3=2
    p3=3