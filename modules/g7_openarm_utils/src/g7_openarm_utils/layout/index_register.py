from .control_layout import Joint


class IndexRegister:
    def __init__(self):
        self._data: dict[Joint, int] = {}

    def register(self, joint: Joint, index: int):
        if self._data.get(joint) is not None:
            raise RuntimeError(f"{joint} is registered")
        self._data[joint] = index

    def is_registered(self, joint: Joint):
        return self._data.get(joint) is not None

    def get(self, joint: Joint):
        result = self._data.get(joint)
        
        if result is None:
            raise RuntimeError(f"{joint} is not registered")

        return result

    def is_full(self):
        for joint in Joint:
            if self._data.get(joint) is None:
                return False
        return True
