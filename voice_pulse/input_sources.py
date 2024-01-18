import os
from queue import Queue
from typing import Optional, Protocol, Union

import numpy as np
import sounddevice as sd
import soundfile as sf


class InputSource(Protocol):
    def start(self) -> None:
        ...

    def read(self) -> Optional[np.ndarray]:
        ...

    def close(self) -> None:
        ...


class MicrophoneInput:
    def __init__(
        self,
        device: Union[int, str, None],
        blocksize: int,
        samplerate: int,
        channels: int,
    ) -> None:
        self.stream = sd.InputStream(
            callback=self._callback,
            device=device,
            samplerate=samplerate,
            blocksize=blocksize,
            channels=channels,
        )
        self.buffer = Queue()

    def _callback(self, indata: np.ndarray, frames: int, time, status) -> None:
        self.buffer.put(indata.flatten())

    def start(self) -> None:
        self.stream.start()

    def read(self) -> Optional[np.ndarray]:
        return self.buffer.get()

    def close(self) -> None:
        self.stream.close(ignore_errors=True)


class FileInput:
    def __init__(self, file_path: str, blocksize: int, channels: bool) -> None:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File '{file_path}' is not found.")

        # TODO: Convert audio to same samplerate which used in all pipeline.
        self.file = sf.SoundFile(
            file_path,
        )
        self.blocksize = blocksize

    def start(self) -> None:
        # For a file, no action is needed on start
        pass

    def read(self) -> Optional[np.ndarray]:
        data = self.file.read(frames=self.blocksize)

        # data: np.ndarray = np.mean(data, axis=1)
        data = np.array(data, dtype=np.float32)

        if len(data) != self.blocksize:
            raise StopIteration()

        return data if len(data) > 0 else None

    def close(self) -> None:
        self.file.close()


class CallbackInput:
    def __init__(self, blocksize: int) -> None:
        self.blocksize = blocksize
        self.buffer = Queue()

    def start(self) -> None:
        pass

    def read(self) -> Optional[np.ndarray]:
        return self.buffer.get()

    def close(self) -> None:
        pass

    def receive_chunk(self, data: np.ndarray[np.float32]) -> None:
        self.buffer.put(data)
