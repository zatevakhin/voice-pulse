from queue import Queue
from typing import Optional

import numpy as np


class SpeechSegmentCollector:
    def __init__(self, threshold: int, samplerate: int) -> None:
        self.queue: Queue = Queue()
        self.threshold: int = threshold
        self.samplerate: int = samplerate

    def add(self, data: np.ndarray) -> None:
        self.queue.put(data)

    def collect(self) -> Optional[np.ndarray]:
        if self.queue.qsize() < self.threshold:
            self.queue = Queue()
            return

        data = np.array([], dtype=np.float16)

        while self.queue.qsize() > 0:
            data = np.append(data, self.queue.get())

        # Appending zeros to the audio data as a workaround for small audio packets
        return np.concatenate([data, np.zeros(self.samplerate + 10)])
