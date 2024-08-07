from datetime import datetime

import numpy as np
from pydantic import BaseModel


class SpeechSegment(BaseModel):
    speech: np.ndarray
    timestamp: datetime

    class Config:
        arbitrary_types_allowed = True

    def to_bytes(self):
        return self.speech.tobytes()

    @classmethod
    def from_bytes(cls, data: bytes, timestamp: datetime = None):
        speech = np.frombuffer(data, dtype=np.float32)
        return cls(speech=speech, timestamp=timestamp or datetime.now())
