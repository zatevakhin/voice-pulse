import sounddevice as sd
from pydantic import Field, BaseModel, computed_field
from typing import Any, Union, Literal, Optional, Protocol, Generator
import numpy as np
import logging
from queue import Queue
from enum import Enum, auto
import onnxruntime as ort

ort.set_default_logger_severity(3)

logger = logging.getLogger(__name__)


class VadAggressiveness(Enum):
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    HIGHEST = auto()


class VadEngine(Enum):
    WEBRTC = auto()
    SILERIO = auto()


class InputStreamConfig(BaseModel):
    device: Union[int, str, None] = Field(None)
    channels: Literal[1, 2] = Field(1)
    samplerate: Literal[8000, 16000, 32000] = Field(16000)
    # This property should not be included in the model
    # dump because InputStream does not have a parameter with this name
    block_duration: Literal[10, 20, 30, 32] = Field(30, exclude=True)

    @computed_field
    @property
    def blocksize(self) -> int:
        return int(self.samplerate * self.block_duration / 1000)


class ListenerConfig(BaseModel):
    vad_aggresivenes: VadAggressiveness = Field(VadAggressiveness.LOW)
    vad_engine: VadEngine = Field(VadEngine.WEBRTC)
    silence_threshold: int = Field(8)
    collect_threshold: int = Field(8)
    stream_config: InputStreamConfig = InputStreamConfig()


class SpeechFilter(Protocol):
    def __call__(self, data: np.ndarray) -> bool:
        ...

    def is_block_duration_supported(self, duration: int) -> bool:
        ...


class SpeechFilterWebrtcVad:
    def __init__(self, vad_aggressiveness: int, samplerate: int) -> None:
        import webrtcvad

        mode = (
            {
                VadAggressiveness.LOW: 0,
                VadAggressiveness.MEDIUM: 1,
                VadAggressiveness.HIGH: 2,
                VadAggressiveness.HIGHEST: 3,
            }
        ).get(vad_aggressiveness)

        self.vad = webrtcvad.Vad(mode=mode)
        self.samplerate = samplerate

    def __call__(self, data: np.ndarray) -> bool:
        # Firstly normalize from [-1,+1] to [0,1]
        # Convert data to bytes to be digestible by VAD is_speech function
        data = ((data + 1) / 2).astype(np.float16).tobytes()
        return self.vad.is_speech(data, self.samplerate)

    def is_block_duration_supported(self, duration: int) -> bool:
        return duration in [10, 20, 30]


class SpeechFilterSilerioVad:
    def __init__(self, vad_aggressiveness: VadAggressiveness, samplerate: int) -> None:
        import torch

        torch.set_num_threads(1)

        self.tensor_from_numpy = torch.from_numpy
        self.samplerate: int = samplerate

        self.model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=True,
        )

        self.threshold = (
            {
                VadAggressiveness.LOW: 0.75,
                VadAggressiveness.MEDIUM: 0.85,
                VadAggressiveness.HIGH: 0.90,
                VadAggressiveness.HIGHEST: 0.95,
            }
        ).get(vad_aggressiveness)

    def __call__(self, data: np.ndarray) -> bool:
        probability = self.model(self.tensor_from_numpy(data), self.samplerate).item()

        return probability >= self.threshold

    def is_block_duration_supported(self, duration: int) -> bool:
        return duration in [32]


class SpeachSegmentCollector:
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


class Listener:
    def __init__(self, config: ListenerConfig) -> None:
        self._silence_counter: int = 0
        self.config: ListenerConfig = config

        self.speech_filter = None
        if config.vad_engine == VadEngine.WEBRTC:
            self.speech_filter = SpeechFilterWebrtcVad(
                config.vad_aggresivenes, self.config.stream_config.samplerate
            )
        elif config.vad_engine == VadEngine.SILERIO:
            self.speech_filter = SpeechFilterSilerioVad(
                config.vad_aggresivenes, self.config.stream_config.samplerate
            )
        else:
            raise ValueError(
                f"Support for '{config.vad_engine}' VAD Engine not implemented."
            )

        if not self.speech_filter.is_block_duration_supported(
            self.config.stream_config.block_duration
        ):
            raise ValueError(
                f"Block duration '{self.config.stream_config.block_duration}' for '{self.speech_filter.__class__.__name__}'."
            )

        self.collector = SpeachSegmentCollector(
            threshold=config.collect_threshold,
            samplerate=self.config.stream_config.samplerate,
        )
        self.stream = sd.InputStream(
            callback=self, **self.config.stream_config.model_dump()
        )
        self.speech_data: Queue = Queue()

    def __call__(self, indata: np.ndarray, frames: int, time, status) -> Any:
        indata = indata.flatten()

        if self.apply_speech_filter(indata):
            self.handle_speech(indata)
        else:
            self.handle_silence()

    def apply_speech_filter(self, indata: np.ndarray) -> bool:
        return self.speech_filter(indata)

    def handle_speech(self, indata: np.ndarray) -> None:
        self.collector.add(indata)
        self._silence_counter = 0

    def handle_silence(self) -> None:
        if self._silence_counter >= self.config.silence_threshold:
            self.process_collected_data()
        else:
            self._silence_counter += 1

    def process_collected_data(self) -> None:
        collected_data = self.collector.collect()
        if collected_data is not None:
            self.speech_data.put(collected_data)
            self._silence_counter = 0

    def __del__(self):
        self.stream.close(ignore_errors=True)

    def __iter__(self):
        if self.stream.active:
            logger.warn("Stream was already activated by creation of another iterator.")
        else:
            self.stream.start()

        try:
            while True:
                yield self.speech_data.get()
        except GeneratorExit:
            self.stream.close(ignore_errors=True)


def main():
    config = ListenerConfig(
        vad_engine=VadEngine.SILERIO,
        stream_config=InputStreamConfig(
            block_duration=32,
        ),
    )

    for speech in Listener(config):
        print(f"speech ({type(speech)})", len(speech))


if __name__ == "__main__":
    main()
