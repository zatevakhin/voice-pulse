import logging
from queue import Queue
from datetime import datetime
from typing import Union, Iterator

import numpy as np
import onnxruntime as ort

from .config import Config
from .input_sources import InputSource
from .speech_filters import create_vad_speech_filter
from .speech_processing import SpeechSegmentCollector
from .speech_segment import SpeechSegment

ort.set_default_logger_severity(3)

logger = logging.getLogger(__name__)


SpeechData = Union[np.ndarray, SpeechSegment]


class Listener:
    def __init__(self, config: Config, stream: InputSource) -> None:
        self._silence_counter: int = 0
        self._config: Config = config

        self._speech_filter = create_vad_speech_filter(
            self._config.vad_engine,
            self._config.vad_aggressiveness,
            self._config.samplerate,
        )

        self._collector = SpeechSegmentCollector(
            threshold=self._config.collect_threshold,
            samplerate=self._config.samplerate,
        )

        self._stream = stream
        self._speech_data: Queue[SpeechData] = Queue()

    def _apply_speech_filter(self, indata: np.ndarray) -> bool:
        return self._speech_filter(indata)

    def _handle_speech(self, indata: np.ndarray) -> None:
        self._collector.add(indata)
        self._silence_counter = 0

    def _handle_silence(self) -> None:
        if self._silence_counter >= self._config.silence_threshold:
            self._process_collected_data()
        else:
            self._silence_counter += 1

    def _process_collected_data(self) -> None:
        collected_data = self._collector.collect()
        if collected_data is not None:
            self._speech_data.put(collected_data)
            self._silence_counter = 0

    def __del__(self):
        self._stream.close()

    def __iter__(self) -> Iterator[SpeechData]:
        self._stream.start()

        try:
            while True:
                indata = self._stream.read()

                if indata is None:
                    continue

                if self._apply_speech_filter(indata):
                    self._handle_speech(indata)
                else:
                    self._handle_silence()

                if not self._speech_data.empty():
                    yield self._speech_data.get()

        except (GeneratorExit, StopIteration):
            self._stream.close()


class ListenerStamped(Listener):
    def __init__(self, config: Config, stream: InputSource) -> None:
        super().__init__(config, stream)

    def _process_collected_data(self) -> None:
        collected_data = self._collector.collect()
        if collected_data is not None:
            speech_segment = SpeechSegment(speech=collected_data, timestamp=datetime.now())
            self._speech_data.put(speech_segment)
            self._silence_counter = 0
