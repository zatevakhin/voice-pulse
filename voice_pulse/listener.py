import logging
from queue import Queue

import numpy as np
import onnxruntime as ort

from .config import Config
from .input_sources import InputSource
from .speech_filters import create_vad_speech_filter
from .speech_processing import SpeechSegmentCollector

ort.set_default_logger_severity(3)

logger = logging.getLogger(__name__)


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
        self._speech_data: Queue = Queue()

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

    def __iter__(self):
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


def main():
    from faster_whisper import WhisperModel

    from .enums import VadEngine
    from .input_sources import FileInput, MicrophoneInput

    model = WhisperModel("tiny", device="cpu", compute_type="int8")

    config = Config(
        vad_engine=VadEngine.SILERIO,
        block_duration=32,
    )

    # stream = FileInput(
    #     file_path="my.wav",
    #     blocksize=config.blocksize,
    #     channels=config.channels,
    # )

    stream = MicrophoneInput(config.device, config.blocksize, config.samplerate, config.channels)

    for speech in Listener(config, stream):
        # print(f"speech ({type(speech)})", len(speech))

        segments, info = model.transcribe(speech)

        print(f"> {info.language} ({info.language_probability})")
        for seg in segments:
            print("-", seg.text)


if __name__ == "__main__":
    main()
