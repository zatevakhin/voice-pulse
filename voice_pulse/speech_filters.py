from typing import Protocol

import numpy as np

from .enums import VadAggressiveness, VadEngine


class SpeechFilter(Protocol):
    def __call__(self, data: np.ndarray) -> bool:
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
            trust_repo=True,
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


def create_vad_speech_filter(
    vad_engine: VadEngine, vad_aggressivenes: VadAggressiveness, samplerate: int
) -> SpeechFilter:
    if VadEngine.SILERIO == vad_engine:
        return SpeechFilterSilerioVad(vad_aggressivenes, samplerate)
    elif VadEngine.WEBRTC == vad_engine:
        return SpeechFilterWebrtcVad(vad_aggressivenes, samplerate)
    else:
        raise NotImplementedError(f"No implementation for {vad_engine} VAD engine.")
