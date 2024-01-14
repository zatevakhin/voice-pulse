from typing import Literal, Union

from pydantic import BaseModel, Field, computed_field, model_validator

from .enums import VadAggressiveness, VadEngine


class Config(BaseModel):
    vad_aggressiveness: VadAggressiveness = Field(VadAggressiveness.LOW)
    vad_engine: VadEngine = Field(VadEngine.WEBRTC)

    device: Union[int, str, None] = Field(None)
    channels: Literal[1, 2] = Field(1)
    samplerate: Literal[8000, 16000, 32000] = Field(16000)
    block_duration: Literal[10, 20, 30, 32] = Field(30)

    collect_threshold: int = Field(8)
    silence_threshold: int = Field(8)

    @computed_field
    @property
    def blocksize(self) -> int:
        return int(self.samplerate * self.block_duration / 1000)

    @model_validator(mode="after")
    def is_block_duration_supported(self) -> "Config":
        if self.block_duration in [10, 20, 30] and self.vad_engine is not VadEngine.WEBRTC:
            raise ValueError("Vad Engine WebRTC supports block durations 10, 20, and 30 ms")

        if self.block_duration in [32] and self.vad_engine is not VadEngine.SILERIO:
            raise ValueError("Vad Engine Silerio supports block duration only 32 ms")

        return self
