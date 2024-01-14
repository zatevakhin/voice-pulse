def main():
    from faster_whisper import WhisperModel

    from voice_pulse.config import Config
    from voice_pulse.enums import VadEngine
    from voice_pulse.input_sources import FileInput, MicrophoneInput
    from voice_pulse.listener import Listener

    model = WhisperModel("medium", device="cpu", compute_type="int8")

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

        segments, info = model.transcribe(speech, best_of=2)

        print(f"> {info.language} ({info.language_probability})")
        for seg in segments:
            print("-", seg.text)


if __name__ == "__main__":
    main()
