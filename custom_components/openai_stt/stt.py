from __future__ import annotations
import logging
import os
import wave
import io
import async_timeout
import voluptuous as vol
from collections.abc import AsyncIterable
from homeassistant.components.stt import (
    AudioBitRates,
    AudioChannels,
    AudioCodecs,
    AudioFormats,
    AudioSampleRates,
    Provider,
    SpeechMetadata,
    SpeechResult,
    SpeechResultState,
)

import homeassistant.helpers.config_validation as cv
import whisper

_LOGGER = logging.getLogger(__name__)

CONF_API_KEY = "api_key"
CONF_API_URL = "api_url"
CONF_MODEL = "model"
CONF_PROMPT = "prompt"
CONF_TEMP = "temperature"

DEFAULT_MODEL = "base"  # Use 'base' or another model like 'small', 'medium', or 'large'
DEFAULT_PROMPT = ""
DEFAULT_TEMP = 0

SUPPORTED_MODELS = [
    "base", "small", "medium", "large",  # Adjust for local models
]

SUPPORTED_LANGUAGES = [
    "af", "ar", "hy", "az", "be", "bs", "bg", "ca", "zh", "hr", "cs", "da", "nl",
    "en", "et", "fi", "fr", "gl", "de", "el", "he", "hi", "hu", "is", "id", "it",
    "ja", "kn", "kk", "ko", "lv", "lt", "mk", "ms", "mr", "mi", "ne", "no", "fa", 
    "pl", "pt", "ro", "ru", "sr", "sk", "sl", "es", "sw", "sv", "tl", "ta", "th", 
    "tr", "uk", "ur", "vi", "cy",
]

MODEL_SCHEMA = vol.In(SUPPORTED_MODELS)

PLATFORM_SCHEMA = cv.PLATFORM_SCHEMA.extend({
    vol.Optional(CONF_MODEL, default=DEFAULT_MODEL): cv.string,
    vol.Optional(CONF_PROMPT, default=DEFAULT_PROMPT): cv.string,
    vol.Optional(CONF_TEMP, default=DEFAULT_TEMP): cv.positive_int,
})


async def async_get_engine(hass, config, discovery_info=None):
    """Set up the OpenAI STT component."""
    model = config.get(CONF_MODEL, DEFAULT_MODEL)
    prompt = config.get(CONF_PROMPT, DEFAULT_PROMPT)
    temperature = config.get(CONF_TEMP, DEFAULT_TEMP)
    return LocalWhisperSTTProvider(hass, model, prompt, temperature)


class LocalWhisperSTTProvider(Provider):
    """The Local Whisper STT provider."""

    def __init__(self, hass, model, prompt, temperature) -> None:
        """Init Local Whisper STT service."""
        self.hass = hass
        self.name = "Local Whisper STT"

        # Load the Whisper model locally
        self._model = whisper.load_model(model)
        self._prompt = prompt
        self._temperature = temperature

    @property
    def supported_languages(self) -> list[str]:
        """Return a list of supported languages."""
        return SUPPORTED_LANGUAGES

    @property
    def supported_formats(self) -> list[AudioFormats]:
        """Return a list of supported formats."""
        return [AudioFormats.WAV, AudioFormats.OGG]

    @property
    def supported_codecs(self) -> list[AudioCodecs]:
        """Return a list of supported codecs."""
        return [AudioCodecs.PCM, AudioCodecs.OPUS]

    @property
    def supported_bit_rates(self) -> list[AudioBitRates]:
        """Return a list of supported bitrates."""
        return [AudioBitRates.BITRATE_16]

    @property
    def supported_sample_rates(self) -> list[AudioSampleRates]:
        """Return a list of supported samplerates."""
        return [AudioSampleRates.SAMPLERATE_16000]

    @property
    def supported_channels(self) -> list[AudioChannels]:
        """Return a list of supported channels."""
        return [AudioChannels.CHANNEL_MONO]

    async def async_process_audio_stream(
        self, metadata: SpeechMetadata, stream: AsyncIterable[bytes]
    ) -> SpeechResult:
        # Collect data
        audio_data = b""
        async for chunk in stream:
            audio_data += chunk

        # Convert audio data to the correct format
        wav_stream = io.BytesIO()

        with wave.open(wav_stream, 'wb') as wf:
            wf.setnchannels(metadata.channel)
            wf.setsampwidth(metadata.bit_rate // 8)
            wf.setframerate(metadata.sample_rate)
            wf.writeframes(audio_data)

        # Use Whisper model to transcribe audio
        wav_stream.seek(0)  # Reset stream to the beginning for Whisper
        audio = whisper.load_audio(wav_stream)
        result = self._model.transcribe(audio, language=metadata.language)

        if result['text']:
            return SpeechResult(result['text'], SpeechResultState.SUCCESS)
        else:
            return SpeechResult("", SpeechResultState.ERROR)
