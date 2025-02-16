from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterable
import wave
import io
import httpx  # To make HTTP requests to the local FastAPI endpoint

import async_timeout
import voluptuous as vol
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

_LOGGER = logging.getLogger(__name__)

CONF_API_URL = "api_url"
CONF_MODEL = "model"
CONF_PROMPT = "prompt"
CONF_TEMP = "temperature"

DEFAULT_API_URL = "http://localhost:8000/transcribe"  # Local FastAPI endpoint
DEFAULT_MODEL = "whisper-1"
DEFAULT_PROMPT = ""
DEFAULT_TEMP = 0

SUPPORTED_MODELS = [
    "whisper-1",
]

SUPPORTED_LANGUAGES = [
    "en",  # Add other languages if needed
]

MODEL_SCHEMA = vol.In(SUPPORTED_MODELS)

PLATFORM_SCHEMA = cv.PLATFORM_SCHEMA.extend({
    vol.Optional(CONF_API_URL, default=DEFAULT_API_URL): cv.string,
    vol.Optional(CONF_MODEL, default=DEFAULT_MODEL): cv.string,
    vol.Optional(CONF_PROMPT, default=DEFAULT_PROMPT): cv.string,
    vol.Optional(CONF_TEMP, default=DEFAULT_TEMP): cv.positive_int,
})


async def async_get_engine(hass, config, discovery_info=None):
    """Set up the local STT component."""
    api_url = config.get(CONF_API_URL, DEFAULT_API_URL)
    model = config.get(CONF_MODEL, DEFAULT_MODEL)
    prompt = config.get(CONF_PROMPT, DEFAULT_PROMPT)
    temperature = config.get(CONF_TEMP, DEFAULT_TEMP)
    return LocalSTTProvider(hass, api_url, model, prompt, temperature)


class LocalSTTProvider(Provider):
    """The local STT provider that uses the FastAPI Whisper endpoint."""

    def __init__(self, hass, api_url, model, prompt, temperature) -> None:
        """Init local STT service."""
        self.hass = hass
        self.name = "Local STT"

        self._api_url = api_url
        self._model = model
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
        """Process the incoming audio stream."""
        
        # Collect data from stream
        audio_data = b""
        async for chunk in stream:
            audio_data += chunk

        # Prepare audio file to send to the FastAPI server
        wav_stream = io.BytesIO()
        with wave.open(wav_stream, 'wb') as wf:
            wf.setnchannels(metadata.channel)
            wf.setsampwidth(metadata.bit_rate // 8)
            wf.setframerate(metadata.sample_rate)
            wf.writeframes(audio_data)

        wav_stream.seek(0)
        files = {"audio": ("audio.wav", wav_stream, "audio/wav")}
        
        # Send audio to the FastAPI server for transcription
        async with httpx.AsyncClient() as client:
            try:
                _LOGGER.debug("Log Test")
                response = await client.post(self._api_url, files=files)
                _LOGGER.debug(response.text)
                response.raise_for_status()  # Will raise an exception for 4xx/5xx errors
                transcription = response.json()  # Assuming JSON response from FastAPI
                _LOGGER.debug(transcription)
                # Return transcription result
                if 'text' in transcription:
                    return SpeechResult(
                        transcription['text'],
                        SpeechResultState.SUCCESS,
                    )

            except httpx.HTTPStatusError as e:
                _LOGGER.exception("Error during transcription:")
            except Exception as e:
                _LOGGER.exception("Unexpected error:")
        return SpeechResult("", SpeechResultState.ERROR)
