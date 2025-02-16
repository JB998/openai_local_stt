import logging
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
import aiohttp

_LOGGER = logging.getLogger(__name__)

DEFAULT_API_URL = "http://127.0.0.1:8000/transcribe"  # Local API URL
DEFAULT_MODEL = "base"  # The model you want to run locally
DEFAULT_PROMPT = ""
DEFAULT_TEMP = 0

SUPPORTED_LANGUAGES = [
    "en", "es", "fr", "de", "zh",  # Add other languages as needed
]

PLATFORM_SCHEMA = cv.PLATFORM_SCHEMA.extend({
    vol.Optional("api_key"): cv.string,
    vol.Optional("model", default=DEFAULT_MODEL): cv.string,
    vol.Optional("prompt", default=DEFAULT_PROMPT): cv.string,
    vol.Optional("temperature", default=DEFAULT_TEMP): cv.positive_int,
})


async def async_get_engine(hass, config, discovery_info=None):
    """Set up the local Whisper STT component."""
    api_url = config.get("api_key", DEFAULT_API_URL)
    model = config.get("model", DEFAULT_MODEL)
    prompt = config.get("prompt", DEFAULT_PROMPT)
    temperature = config.get("temperature", DEFAULT_TEMP)
    temperature = config.get("temperature", DEFAULT_TEMP)
    return LocalWhisperSTTProvider(hass, model, prompt, temperature, api_url)


class LocalWhisperSTTProvider(Provider):
    """The Local Whisper STT provider."""

    def __init__(self, hass, model, prompt, temperature, api_url) -> None:
        """Init Local Whisper STT service."""
        self.hass = hass
        self.name = "Local Whisper STT"

        # Set up the local API URL and parameters
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
        # Collect data from the audio stream
        audio_data = b""
        async for chunk in stream:
            audio_data += chunk

        # Convert audio data to a BytesIO stream (simulating a file)
        wav_stream = io.BytesIO(audio_data)

        # Prepare the file for the API call
        file = ("audio.wav", wav_stream, "audio/wav")

        # Call the local API to transcribe the audio
        async with aiohttp.ClientSession() as session:
            async with session.post(self._api_url, data={"file": file}) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get("text"):
                        return SpeechResult(result["text"], SpeechResultState.SUCCESS)
                _LOGGER.error("Failed to transcribe audio.")
                return SpeechResult("", SpeechResultState.ERROR)
