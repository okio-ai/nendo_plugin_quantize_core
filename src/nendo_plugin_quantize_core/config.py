"""Default settings for the Nendo demucs stemifier."""
from nendo import NendoConfig


class QuantizeConfig(NendoConfig):
    """Configuration defaults for the quantizer plugin.

    This is the configuration class that will be imported into nendo.

    Parameters:
        keep_original_bpm (bool): Whether to keep the original BPM of the track.
    """

    keep_original_bpm: bool = False

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True
