"""A nendo core plugin for music quantization."""
import math
from logging import Logger
from typing import List, Tuple

import essentia.standard as es
import librosa
import numpy as np
import pyrubberband as pyrb

from nendo import Nendo, NendoConfig, NendoGeneratePlugin, NendoTrack

from .config import QuantizeConfig

settings = QuantizeConfig()


class CoreQuantizer(NendoGeneratePlugin):
    """A nendo plugin for music quantization based on rubberband and librosa.

    https://breakfastquay.com/rubberband/

    Examples:
        ```python
        from nendo import Nendo, NendoConfig

        nendo = Nendo(config=NendoConfig(plugins=["nendo_plugin_quantize_core"]))
        track = nendo.library.add_track_from_file(
            file_path="path/to/file.wav",
        )

        quantized = nendo.plugins.quantize_core(
            track=track,
            bpm=120,
        )
        ```
    """

    nendo_instance: Nendo = None
    config: NendoConfig = None
    logger: Logger = None

    def sequence_generator(self, duration: int):
        """Generates a sequence of powers of 2 until the duration.

        Args:
            duration: The maximum duration.

        Yields:
            An int which is the next power of 2 in the sequence.
        """
        power = int(np.log2(duration))
        yield from 2 ** np.arange(power, 0, -1)

    def extract_beat(self, y, sr):
        """Extracts beat from a given audio signal and converts beats to frames.

        Args:
            y: The audio signal.
            sr: The sample rate.

        Returns:
            A tuple of (tempo, beat_frames).
        """
        _, y_percussive = librosa.effects.hpss(y)

        # Beat Track: Simple
        tempo, beats = librosa.beat.beat_track(y=y_percussive, sr=sr, trim=False)

        # Beat track: Complex
        # tempo, beats = librosa.beat.beat_track(
        #     sr=sr, onset_envelope=librosa.onset.onset_strength(y=y_percussive, sr=sr), trim=False)

        beat_frames = librosa.frames_to_samples(beats)

        return tempo, beat_frames

    def extract_beat_essentia(self, y, sr):
        """Extracts beat from a given audio signal and converts beats to frames.

        Args:
            y: The audio signal.
            sr: The sample rate.

        Returns:
            A tuple of (tempo, beat_frames).
        """
        rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
        bpm, beats, beats_confidence, _, _ = rhythm_extractor(y)

        # Convert beats to frames
        # (Essentia's RhythmExtractor2013 returns beats in seconds)
        beat_frames = (beats * sr).astype(int)

        # Ensure unique beat frames
        beat_frames = np.unique(beat_frames)

        # Ensure monotonic beat frames
        beat_frames_diff = np.diff(beat_frames)
        if np.any(beat_frames_diff <= 0):
            beat_frames = beat_frames_diff.cumsum()

        return bpm, beat_frames

    def construct_time_map(
        self,
        beat_frames: np.ndarray,
        scale: float,
        audio_length: int,
    ) -> List[List[int]]:
        """Constructs the time map using Numpy.

        Args:
            beat_frames: Array of beat frames.
            scale: Scaling factor.
            audio_length: Length of the audio in samples.

        Returns:
            The constructed time map.
        """
        beat_frames_scaled = np.round(beat_frames * scale).astype(int)
        time_map = np.column_stack((beat_frames, beat_frames_scaled)).tolist()
        time_map.append([audio_length, int(audio_length * scale)])

        return time_map

    @NendoGeneratePlugin.run_track
    def quantize_audio(
        self,
        track: NendoTrack,
        bpm: int = 120,
        keep_original_bpm: bool = settings.keep_original_bpm,
    ) -> NendoTrack:
        """Run the quantizer plugin.

        Args:
            track (NendoTrack): The track to quantize.
            bpm (int): The BPM to quantize to.
            keep_original_bpm (bool): Whether to keep the original BPM of the track.

        Returns:
            NendoTrack: The quantized track.
        """
        if hasattr(self.config, "keep_original_bpm"):
            keep_bpm = keep_original_bpm or self.config.keep_original_bpm
        else:
            keep_bpm = keep_original_bpm or settings.keep_original_bpm

        # Transpose signal and take the first channel
        signal = track.signal
        sr = track.sr
        signal = signal[0] if len(signal.shape) > 1 else signal

        # Extract beat
        tempo, beat_frames = self.extract_beat_essentia(signal, sr)

        # flag determines whether to keep the original bpm
        bpm = tempo if keep_bpm else bpm

        # Generate scaling factor based on original and target tempo
        scale = tempo / bpm

        # Construct time map using numpy
        time_map = self.construct_time_map(beat_frames, scale, len(signal))

        # Time stretch audio
        streched_signal = pyrb.timemap_stretch(signal, sr, time_map)

        # Time stretch to original length, rounded to nearest beat
        duration = len(streched_signal) / sr
        rounded_duration = math.ceil(duration)
        sequence = self.sequence_generator(rounded_duration)
        nearest_value = min(sequence, key=lambda x: abs(x - rounded_duration))
        length_ratio = len(streched_signal) / (nearest_value * sr)

        # Preallocate final_audio for performance
        streched_signal = pyrb.time_stretch(streched_signal, sr, length_ratio)

        track_title = (
            f"{track.resource.meta['original_filename']} - Quantized"
            if "original_filename" in track.resource.meta
            else "Quantized"
        )

        streched_track = self.nendo_instance.library.add_related_track_from_signal(
            signal=streched_signal,
            sr=int(track.sr),
            related_track_id=track.id,
            track_type="quantized",
            relationship_type="quantized",
            track_meta={
                "title": track_title,
            },
        )

        return streched_track.add_plugin_data(
            plugin_name="nendo_plugin_quantize_core",
            key="tempo",
            value=str(bpm),
        )
