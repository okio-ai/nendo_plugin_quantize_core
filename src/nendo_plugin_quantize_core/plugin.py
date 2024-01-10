"""A nendo core plugin for music quantization."""
import math
from logging import Logger
from typing import List

import librosa
import numpy as np
import pyrubberband as pyrb
from BeatNet.BeatNet import BeatNet

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

    def extract_beat_beatnet(self, filename: str):
        """Extracts beat from a given audio signal and converts beats to frames.

        Args:
            filename: The name of the target file.

        Returns:
            A tuple of (tempo, beat_frames).
        """
        estimator = BeatNet(
            1,
            mode="offline",
            inference_model="DBN",
            plot=[],
            thread=False,
        )
        beat_frames = estimator.process(filename)
        # result is a vector including beat times and downbeat identifier columns
        # respectively with the following shape: numpy_array(num_beats, 2)
        beat_frames = np.transpose(beat_frames)[0]

        # compute bpm
        if len(beat_frames) < 2:
            return 0  # Not enough beats to calculate BPM

        # Calculate intervals between beats
        intervals = [
            beat_frames[i] - beat_frames[i - 1] for i in range(1, len(beat_frames))]

        # Average interval
        avg_interval = sum(intervals) / len(intervals)

        # Convert to BPM (60 seconds per minute)
        tempo = 60 / avg_interval

        return tempo, beat_frames

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
        sr = track.sr
        left_channel = track.signal[0, :] if len(track.signal.shape) > 1 else track.signal
        right_channel = track.signal[1, :] if len(track.signal.shape) > 1 else track.signal

        # Use left channel for beat extraction
        tempo, beat_frames = self.extract_beat_beatnet(track.local())

        # flag determines whether to keep the original bpm
        bpm = tempo if keep_bpm else bpm

        # Generate scaling factor based on original and target tempo
        scale = tempo / bpm

        # Construct time of left channel map using numpy
        time_map = self.construct_time_map(beat_frames, scale, len(left_channel))

        # Time stretch left channel
        streched_left_channel = pyrb.timemap_stretch(left_channel, sr, time_map)

        # Time stretch to original length, rounded to nearest beat
        duration = len(streched_left_channel) / sr
        rounded_duration = math.ceil(duration)
        sequence = self.sequence_generator(rounded_duration)
        nearest_value = min(sequence, key=lambda x: abs(x - rounded_duration))
        length_ratio = len(streched_left_channel) / (nearest_value * sr)

        # Preallocate final_audio for performance
        streched_left_channel = pyrb.time_stretch(streched_left_channel, sr, length_ratio)

        # Construct time of left channel map using numpy
        time_map = self.construct_time_map(beat_frames, scale, len(right_channel))

        # Time stretch right channel
        streched_right_channel = pyrb.timemap_stretch(right_channel, sr, time_map)

        # Time stretch to original length, rounded to nearest beat
        duration = len(streched_right_channel) / sr
        rounded_duration = math.ceil(duration)
        sequence = self.sequence_generator(rounded_duration)
        nearest_value = min(sequence, key=lambda x: abs(x - rounded_duration))
        length_ratio = len(streched_right_channel) / (nearest_value * sr)

        # Preallocate final_audio for performance
        streched_right_channel = pyrb.time_stretch(streched_right_channel, sr, length_ratio)

        streched_signal = np.array([streched_left_channel, streched_right_channel], dtype="float32")

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
