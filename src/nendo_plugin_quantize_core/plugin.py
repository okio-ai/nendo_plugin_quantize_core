"""A nendo core plugin for music quantization."""
from logging import Logger

import librosa
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

    @NendoGeneratePlugin.run_track
    def core_quantize(
        self,
        track: NendoTrack,
        bpm: int = 120,
        keep_original_bpm: bool = settings.keep_original_bpm,
    ):
        """Run the quantizer plugin.

        Args:
            track (NendoTrack): The track to quantize.
            bpm (int): The BPM to quantize to.
            keep_original_bpm (bool): Whether to keep the original BPM of the track.

        Returns:
            NendoTrack: The quantized track.
        """
        y, sr = track.signal, track.sr

        if hasattr(self.config, "keep_original_bpm"):
            keep_bpm = keep_original_bpm or self.config.keep_original_bpm
        else:
            keep_bpm = keep_original_bpm or settings.keep_original_bpm

        y_harmonic, y_percussive = librosa.effects.hpss(y[0] if len(y.shape) > 1 else y)
        onset_strength = librosa.onset.onset_strength(y=y_percussive, sr=sr)

        tempo, beats = librosa.beat.beat_track(
            sr=sr,
            onset_envelope=onset_strength,
            trim=False,
        )
        beat_frames = librosa.frames_to_samples(beats)

        bpm = tempo if keep_bpm else bpm

        # generate metronome
        fixed_beat_times = []
        for i in range(len(beat_frames)):
            fixed_beat_times.append(i * 120 / bpm)
        fixed_beat_frames = librosa.time_to_samples(fixed_beat_times)

        # construct time map
        time_map = []
        for i in range(len(beat_frames)):
            new_member = (beat_frames[i], fixed_beat_frames[i])
            time_map.append(new_member)

        # add ending to time map
        original_length = len(y[0] + 1)
        orig_end_diff = original_length - time_map[i][0]
        new_ending = int(round(time_map[i][1] + orig_end_diff * (tempo / bpm)))
        new_member = (original_length, new_ending)
        time_map.append(new_member)

        strechedaudio = pyrb.timemap_stretch(y.T, sr, time_map)

        track_title = f"{track.resource.meta['original_filename']} - Quantized" if "original_filename" in track.resource.meta else f"Quantized"

        return self.nendo_instance.library.add_related_track_from_signal(
            signal=strechedaudio,
            sr=int(track.sr),
            related_track_id=track.id,
            track_type="quantized",
            relationship_type="quantized",
            track_meta={
                "bpm": int(bpm),
                "title": track_title,
            },
        )
