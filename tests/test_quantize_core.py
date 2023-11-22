# -*- encoding: utf-8 -*-
"""Tests for the Nendo core quantizer plugin."""
from nendo import Nendo, NendoConfig

import unittest

nd = Nendo(
    config=NendoConfig(
        library_path="./library",
        log_level="INFO",
        plugins=["nendo_plugin_quantize_core"],
    )
)


class QuantizePluginTests(unittest.TestCase):
    def test_run_plugin_fixed_bpm(self):
        nd.library.reset(force=True)
        track = nd.library.add_track(file_path="tests/assets/test.wav")
        quantized_track = nd.plugins.quantize_core(track=track, bpm=110)

        self.assertTrue(quantized_track.has_relationship_to(track.id))

    def test_run_process_plugin_fixed_bpm(self):
        nd.library.reset(force=True)
        track = nd.library.add_track(file_path="tests/assets/test.wav")
        quantized_track = track.process("nendo_plugin_quantize_core", bpm=110)

        self.assertTrue(quantized_track.has_relationship_to(track.id))

    def test_run_plugin_original_bpm(self):
        nd.library.reset(force=True)
        track = nd.library.add_track(file_path="tests/assets/test.wav")
        quantized_track = nd.plugins.quantize_core(
            track=track, keep_original_bpm=True,
        )
        self.assertTrue(quantized_track.has_relationship_to(track.id))


if __name__ == "__main__":
    unittest.main()
