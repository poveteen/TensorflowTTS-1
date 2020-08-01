# -*- coding: utf-8 -*-
# This code is copy and modify from https://github.com/keithito/tacotron.
"""Perform preprocessing and raw feature extraction."""

import re
import os

import numpy as np
import soundfile as sf

from tensorflow_tts.utils import cleaners

_korean_jaso_code = list(range(0x1100, 0x1113)) + list(range(0x1161, 0x1176)) + list(range(0x11a8, 0x11c3))
_korean_jaso = list(chr(c) for c in _korean_jaso_code)

_pad = "_"
_eos = "~"
_punctuation = " .!?"

# Export all symbols:
symbols = (
    [_pad] + list(_punctuation) + list(_korean_jaso) + [_eos]
)

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}


class LJSpeechProcessor(object):
    """LJSpeech processor."""

    def __init__(self, root_path, cleaner_names):
        self.root_path = root_path
        self.cleaner_names = cleaner_names

        items = []
        self.speaker_name = "ljspeech"
        if root_path is not None:
            with open(os.path.join(root_path, "metadata.csv"), encoding="utf-8") as ttf:
                for line in ttf:
                    parts = line.strip().split("|")
                    wav_path = os.path.join(root_path, "wavs", "%s.wav" % parts[0])
                    text = parts[1]
                    items.append([text, wav_path, self.speaker_name])

            self.items = items

    def get_one_sample(self, idx):
        text, wav_file, speaker_name = self.items[idx]

        # normalize audio signal to be [-1, 1], soundfile already norm.
        audio, rate = sf.read(wav_file)
        audio = audio.astype(np.float32)

        # convert text to ids
        text_ids = np.asarray(self.text_to_sequence(text), np.int32)

        sample = {
            "raw_text": text,
            "text_ids": text_ids,
            "audio": audio,
            "utt_id": self.items[idx][1].split("/")[-1].split(".")[0],
            "speaker_name": speaker_name,
            "rate": rate,
        }

        return sample

    def text_to_sequence(self, text):
        global _symbol_to_id

        sequence = []
        # Check for curly braces and treat their contents as ARPAbet:
        if len(text):
            sequence += _symbols_to_sequence(
                _clean_text(text, [self.cleaner_names])
            )
        return sequence


def _clean_text(text, cleaner_names):
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        if not cleaner:
            raise Exception("Unknown cleaner: %s" % name)
        text = cleaner(text)
    return text


def _symbols_to_sequence(symbols):
    return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]


def _should_keep_symbol(s):
    return s in _symbol_to_id and s != "_" and s != "~"
