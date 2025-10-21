import collections
import math
import struct
import wave
from pathlib import Path

import numpy as np


def text_to_sustained_melody(
        text: str,
        sample_rate: int = 44100,
        base_duration: float = 0.4,  # seconds for the *least* frequent character
        out_path: str | Path = "text_melody.wav",
) -> None:
    """
    Convert a string into a WAV file where each distinct character becomes a tone.

    The pitch is proportional to how often that character appears in `text`,
    and the sustain time scales inversely (rare chars last longer).

    Parameters
    ----------
    text : str
        Source text. Non‑ASCII characters are ignored.
    sample_rate : int, optional
        Samples per second (default: 44100).
    base_duration : float, optional
        Duration (in seconds) of the tone for the *most* frequent character.
    out_path : str | Path, optional
        Destination WAV file path.

    Notes
    -----
    The mapping is:
        freq   = f_base + (freq_of_char / total_chars) * Δf
        dur    = base_duration * (max_freq / freq_of_char)
    where `Δf` is 400 Hz, giving a nice spread from ~200 Hz to ~600 Hz.
    """
    # ------------------------------------------------------------------
    # 1. Pre‑process: keep only printable ASCII for stability
    # ------------------------------------------------------------------
    clean_text = "".join(ch for ch in text if 32 <= ord(ch) < 127)
    if not clean_text:
        raise ValueError("Text contains no printable ASCII characters.")

    # ------------------------------------------------------------------
    # 2. Count character frequencies
    # ------------------------------------------------------------------
    counter = collections.Counter(clean_text)
    total_chars = sum(counter.values())
    max_freq = max(counter.values())

    f_base = 220.0  # A3 – pleasant starting note
    delta_f = 400.0  # Frequency span

    # ------------------------------------------------------------------
    # 3. Build a mono waveform by concatenating tones per character
    # ------------------------------------------------------------------
    frames: list[float] = []

    for ch, freq_of_ch in counter.items():
        # Pitch based on *relative* frequency
        pitch = f_base + (freq_of_ch / total_chars) * delta_f

        # Duration inversely proportional to frequency
        dur = base_duration * (max_freq / freq_of_ch)

        t = np.linspace(0, dur, int(sample_rate * dur), False)

        # Simple ADSR envelope: attack 10 %, decay 20 %, sustain 60 %, release 10 %
        a = int(0.1 * len(t))
        d = int(0.2 * len(t))
        s = int(0.6 * len(t))
        r = len(t) - (a + d + s)

        envelope = np.concatenate([
            np.linspace(0, 1, a),  # attack
            np.linspace(1, 0.7, d),  # decay
            np.full(s, 0.7),  # sustain
            np.linspace(0.7, 0, r)  # release
        ])

        tone = envelope * 0.5 * np.sin(2 * math.pi * pitch * t)
        frames.extend(tone.tolist())

    # ------------------------------------------------------------------
    # 4. Convert to signed 16‑bit PCM and write WAV
    # ------------------------------------------------------------------
    pcm_data = struct.pack(
        "<" + "h" * len(frames),
        *[int(max(-1, min(1, x)) * 32767) for x in frames]
    )

    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16‑bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)

    print(f"Generated {out_path} ({len(frames) / sample_rate:.2f}s long).")


# ----------------------------------------------------------------------
# Demo – feel free to replace the string with anything you like!
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = """
    The quick brown fox jumps over the lazy dog.
    Every good boy does fine.
    """
    text_to_sustained_melody(sample_text, out_path="frequency_based.wav")
