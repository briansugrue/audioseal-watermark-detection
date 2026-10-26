import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"   

import torch
import torchaudio
import soundfile as sf
from audioseal import AudioSeal
import pandas as pd
import subprocess
import tempfile

import random
from collections import defaultdict
from pathlib import Path

from sklearn.metrics import roc_auc_score
import numpy as np


pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

# SETTINGS -----------------------------------------------------------------
DATA_DIR = Path("C:/LibriSpeech/test-clean")
SR = 16000          # sample rate
SEGMENT_S = 5       # seconds per clip
N_CLIPS = 100
PER_SPEAKER = 6     # candidate files per speaker (some are too short and get skipped)


# LOAD MODEL ---------------------------------------------------------------
# Model name corresponds to the YAML card file name found in audioseal/cards
model = AudioSeal.load_generator("audioseal_wm_16bits")
model.eval()
detector = AudioSeal.load_detector("audioseal_detector_16bits")
detector.eval()


# FIND FILES ---------------------------------------------------------------
files = sorted(DATA_DIR.rglob("*.flac"))
print(len(files))   # should be 2620

by_speaker = defaultdict(list)
for f in files:
    by_speaker[f.parts[-3]].append(f)
print(len(by_speaker))   # should be 40

random.seed(0)
torch.manual_seed(0)

selected = []
for speaker_files in by_speaker.values():
    selected += random.sample(speaker_files, min(PER_SPEAKER, len(speaker_files)))
random.shuffle(selected)
print(len(selected))   # about 240


# ATTACK FUNCTIONS ---------------------------------------------------------
def codec_roundtrip(x, codec, ext, bitrate):
    """x has shape (1, 1, T) at SR. Encode with ffmpeg, decode, return the same shape."""
    T = x.shape[-1]
    with tempfile.TemporaryDirectory() as d:
        src, enc, out = f"{d}/in.wav", f"{d}/enc.{ext}", f"{d}/out.wav"
        sf.write(src, x.squeeze().clamp(-1, 1).numpy(), SR)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src,
                        "-c:a", codec, "-b:a", bitrate, enc], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", enc,
                        "-ar", str(SR), "-ac", "1", out], check=True)
        y, _ = sf.read(out, dtype="float32")
    y = torch.from_numpy(y).reshape(1, 1, -1)
    if y.shape[-1] < T:
        y = torch.nn.functional.pad(y, (0, T - y.shape[-1]))
    return y[..., :T]


def add_noise(x, snr_db):
    noise = torch.randn_like(x)
    gain = x.pow(2).mean().sqrt() / (noise.pow(2).mean().sqrt() * 10 ** (snr_db / 20))
    return x + gain * noise


attacks = {
    "none":           lambda x: x,
    "noise_30dB":     lambda x: add_noise(x, 30),
    "noise_25dB":     lambda x: add_noise(x, 25),
    "noise_20dB":     lambda x: add_noise(x, 20),
    "noise_15dB":     lambda x: add_noise(x, 15),
    "noise_10dB":     lambda x: add_noise(x, 10),
    "noise_0dB":      lambda x: add_noise(x, 0),
    "lowpass_4000Hz": lambda x: torchaudio.functional.lowpass_biquad(x, SR, 4000),
    "resample_8k":    lambda x: torchaudio.functional.resample(
                          torchaudio.functional.resample(x, SR, 8000), 8000, SR),
    "mp3_64k":        lambda x: codec_roundtrip(x, "libmp3lame", "mp3", "64k"),
    "mp3_32k":        lambda x: codec_roundtrip(x, "libmp3lame", "mp3", "32k"),
    "opus_24k":       lambda x: codec_roundtrip(x, "libopus", "ogg", "24k"),
    "opus_12k":       lambda x: codec_roundtrip(x, "libopus", "ogg", "12k"),
}


# DETECTION HELPER ---------------------------------------------------------
def detect(x):
    with torch.no_grad():
        prob, bits = detector.detect_watermark(x, SR)
        frame_out, _ = detector(x, SR)
    return prob.item(), frame_out[:, 1, :].mean().item(), bits


# RUN A CLIP ---------------------------------------------------------------
# Audio tensors have shape (batch, channels, samples). The audio must be at
# the sample rate the model expects (16 kHz here).

def run_clip(path):
    # Load and prepare the audio
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(data.T)            # (samples, channels) -> (channels, samples)
    wav = wav.mean(dim=0, keepdim=True)       # mono

    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)

    seg = SEGMENT_S * SR
    if wav.shape[-1] < seg:
        return None                           # file too short, skip it
    start = random.randint(0, wav.shape[-1] - seg)
    wav = wav[:, start:start + seg]
    wav = wav.unsqueeze(0)

    # A random 16-bit message to embed in the watermark
    msg = torch.randint(0, 2, (wav.shape[0], model.msg_processor.nbits))

    with torch.no_grad():
        # Embed
        watermark = model.get_watermark(wav, SR, message=msg)
        watermarked_audio = wav + watermark

    # Detect under each attack, on watermarked and on clean audio
    results = []
    for name, attack in attacks.items():
        wm_prob, wm_score, bits = detect(attack(watermarked_audio))
        clean_prob, clean_score, _ = detect(attack(wav))
        bit_acc = (bits == msg).float().mean().item()
        results.append(dict(attack=name, wm_prob=wm_prob, wm_score=wm_score,
                            clean_prob=clean_prob, clean_score=clean_score, bit_acc=bit_acc))
    return results


# MAIN ---------------------------------------------------------------------
rows = []
n_valid = 0
for f in selected:
    out = run_clip(f)
    if out is None:
        continue
    for r in out:                                   # one dict per attack
        rows.append(dict(speaker=f.parts[-3], file=f.name, **r))
    n_valid += 1
    if n_valid == N_CLIPS:
        break

df = pd.DataFrame(rows)
print(len(df), df.speaker.nunique())                # rows = clips x attacks


# SUMMARY ------------------------------------------------------------------
summary = df.groupby("attack", sort=False).agg(
    detect_rate=("wm_prob", lambda s: (s > 0.5).mean()),
    false_alarm_rate=("clean_prob", lambda s: (s > 0.5).mean()),
    mean_wm_score=("wm_score", "mean"),
    mean_clean_score=("clean_score", "mean"),
    bit_acc=("bit_acc", "mean"),
)

auc = {}
for name, g in df.groupby("attack", sort=False):
    y = np.r_[np.ones(len(g)), np.zeros(len(g))]
    s = np.r_[g.wm_score.values, g.clean_score.values]
    auc[name] = roc_auc_score(y, s)
summary["auc"] = pd.Series(auc)
print(summary.round(3))

df.to_csv("100_clip_test_results.csv", index=False)
summary.round(3).to_csv("summary.csv")