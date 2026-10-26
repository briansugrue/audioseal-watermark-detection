# AudioSeal watermark detection: a small robustness test

A short evaluation of how well the detector in Meta's [AudioSeal](https://github.com/facebookresearch/audioseal) (16-bit model) holds up when watermarked speech is degraded by noise, filtering, resampling and lossy compression. I wrote it to get hands-on experience with audio watermarking and detector evaluation.

This is a small test of an existing model. It does not propose a new method.

## Setup

- **Data:** 100 clips from _test-clean.tar.gz_ (https://www.openslr.org/12), taken from 38 different speakers. Each clip is a random 5-second window, converted to 16 kHz mono.
- **Model:** AudioSeal `audioseal_wm_16bits` (generator) and `audioseal_detector_16bits` (detector), used as released, with no training or changes.
- **Watermarking:** each clip gets a random 16-bit message embedded.
- **Controls:** every attack is applied to both the watermarked clip and the original clean clip, so false alarms are measured under the same processing.
- **Randomness:** seed 0. Running the script a second time gave the same table.

### Attacks

| Attack | Description |
|---|---|
| `noise_XdB` | White Gaussian noise added at X dB signal-to-noise ratio, relative to the whole watermarked signal (30, 25, 20, 15, 10, 0 dB) |
| `lowpass_4000Hz` | Low-pass biquad filter at 4 kHz |
| `resample_8k` | Resampled 16 kHz → 8 kHz → 16 kHz |
| `mp3_64k`, `mp3_32k` | MP3 encode and decode with ffmpeg (libmp3lame) at 64 and 32 kbps |
| `opus_24k`, `opus_12k` | Opus encode and decode with ffmpeg (libopus, Ogg container) at 24 and 12 kbps |

### Metrics

- **detect_rate:** share of watermarked clips the detector flags (clip-level probability > 0.5).
- **false_alarm_rate:** the same threshold applied to the clean clips.
- **mean_wm_score / mean_clean_score:** mean frame-level watermark probability from the detector, averaged over clips.
- **bit_acc:** fraction of the 16 embedded bits decoded correctly (0.5 is guessing).
- **auc:** how well the mean frame-level score separates watermarked from clean clips, regardless of threshold.

## Results (100 clips per row)

| Attack | detect_rate | false_alarm_rate | mean_wm_score | mean_clean_score | bit_acc | auc |
|---|---|---|---|---|---|---|
| none | 1.00 | 0.00 | 1.000 | 0.005 | 1.000 | 1.000 |
| noise_30dB | 1.00 | 0.00 | 0.924 | 0.005 | 0.999 | 1.000 |
| noise_25dB | 0.94 | 0.00 | 0.763 | 0.006 | 0.998 | 1.000 |
| noise_20dB | 0.55 | 0.00 | 0.541 | 0.005 | 0.979 | 1.000 |
| noise_15dB | 0.07 | 0.00 | 0.356 | 0.005 | 0.931 | 1.000 |
| noise_10dB | 0.02 | 0.00 | 0.227 | 0.005 | 0.841 | 1.000 |
| noise_0dB | 0.00 | 0.00 | 0.043 | 0.008 | 0.588 | 0.899 |
| lowpass_4000Hz | 1.00 | 0.00 | 1.000 | 0.006 | 1.000 | 1.000 |
| resample_8k | 1.00 | 0.00 | 1.000 | 0.005 | 1.000 | 1.000 |
| mp3_64k | 1.00 | 0.00 | 1.000 | 0.005 | 1.000 | 1.000 |
| mp3_32k | 1.00 | 0.00 | 0.997 | 0.004 | 1.000 | 1.000 |
| opus_24k | 1.00 | 0.00 | 0.999 | 0.005 | 1.000 | 1.000 |
| opus_12k | 0.99 | 0.00 | 0.846 | 0.005 | 0.996 | 1.000 |

AUC values are rounded to three decimals, so 1.000 means "at or very near 1", not necessarily exactly 1. The full per-clip results are in `results.csv`.

## Findings

- **Filtering and compression had little effect.** Low-pass filtering, 8 kHz resampling, MP3 (64 and 32 kbps) and Opus at 24 kbps left detection at 100% and bit accuracy at about 1.0. Even Opus at 12 kbps kept 99% detection, though the mean score dropped to 0.846.
- **Additive white noise is what degrades detection.** At the default 0.5 threshold, detection falls from 94% at 25 dB SNR to 55% at 20 dB and to 7% or less at 15 dB and below.
- **The score still separates the two groups under noise.** Down to 10 dB SNR the AUC stays at about 1.0 even though the detection rate at 0.5 has collapsed, so the failure at those levels is mostly the fixed threshold, not a lack of signal. At 0 dB the AUC falls to 0.899. Note that AUC assumes a threshold chosen with knowledge of the attack, which a real detector would not have.
- **The decoded message degrades more gradually than detection.** At 20 dB, bit accuracy is still 0.979 while only 55% of clips are flagged. I observed this but did not investigate why.
- **No false alarms** on clean audio in any condition (0 of 100 per condition). With 100 clips, that supports a false-alarm rate of a few percent at most, not exactly zero.

## Limitations

- Tested one model (AudioSeal 16-bit), on 100 five-second clips of read English speech. Results may differ for music, other languages, other durations or other models.
- Only white noise was tested as an additive attack. No mixing with other speech or music, reverberation, speed or pitch changes, or cropping.
- No localization test (finding a watermarked segment inside longer audio) and no adversarial or resynthesis attacks.
- A single run with one random seed. Clips from the same speaker are not fully independent, and no confidence intervals are reported.
- Codec round-trips may shift the audio by a few samples. I did not check alignment, though the detector scores the whole clip, so this is unlikely to matter at the clip level.

## Reproducing

Requirements:

- Python 3.11.9 or newer
- [ffmpeg] (https://ffmpeg.org/) available on your PATH, for the MP3 and Opus attacks (tested with 9.0.2, the Gyan.dev full build for Windows)
- Python packages: `audioseal`, `torch`, `torchaudio`, `soundfile`, `pandas`, `scikit-learn`, `numpy` (tested with PyTorch 2.14.1, CPU build)

Install the packages with:

```
pip install torch torchaudio audioseal soundfile pandas scikit-learn numpy
```

1. Download _test-clean.tar.gz_ from https://www.openslr.org/12 and extract it.
2. Set `DATA_DIR` at the top of `main.py` to the extracted `test-clean` folder.
3. Run `python main.py`. It prints the summary table and writes `results.csv` and `summary.csv`.

The script sets `TORCHDYNAMO_DISABLE=1` because PyTorch's compile step needed a C++ compiler on my Windows machine and is not needed here. It ran on CPU.

## Credits

Uses [AudioSeal](https://github.com/facebookresearch/audioseal) by Meta (San Roman et al., 2024) and the [LibriSpeech](https://www.openslr.org/12) corpus (Panayotov et al., 2015). Check each project's licence and citation requirements before reusing their code or data.
