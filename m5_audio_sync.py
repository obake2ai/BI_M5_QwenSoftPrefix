import os, math, subprocess, textwrap, shlex
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Sequence

import numpy as np
import soundfile as sf
from scipy import signal

# from IPython.display import Audio, display

WORKDIR = Path("work")
WORKDIR.mkdir(exist_ok=True, parents=True)

def sh(cmd: Union[str, Sequence[str]], check: bool=True) -> subprocess.CompletedProcess:
    """コマンド実行ヘルパ（失敗時に ffmpeg のエラー本文を必ず表示）

    - cmd が str のとき: shell=True（従来互換。クオートは自前で含める）
    - cmd が list/tuple のとき: shell=False（推奨：クオート事故が起きない）

    NOTE:
      以前の版では stdout/stderr をパイプに吸って例外だけが出るため、
      何が原因で ffmpeg が落ちたのか分かりにくい問題がありました。
      この sh() は失敗時に必ず p.stdout を表示してから例外を投げます。
    """
    if isinstance(cmd, (list, tuple)):
        printable = " ".join(shlex.quote(str(x)) for x in cmd)
        shell = False
    else:
        printable = cmd
        shell = True

    print(f"$ {printable}")
    p = subprocess.run(
        cmd,
        shell=shell,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    if check and p.returncode != 0:
        # ここで ffmpeg のエラーメッセージが見えるようにする
        print(p.stdout)
        raise subprocess.CalledProcessError(p.returncode, printable, output=p.stdout)
    return p

# def play(path: str, autoplay: bool=False):
#     """Jupyter上で音を再生"""
#     path = str(path)
#     if not Path(path).exists():
#         raise FileNotFoundError(path)
#     display(Audio(path, autoplay=autoplay))

def ffprobe_duration_sec(path: str) -> float:
    out = sh([
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]).stdout.strip()
    return float(out) if out else 0.0

def to_wav_16k_mono(in_path: str, out_path: str):
    """扱いやすい 16kHz mono wav に正規化"""
    out_path = str(out_path)
    sh([
        "ffmpeg", "-y",
        "-hide_banner", "-loglevel", "error",
        "-i", str(in_path),
        "-ac", "1", "-ar", "16000",
        "-vn",
        out_path
    ])

def ffmpeg_apply_filter(in_wav: str, out_wav: str, afilter: str):
    """ffmpegの -af でエフェクト適用"""
    sh([
        "ffmpeg", "-y",
        "-hide_banner", "-loglevel", "error",
        "-i", str(in_wav),
        "-ac", "1", "-ar", "16000",
        "-af", afilter,
        str(out_wav)
    ])

def list_ffmpeg_filters() -> str:
    return sh(["ffmpeg", "-hide_banner", "-filters"]).stdout

FFMPEG_FILTERS_TEXT = list_ffmpeg_filters()

def ffmpeg_has_filter(filter_name: str) -> bool:
    # 行頭にフィルタ名が出るので空白境界で見る（false positive を避けるため少し厳しめ）
    return (f" {filter_name} " in FFMPEG_FILTERS_TEXT) or (f"\t{filter_name} " in FFMPEG_FILTERS_TEXT)

print("ffmpeg has rubberband:", ffmpeg_has_filter("rubberband"))
print("ffmpeg has asubboost :", ffmpeg_has_filter("asubboost"))

def load16k(path: str) -> np.ndarray:
    y, sr = sf.read(path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != 16000:
        tmp = str(WORKDIR / "_tmp_16k.wav")
        to_wav_16k_mono(path, tmp)
        y, sr = sf.read(tmp)
        if y.ndim > 1:
            y = y.mean(axis=1)
    y = y.astype(np.float32)
    # DC除去
    y = y - float(np.mean(y))
    return y

def write16k(path: str, y: np.ndarray):
    y = np.nan_to_num(y.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    sf.write(str(path), y, 16000)

def peak_norm(y: np.ndarray, peak: float = 0.95) -> np.ndarray:
    y = np.nan_to_num(y.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    m = float(np.max(np.abs(y)) + 1e-9)
    if m < 1e-8:
        return y
    return (peak / m) * y

def rms(y: np.ndarray) -> float:
    return float(np.sqrt(np.mean(y*y) + 1e-12))

def audio_report(path: str):
    y = load16k(path)
    dur = len(y) / 16000.0
    nan_count = int(np.isnan(y).sum())
    inf_count = int(np.isinf(y).sum())
    print(f"{path}")
    print(f"  duration: {dur:.3f}s  peak: {np.max(np.abs(y)):.4f}  rms: {rms(y):.4f}")
    if nan_count or inf_count:
        print(f"  !!! contains NaN/Inf: NaN={nan_count}, Inf={inf_count}")

def atempo_chain(rate: float) -> str:
    """ffmpeg atempo は 0.5〜2.0 制限があるので分解する"""
    if rate <= 0:
        raise ValueError("rate must be > 0")
    parts = []
    r = float(rate)
    while r > 2.0 + 1e-9:
        parts.append(2.0); r /= 2.0
    while r < 0.5 - 1e-9:
        parts.append(0.5); r /= 0.5
    if abs(r - 1.0) > 1e-6:
        parts.append(r)
    if not parts:
        parts = [1.0]
    return ",".join([f"atempo={p:.6f}" for p in parts])

def pitch_shift_ffmpeg_16k(in_wav_16k: str,
                           out_wav_16k: str,
                           semitone_steps: float,
                           method: str = "auto"):
    """16k mono wav をピッチシフト（テンポ維持）

    - method="auto": rubberband を“まず試して”、失敗したら asetrate に自動fallback。
      （RunPod等で ffmpeg が rubberband を“表示はするが実行で落ちる”環境に備える）
    """
    sr = 16000
    ratio = 2 ** (semitone_steps / 12.0)  # pitch scale

    if method == "auto":
        methods = []
        if ffmpeg_has_filter("rubberband"):
            methods.append("rubberband")
        methods.append("asetrate")  # 最後は必ずこれを試す
    else:
        methods = [method]

    last_err = None
    for m in methods:
        if m == "rubberband":
            af = f"rubberband=pitch={ratio:.6f}:tempo=1"
        elif m == "asetrate":
            factor = ratio
            atempo = atempo_chain(1.0 / factor)
            af = ",".join([
                f"asetrate={sr*factor:.3f}",
                atempo,
                f"aresample={sr}",
            ])
        else:
            raise ValueError(f"Unknown method: {m}")

        try:
            sh([
                "ffmpeg", "-y",
                "-hide_banner", "-loglevel", "error",
                "-i", str(in_wav_16k),
                "-ac", "1", "-ar", "16000",
                "-af", af,
                str(out_wav_16k)
            ])
            return
        except subprocess.CalledProcessError as e:
            last_err = e
            print(f"[pitch_shift] method '{m}' failed -> trying fallback ...")
            continue

    # 全部ダメなら最後のエラーを投げる（stdout が上に出ているはず）
    if last_err is not None:
        raise last_err
    raise RuntimeError("pitch_shift_ffmpeg_16k failed unexpectedly")

def envelope_follower(x: np.ndarray,
                      sr: int = 16000,
                      attack_ms: float = 5.0,
                      release_ms: float = 120.0,
                      power: float = 1.25) -> np.ndarray:
    """NaNを出さないエンベロープ抽出（0..1）。IIR+冪乗でも負値が出ないようにする。"""
    x = np.abs(np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0))
    # 1-pole smoothing（attack/release）
    a_a = math.exp(-1.0 / (max(1.0, attack_ms) * 0.001 * sr))
    a_r = math.exp(-1.0 / (max(1.0, release_ms) * 0.001 * sr))
    env = np.zeros_like(x, dtype=np.float32)
    prev = 0.0
    for i, v in enumerate(x):
        a = a_a if v > prev else a_r
        prev = a * prev + (1.0 - a) * float(v)
        env[i] = prev

    # 正規化（0..1）→ clip → 冪乗
    env = env / (float(np.max(env)) + 1e-9)
    env = np.clip(env, 0.0, 1.0)
    env = np.power(env, float(power)).astype(np.float32)
    return env

def butter_filter(y: np.ndarray, sr: int, btype: str, cutoff, order: int = 4) -> np.ndarray:
    b, a = signal.butter(order, cutoff, btype=btype, fs=sr)
    return signal.lfilter(b, a, y).astype(np.float32)

def tanh_drive(y: np.ndarray, drive: float) -> np.ndarray:
    if drive <= 0:
        return y.astype(np.float32)
    k = 1.0 + float(drive) * 5.0
    return np.tanh(k * y).astype(np.float32)

def make_rumble_noise(voice: np.ndarray,
                      sr: int,
                      base_hz: float = 55.0,
                      amount: float = 0.25,
                      seed: int = 0) -> np.ndarray:
    """低域ノイズ（地鳴り）を生成し、エンベロープで揺らす"""
    if amount <= 0:
        return np.zeros_like(voice, dtype=np.float32)

    rng = np.random.default_rng(seed)
    n = len(voice)
    t = np.arange(n) / sr

    env = envelope_follower(voice, sr=sr, attack_ms=8, release_ms=220, power=1.35)

    # ブラウン寄りノイズ（低域が自然）
    white = rng.standard_normal(n).astype(np.float32)
    brown = np.cumsum(white).astype(np.float32)
    brown = brown / (np.max(np.abs(brown)) + 1e-9)

    # 複数帯域を合成して“ハム”化を避ける
    def band(low, high):
        return butter_filter(brown, sr, "bandpass", [low, high], order=4)

    b1 = band(max(20.0, base_hz*0.45), min(650.0, base_hz*2.2))
    b2 = band(max(20.0, base_hz*0.90), min(650.0, base_hz*3.6))
    low_wide = butter_filter(brown, sr, "lowpass", min(220.0, base_hz*3.0), order=4)

    rum = (0.55*low_wide + 0.30*b1 + 0.15*b2).astype(np.float32)

    # “うねり”を足す（ゆっくりしたAM）
    lfo_f = 0.25 + 0.35 * rng.random()
    lfo = (0.65 + 0.35*np.sin(2*np.pi*lfo_f*t + 2*np.pi*rng.random())).astype(np.float32)

    rum = rum * env * lfo

    # 量調整：RMSで声に対して相対量を決める
    target = rms(voice) * (0.9 * amount)
    rum = rum * (target / (rms(rum) + 1e-9))
    return rum.astype(np.float32)

def rumble_layered(in_wav_16k: str,
                   out_wav_16k: str,
                   pitch_steps: float = -6.0,
                   sub_oct_mix: float = 0.55,
                   rumble_mix: float = 0.25,
                   rumble_base_hz: float = 55.0,
                   drive: float = 0.55,
                   xover_hz: float = 280.0,
                   seed: int = 42):
    sr = 16000
    dry = load16k(in_wav_16k)

    # ピッチシフト（main/sub）をffmpegで作ってからPythonで合成
    main_ps = str(WORKDIR / "_r2_main.wav")
    sub_ps  = str(WORKDIR / "_r2_sub.wav")
    pitch_shift_ffmpeg_16k(in_wav_16k, main_ps, pitch_steps, method="auto")
    pitch_shift_ffmpeg_16k(in_wav_16k, sub_ps,  pitch_steps - 12.0, method="auto")

    main = load16k(main_ps)
    sub  = load16k(sub_ps)

    n = min(len(dry), len(main), len(sub))
    if n < sr * 0.2:
        raise RuntimeError(f"audio too short after pitch shift: n={n}")
    dry, main, sub = dry[:n], main[:n], sub[:n]

    # 低域バス（地鳴り側）
    low_main = butter_filter(main, sr, "lowpass", 420, order=4)
    sub_lp_hz = float(min(700.0, max(220.0, rumble_base_hz * 6.0)))
    low_sub  = butter_filter(sub,  sr, "lowpass", sub_lp_hz, order=4)

    # サブ層が鳴りっぱなしになりにくいようにゲート（NaNが出ないエンベロープ）
    gate = envelope_follower(main, sr=sr, attack_ms=6, release_ms=180, power=1.05)
    low_sub = low_sub * gate

    noise = make_rumble_noise(main, sr, base_hz=rumble_base_hz, amount=rumble_mix, seed=seed)

    low_bus = low_main + float(sub_oct_mix)*low_sub + noise
    low_bus = tanh_drive(low_bus, drive)

    # クロスオーバー：中高域はmain、低域はlow_bus
    high_voice = butter_filter(main, sr, "highpass", xover_hz, order=4)
    low_rumble = butter_filter(low_bus, sr, "lowpass",  xover_hz, order=4)

    mix = high_voice + low_rumble
    mix = mix - float(np.mean(mix))
    mix = peak_norm(mix, 0.95)

    write16k(out_wav_16k, mix)


def rumble_layered_with_fx(in_wav_16k: str,
                           out_wav_16k: str,
                           **kwargs):
    tmp = str(WORKDIR / "_tmp_r3_base.wav")
    rumble_layered(in_wav_16k, tmp, **kwargs)

    # 仕上げ：軽い残響 + EQ + コンプレッション
    af = ",".join([
        "aecho=0.8:0.85:120|240:0.25|0.18",
        "equalizer=f=140:t=q:w=1.1:g=3",
        "acompressor=threshold=0.18:ratio=4:attack=15:release=260:makeup=1.5",
        "alimiter=limit=0.97",
    ])
    ffmpeg_apply_filter(tmp, out_wav_16k, af)
