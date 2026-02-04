#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple Module LLM (JSONL over TCP: newline-delimited JSON) -> (optional) OpenAI-compatible TTS -> ffmpeg -> tinyplay

Changes (2026-02):
- Start TTS/playback BEFORE LLM cleanup by default to reduce "text->audio" latency
- Add --llm (alias of --preset) to choose qwen/tinyswallow
- Auto-select softprefix hidden size (H) by preset when --softprefix-h is not given:
    qwen: 896, tinyswallow: 1536
- Add timing logs for TTS/ffmpeg/playback
- Optional: --cleanup-before-tts to keep old behavior (cleanup first)
"""

from __future__ import annotations

import argparse
import base64
import json
import socket
import struct
import subprocess
import sys
import time
import urllib.request
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple
import random

from m5_audio_sync import *


# -----------------------------
# Presets (model + softprefix H)
# -----------------------------

LLM_PRESETS: Dict[str, Dict[str, Any]] = {
    "qwen": {
        "model": "qwen2.5-0.5B-prefill-20e",
        "softprefix_h": 896,
    },
    "tinyswallow": {
        "model": "TinySwallow-1.5B-Instruct-ax630c",
        "softprefix_h": 1536,
    },
}


# -----------------------------
# SoftPrefix helpers (bf16 b64)
# -----------------------------

def f32_to_bf16_u16(x: float) -> int:
    """float32 -> bf16(truncate) -> u16"""
    u32 = struct.unpack("<I", struct.pack("<f", float(x)))[0]
    return (u32 >> 16) & 0xFFFF


def make_soft_prefix_b64_constant(P: int, H: int, val: float) -> str:
    """base64 of bf16 little-endian u16 repeated P*H times."""
    u16 = f32_to_bf16_u16(val)
    raw = struct.pack("<H", u16) * (P * H)
    return base64.b64encode(raw).decode("ascii")


def make_soft_prefix_b64_random_scaled( #植物センサーの入力
    P: int,
    H: int,
    val: float,
    *,
    seed: int | None = None,
    dist: str = "uniform",      # "uniform" | "normal"
    low: float = -1.0,          # dist="uniform" の範囲
    high: float = 1.0,
    mean: float = 0.0,          # dist="normal" の平均
    std: float = 1.0,           # dist="normal" の標準偏差
) -> str:
    """
    base64 of bf16 little-endian u16 values (length P*H).
    Generate random float32 r[i], compute x[i] = r[i] * val, then bf16(truncate).
    """
    n = P * H
    rng = random.Random(seed)

    raw = bytearray(2 * n)
    off = 0

    if dist == "uniform":
        for _ in range(n):
            r = rng.random() * (high - low) + low  # [low, high)
            x = r * float(val)
            u16 = f32_to_bf16_u16(x)
            raw[off:off + 2] = struct.pack("<H", u16)
            off += 2
    elif dist == "normal":
        for _ in range(n):
            r = rng.gauss(mean, std)
            x = r * float(val)
            u16 = f32_to_bf16_u16(x)
            raw[off:off + 2] = struct.pack("<H", u16)
            off += 2
    else:
        raise ValueError("dist must be 'uniform' or 'normal'")

    return base64.b64encode(raw).decode("ascii")


# -----------------------------
# JSONL socket (no makefile)
# -----------------------------

class JSONLSocket:
    """
    Newline-delimited JSON over TCP.
    Implemented with recv buffer (no socket.makefile) to avoid readline/timeout quirks.
    """
    def __init__(
        self,
        host: str,
        port: int,
        connect_timeout_sec: float = 5.0,
        io_timeout_sec: float = 1.0,
        max_line_bytes: int = 2 * 1024 * 1024,  # 2MB safety
        debug_skip: bool = False,
    ) -> None:
        self.host = host
        self.port = port
        self.connect_timeout_sec = connect_timeout_sec
        self.io_timeout_sec = io_timeout_sec
        self.max_line_bytes = max_line_bytes
        self.debug_skip = debug_skip

        self.sock: Optional[socket.socket] = None
        self.rxbuf = bytearray()

    def connect(self) -> None:
        s = socket.create_connection((self.host, self.port), timeout=self.connect_timeout_sec)
        s.settimeout(self.io_timeout_sec)
        # optional: reduce latency
        try:
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except Exception:
            pass
        self.sock = s

    def close(self) -> None:
        if not self.sock:
            return
        try:
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            self.sock.close()
        finally:
            self.sock = None
            self.rxbuf.clear()

    def send_json(self, obj: Dict[str, Any]) -> None:
        if not self.sock:
            raise RuntimeError("socket not connected")
        data = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")
        self.sock.sendall(data)

    def _recv_until_newline(self, deadline: float) -> Optional[bytes]:
        """
        Returns one line without trailing '\n'. None if deadline exceeded.
        Raises RuntimeError on EOF.
        """
        while True:
            nl = self.rxbuf.find(b"\n")
            if nl != -1:
                line = bytes(self.rxbuf[:nl])
                del self.rxbuf[:nl + 1]
                return line

            if time.time() >= deadline:
                return None

            if not self.sock:
                raise RuntimeError("socket not connected")

            try:
                chunk = self.sock.recv(4096)
                if not chunk:
                    raise RuntimeError("socket closed (EOF)")
                self.rxbuf.extend(chunk)
                if len(self.rxbuf) > self.max_line_bytes:
                    # avoid unbounded memory if server stops sending '\n'
                    if self.debug_skip:
                        print(f"[WARN] rxbuf too large ({len(self.rxbuf)} bytes), dropping buffer", file=sys.stderr)
                    self.rxbuf.clear()
            except socket.timeout:
                # try again until deadline
                continue

    def read_json_line(self, deadline: float) -> Optional[Dict[str, Any]]:
        """
        Read and parse one JSON line by deadline.
        - returns None if deadline exceeded
        - returns {} for blank / non-JSON line (skipped)
        """
        raw = self._recv_until_newline(deadline)
        if raw is None:
            return None

        s = raw.strip()
        if not s:
            return {}

        try:
            return json.loads(s.decode("utf-8"))
        except Exception:
            if self.debug_skip:
                print(f"[SKIP NON-JSON] {raw[:200]!r}", file=sys.stderr)
            return {}

    def drain(self, duration_sec: float = 0.3) -> None:
        """Best-effort: read and discard for a short time."""
        end = time.time() + max(0.0, duration_sec)
        while time.time() < end:
            msg = self.read_json_line(deadline=time.time() + 0.05)
            if msg is None:
                break


# -----------------------------
# StackFlow API (subset)
# -----------------------------

@dataclass
class LLMSetupConfig:
    model: str
    system_prompt: str
    response_format: str = "llm.utf-8"       # non-stream output
    input_object: str = "llm.utf-8.stream"  # stream input
    max_token_len: int = 32
    enoutput: bool = True
    enkws: bool = False


class StackFlow:
    def __init__(self, jsock: JSONLSocket):
        self.s = jsock

    def _wait_request(self, request_id: str, timeout_sec: float) -> Dict[str, Any]:
        deadline = time.time() + timeout_sec
        while True:
            msg = self.s.read_json_line(deadline=deadline)
            if msg is None:
                raise TimeoutError(f"timeout waiting response for request_id={request_id}")
            if not msg:
                continue
            if msg.get("request_id") == request_id:
                return msg
            # otherwise skip

    # ---- SYS ----

    def sys_ping(self, timeout_sec: float = 5.0) -> Dict[str, Any]:
        req_id = f"sys_ping_{int(time.time()*1000)}"
        self.s.send_json({"request_id": req_id, "work_id": "sys", "action": "ping"})
        return self._wait_request(req_id, timeout_sec)

    def sys_reset(self, timeout_sec: float = 20.0) -> Optional[Dict[str, Any]]:
        """
        Reset unit. Connection may drop; treat EOF as success.
        """
        req_id = f"sys_reset_{int(time.time()*1000)}"
        try:
            self.s.send_json({"request_id": req_id, "work_id": "sys", "action": "reset"})
        except Exception:
            return None

        try:
            return self._wait_request(req_id, timeout_sec)
        except RuntimeError:
            # connection dropped (EOF) -> likely reset executed
            return None
        except TimeoutError:
            return None

    # ---- LLM ----

    def llm_setup(self, cfg: LLMSetupConfig, timeout_sec: float = 60.0) -> Tuple[str, Dict[str, Any]]:
        req_id = f"setup_{int(time.time()*1000)}"
        req = {
            "request_id": req_id,
            "work_id": "llm",
            "action": "setup",
            "object": "llm.setup",
            "data": {
                "model": cfg.model,
                "response_format": cfg.response_format,
                "input": cfg.input_object,
                "enoutput": bool(cfg.enoutput),
                "enkws": bool(cfg.enkws),
                "max_token_len": int(cfg.max_token_len),
                "prompt": cfg.system_prompt,
            },
        }
        self.s.send_json(req)
        resp = self._wait_request(req_id, timeout_sec)
        work_id = resp.get("work_id")
        if not isinstance(work_id, str) or not work_id:
            work_id = "llm"
        return work_id, resp

    def llm_inference(
        self,
        work_id: str,
        user_prompt: str,
        input_object: str,
        soft_prefix_b64: Optional[str],
        soft_prefix_len: int,
        timeout_sec: float = 240.0,
    ) -> str:
        req_id = f"infer_{int(time.time()*1000)}"
        data_obj: Dict[str, Any] = {"delta": user_prompt, "index": 0, "finish": True}
        if soft_prefix_b64 is not None:
            data_obj["soft_prefix"] = {"len": int(soft_prefix_len), "data_b64": soft_prefix_b64}
            print ("SoftPrefix: ", data_obj["soft_prefix"])

        req = {
            "request_id": req_id,
            "work_id": work_id,
            "action": "inference",
            "object": input_object,  # llm.utf-8.stream
            "data": data_obj,
        }
        self.s.send_json(req)

        deadline = time.time() + timeout_sec
        out_chunks: List[str] = []

        while True:
            msg = self.s.read_json_line(deadline=deadline)
            if msg is None:
                raise TimeoutError(f"timeout waiting inference result request_id={req_id}")
            if not msg:
                continue
            if msg.get("request_id") != req_id:
                continue

            err = msg.get("error", {}) or {}
            if isinstance(err, dict) and err.get("code", 0) != 0:
                raise RuntimeError(f"llm.inference failed: err={err} full={msg}")

            data = msg.get("data")
            if isinstance(data, str):
                out_chunks.append(data)
                break
            if isinstance(data, dict):
                d = data.get("delta")
                if isinstance(d, str) and d:
                    out_chunks.append(d)
                if data.get("finish") is True:
                    break

        return "".join(out_chunks).strip()

    def llm_taskinfo(self, work_id: str, timeout_sec: float = 5.0) -> Optional[Dict[str, Any]]:
        req_id = f"taskinfo_{int(time.time()*1000)}"
        self.s.send_json({"request_id": req_id, "work_id": work_id, "action": "taskinfo"})
        try:
            return self._wait_request(req_id, timeout_sec)
        except Exception:
            return None

    def llm_exit(self, work_id: str, timeout_sec: float = 10.0) -> Optional[Dict[str, Any]]:
        req_id = f"exit_{int(time.time()*1000)}"
        try:
            self.s.send_json({"request_id": req_id, "work_id": work_id, "action": "exit"})
        except Exception:
            return None

        try:
            return self._wait_request(req_id, timeout_sec)
        except Exception:
            return None


# -----------------------------
# OpenAI-compatible TTS client
# -----------------------------

def http_post_json(url: str, payload: Dict[str, Any], timeout_s: float = 120.0) -> bytes:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer sk-",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        body = e.read()
        raise RuntimeError(f"HTTP {e.code} {e.reason}: {body[:500]!r}") from e


def tts_generate_wav(base_url: str, model_id: str, text: str, out_wav_path: Path, speed: float) -> None:
    url = base_url.rstrip("/") + "/audio/speech"
    payload = {
        "model": model_id,
        "input": text,
        "response_format": "wav",
        "speed": float(speed),
    }
    audio = http_post_json(url, payload, timeout_s=120.0)
    out_wav_path.write_bytes(audio)

def ffmpeg_convert_for_tinyplay(
    in_wav: Path,
    out_wav: Path,
    ar_hz: int,
    channels: int,
    sample_fmt: str,
    quiet: bool = True,
) -> None:
    cmd = ["ffmpeg", "-y"]
    if quiet:
        cmd += ["-hide_banner", "-loglevel", "error"]
    cmd += [
        "-i", str(in_wav),
        "-ar", str(ar_hz),
        "-ac", str(channels),
        "-sample_fmt", sample_fmt,
        str(out_wav),
    ]
    subprocess.run(cmd, check=True)

    rumble_layered_with_fx(
        INPUT_WAV_16K,
        out_r3,
        pitch_steps=-16.0,
        sub_oct_mix=0.55,
        rumble_mix=0.25,
        rumble_base_hz=55.0,
        drive=0.55,
        xover_hz=280.0
    )

def ffmpeg_convert_for_tinyplay_with_rumble(
    in_wav: Path,
    out_wav: Path,
    ar_hz: int,
    channels: int,
    sample_fmt: str,
    quiet: bool = True,
) -> None:
    cmd = ["ffmpeg", "-y"]
    if quiet:
        cmd += ["-hide_banner", "-loglevel", "error"]
    cmd += [
        "-i", str(in_wav),
        "-ar", str(ar_hz),
        "-ac", str(channels),
        "-sample_fmt", sample_fmt,
        str(out_wav),
    ]
    subprocess.run(cmd, check=True)

    rumble_layered_with_fx(
        out_wav,
        out_wav,
        pitch_steps=-16.0,
        sub_oct_mix=0.55,
        rumble_mix=0.25,
        rumble_base_hz=55.0,
        drive=0.55,
        xover_hz=280.0
    )


def tinyplay_play(wav_path: Path, card: int, device: int) -> None:
    cmd = ["tinyplay", f"-D{card}", f"-d{device}", str(wav_path)]
    subprocess.run(cmd, check=True)


# -----------------------------
# Helpers
# -----------------------------

def is_already_working_error(resp: Dict[str, Any]) -> bool:
    err = resp.get("error", {}) or {}
    if isinstance(err, dict):
        code = err.get("code", 0)
        # docs: -13 "Module is already working"
        return code == -13
    return False


def cleanup_llm_best_effort(sf: StackFlow, js: JSONLSocket, work_id: Optional[str], args: argparse.Namespace) -> None:
    """
    Always try to cleanup LLM work.
    NOTE: This may take time if waiting 'deinit'. We run it AFTER TTS by default.
    """
    if js.sock is None:
        return
    try:
        if work_id:
            print("[INFO] llm.exit (graceful)...")
            sf.llm_exit(work_id, timeout_sec=float(args.exit_timeout))

            # poll taskinfo until deinit (best-effort)
            end = time.time() + float(args.deinit_timeout)
            while time.time() < end:
                st = sf.llm_taskinfo(work_id, timeout_sec=3.0)
                if not st:
                    break
                data = st.get("data")
                if data == "deinit":
                    break
                e = st.get("error", {}) or {}
                if isinstance(e, dict) and e.get("code") in (-14, -19):
                    break
                time.sleep(0.2)

        # optional strong clean after
        if args.clean:
            print("[INFO] sys.reset (clean after)...")
            sf.sys_reset(timeout_sec=20.0)

        js.drain(0.3)
    except Exception as e:
        print(f"[WARN] cleanup error (ignored): {e}", file=sys.stderr)
    finally:
        js.close()
        print("[INFO] socket closed")


# -----------------------------
# Main
# -----------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Module LLM -> (optional) TTS -> tinyplay with clean shutdown/recovery")

    ap.add_argument("--llm-host", default="127.0.0.1")
    ap.add_argument("--llm-port", type=int, default=10001)

    # timeouts
    ap.add_argument("--connect-timeout", type=float, default=5.0)
    ap.add_argument("--io-timeout", type=float, default=1.0)
    ap.add_argument("--setup-timeout", type=float, default=60.0)
    ap.add_argument("--infer-timeout", type=float, default=240.0)
    ap.add_argument("--exit-timeout", type=float, default=10.0)
    ap.add_argument("--deinit-timeout", type=float, default=10.0)

    # cleaning behavior
    ap.add_argument("--clean", action="store_true", help="sys.reset before and after running (strong clean)")
    ap.add_argument("--no-auto-reset", action="store_true", help="disable auto sys.reset on setup fail/timeout")
    ap.add_argument("--debug-skip", action="store_true")

    # NEW: keep old order if needed
    ap.add_argument(
        "--cleanup-before-tts",
        action="store_true",
        help="(old behavior) do LLM cleanup BEFORE TTS/playback (slower text->audio)",
    )

    # model/prompt
    ap.add_argument(
        "--llm", "--preset",
        dest="preset",
        choices=["qwen", "tinyswallow"],
        default="qwen",
        help="Select LLM preset (alias: --preset)",
    )
    ap.add_argument("--llm-model", default="", help="Override model name (takes precedence over --llm/--preset)")
    ap.add_argument("--system-prompt", default="あなたは親切で簡潔な日本語アシスタントです。短く自然な日本語で答えてください。")
    ap.add_argument("--user", default="こんにちは。自己紹介を一文でお願いします。")
    ap.add_argument("--max-token-len", type=int, default=64)

    # softprefix
    ap.add_argument("--softprefix-val", type=float, default=None)
    ap.add_argument("--softprefix-len", type=int, default=1)
    ap.add_argument("--softprefix-h", type=int, default=None, help="Hidden size H for softprefix. If omitted, auto by preset.")

    # TTS
    ap.add_argument("--openai-base", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--tts-model", default="melotts-ja-jp")
    ap.add_argument("--tts-speed", type=float, default=1.0)
    ap.add_argument("--out-raw", default="/tmp/llm_tts_raw.wav")
    ap.add_argument("--out-play", default="/tmp/llm_tts_32k_stereo_s16.wav")
    ap.add_argument("--no-tts", action="store_true")
    ap.add_argument("--no-play", action="store_true")
    ap.add_argument("--tinyplay-card", type=int, default=0)
    ap.add_argument("--tinyplay-device", type=int, default=1)
    ap.add_argument("--ffmpeg-verbose", action="store_true", help="Show ffmpeg full logs (default: quiet)")
    ap.add_argument("--rumble", action="store_true", help="Activate rumble effects")

    args = ap.parse_args()

    preset_info = LLM_PRESETS.get(args.preset, LLM_PRESETS["qwen"])
    model_name = args.llm_model.strip() or str(preset_info["model"])

    # auto softprefix H when omitted
    if args.softprefix_h is None:
        args.softprefix_h = int(preset_info["softprefix_h"])

    cfg = LLMSetupConfig(
        model=model_name,
        system_prompt=args.system_prompt,
        response_format="llm.utf-8",
        input_object="llm.utf-8.stream",
        max_token_len=int(args.max_token_len),
        enoutput=True,
        enkws=False,
    )

    soft_b64: Optional[str] = None
    if args.softprefix_val is not None:
        # soft_b64 = make_soft_prefix_b64_constant(
        #     int(args.softprefix_len),
        #     int(args.softprefix_h),
        #     float(args.softprefix_val),
        # )
        soft_b64 = make_soft_prefix_b64_random_scaled(
            P=int(args.softprefix_len),
            H=int(args.softprefix_h),
            val=float(args.softprefix_val),
            seed=-1,
            dist="uniform")

    auto_reset = not args.no_auto_reset

    print(f"[INFO] LLM host={args.llm_host}:{args.llm_port}")
    print(f"[INFO] preset={args.preset}, model={cfg.model}")
    print(f"[INFO] response_format={cfg.response_format}, input_object={cfg.input_object}")
    if soft_b64 is None:
        print("[INFO] soft_prefix: disabled")
    else:
        print(f"[INFO] soft_prefix: enabled P={args.softprefix_len} H={args.softprefix_h} val={args.softprefix_val}")

    out_text = ""
    work_id: Optional[str] = None

    js = JSONLSocket(
        args.llm_host,
        args.llm_port,
        connect_timeout_sec=float(args.connect_timeout),
        io_timeout_sec=float(args.io_timeout),
        debug_skip=bool(args.debug_skip),
    )
    sf: Optional[StackFlow] = None

    def do_tts_and_play(text: str) -> None:
        if args.no_tts:
            print("[INFO] --no-tts specified. Done.")
            return

        raw_path = Path(args.out_raw)
        play_path = Path(args.out_play)

        print(f"[INFO] TTS base={args.openai_base}")
        print(f"[INFO] TTS model={args.tts_model} speed={args.tts_speed}")
        print(f"[INFO] Writing wav: {raw_path}")

        t0 = time.time()
        tts_generate_wav(args.openai_base, args.tts_model, text, raw_path, float(args.tts_speed))
        t1 = time.time()
        print(f"[INFO] TTS time: {t1 - t0:.2f}s")

        print(f"[INFO] Converting for tinyplay: {play_path}")
        t2 = time.time()
        if args.rumble:
            ffmpeg_convert_for_tinyplay_with_rumble(
                raw_path,
                play_path,
                ar_hz=32000,
                channels=2,
                sample_fmt="s16",
                quiet=(not args.ffmpeg_verbose),
            )
        else:
            ffmpeg_convert_for_tinyplay(
                raw_path,
                play_path,
                ar_hz=32000,
                channels=2,
                sample_fmt="s16",
                quiet=(not args.ffmpeg_verbose),
            )
        t3 = time.time()
        print(f"[INFO] ffmpeg convert time: {t3 - t2:.2f}s")

        if args.no_play:
            print("[INFO] --no-play specified. Done.")
            return

        print(f"[INFO] tinyplay: card={args.tinyplay_card}, device={args.tinyplay_device}")
        t4 = time.time()
        tinyplay_play(play_path, card=int(args.tinyplay_card), device=int(args.tinyplay_device))
        t5 = time.time()
        print(f"[INFO] playback call time: {t5 - t4:.2f}s")

    try:
        print("[INFO] connecting...")
        js.connect()
        sf = StackFlow(js)
        print("[INFO] connected")

        # optional strong clean before
        if args.clean:
            print("[INFO] sys.reset (clean before)...")
            sf.sys_reset(timeout_sec=20.0)
            js.close()
            time.sleep(1.0)
            print("[INFO] reconnect after reset...")
            js.connect()
            sf = StackFlow(js)
            print("[INFO] reconnected")

        # ping (optional)
        try:
            sf.sys_ping(timeout_sec=5.0)
        except Exception:
            pass

        # ---- setup with recovery ----
        def do_setup_once() -> Tuple[str, Dict[str, Any]]:
            assert sf is not None
            return sf.llm_setup(cfg, timeout_sec=float(args.setup_timeout))

        try:
            work_id, setup_resp = do_setup_once()
        except TimeoutError:
            if not auto_reset:
                raise
            print("[WARN] llm.setup timeout -> sys.reset and retry", file=sys.stderr)
            assert sf is not None
            sf.sys_reset(timeout_sec=20.0)
            js.close()
            time.sleep(1.0)
            js.connect()
            sf = StackFlow(js)
            work_id, setup_resp = do_setup_once()

        err = (setup_resp.get("error") or {})
        if isinstance(err, dict) and err.get("code", 0) != 0:
            if is_already_working_error(setup_resp) and auto_reset:
                print("[WARN] Module already working -> sys.reset and retry", file=sys.stderr)
                assert sf is not None
                sf.sys_reset(timeout_sec=20.0)
                js.close()
                time.sleep(1.0)
                js.connect()
                sf = StackFlow(js)
                work_id, setup_resp = do_setup_once()
                err2 = (setup_resp.get("error") or {})
                if isinstance(err2, dict) and err2.get("code", 0) != 0:
                    raise RuntimeError(f"llm.setup failed after reset: {err2} full={setup_resp}")
            else:
                raise RuntimeError(f"llm.setup failed: {err} full={setup_resp}")

        print(f"[INFO] work_id={work_id}")

        # ---- inference ----
        t0 = time.time()
        assert sf is not None
        out_text = sf.llm_inference(
            work_id=work_id,
            user_prompt=args.user,
            input_object=cfg.input_object,
            soft_prefix_b64=soft_b64,
            soft_prefix_len=int(args.softprefix_len),
            timeout_sec=float(args.infer_timeout),
        )
        dt = time.time() - t0

        print("\n========== LLM OUTPUT ==========")
        print(out_text)
        print("================================")
        print(f"[INFO] inference time: {dt:.2f}s\n")

        # ---- TTS/playback vs cleanup order ----
        if args.cleanup_before_tts:
            # old order (slower perceived latency)
            cleanup_llm_best_effort(sf, js, work_id, args)
            do_tts_and_play(out_text)
        else:
            # new order (faster perceived latency): play first, cleanup later
            do_tts_and_play(out_text)
            cleanup_llm_best_effort(sf, js, work_id, args)

        return 0

    finally:
        # If we already cleaned inside try (either order), js.sock will be None here.
        # If an exception happened before cleanup, do best-effort cleanup here.
        if sf is not None and js.sock is not None:
            cleanup_llm_best_effort(sf, js, work_id, args)


if __name__ == "__main__":
    raise SystemExit(main())
