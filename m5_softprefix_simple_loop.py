#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple M5Stack Module LLM (llm-sys) -> OpenAI-compatible TTS -> ffmpeg -> tinyplay

IMPORTANT:
- llm-sys TCP(10001) framing in your environment is JSONL (newline-delimited JSON),
  NOT "10-byte length header + JSON".
- SoftPrefix is passed as:
    data.soft_prefix = {"len": P, "data_b64": base64(bf16_le_u16 repeated P*H)}

This script keeps it minimal:
- llm.setup
- llm.inference (input: llm.utf-8.stream, response_format: llm.utf-8)
- optional soft_prefix injection
- optional TTS + convert + tinyplay
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
from typing import Any, Dict, Optional, List
import random



# -----------------------------
# SoftPrefix helpers (bf16 b64)
# -----------------------------

def f32_to_bf16_u16(x: float) -> int:
    """float32 -> bf16(truncate) -> u16"""
    u32 = struct.unpack("<I", struct.pack("<f", float(x)))[0]
    return (u32 >> 16) & 0xFFFF


def make_soft_prefix_b64_constant(P: int, H: int, val: float) -> str:
    """
    Build base64 of bf16 little-endian u16 repeated P*H times.
    """
    u16 = f32_to_bf16_u16(val)
    raw = struct.pack("<H", u16) * (P * H)
    return base64.b64encode(raw).decode("ascii")


# -----------------------------
# JSONL socket client (LLM)
# -----------------------------

@dataclass
class LLMSetupConfig:
    model: str
    system_prompt: str
    response_format: str = "llm.utf-8"       # non-stream output
    input_object: str = "llm.utf-8.stream"  # stream input
    max_token_len: int = 128
    enoutput: bool = True
    enkws: bool = False


class JSONLClient:
    """
    Minimal JSONL (newline-delimited JSON) client.
    Matches your working reference: sock.makefile + readline/write + '\n'
    """
    def __init__(self, host: str, port: int, sock_timeout_sec: int, debug_skip: bool = False):
        self.host = host
        self.port = port
        self.sock_timeout_sec = sock_timeout_sec
        self.debug_skip = debug_skip

        self.sock: Optional[socket.socket] = None
        self.r = None
        self.w = None

    def __enter__(self):
        self.sock = socket.create_connection((self.host, self.port), timeout=self.sock_timeout_sec)
        self.sock.settimeout(self.sock_timeout_sec)
        self.r = self.sock.makefile("r", encoding="utf-8", newline="\n")
        self.w = self.sock.makefile("w", encoding="utf-8", newline="\n")
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if self.w:
                self.w.flush()
        except Exception:
            pass
        try:
            if self.r:
                self.r.close()
        except Exception:
            pass
        try:
            if self.w:
                self.w.close()
        except Exception:
            pass
        try:
            if self.sock:
                self.sock.close()
        except Exception:
            pass

    def send_json(self, obj: Dict[str, Any]) -> None:
        self.w.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self.w.flush()

    def read_json_line(self) -> Optional[Dict[str, Any]]:
        """
        Read one JSON line.
        Returns None on socket timeout (so wait loops can continue until global deadline).
        """
        try:
            line = self.r.readline()
        except (socket.timeout, OSError) as e:
            # timed out -> let caller handle global timeout
            if "timed out" in str(e).lower():
                return None
            raise

        if line == "":
            raise RuntimeError("socket closed (EOF)")

        line = line.strip()
        if not line:
            return {}

        try:
            return json.loads(line)
        except json.JSONDecodeError:
            # If server logs garbage lines, you can skip them.
            if self.debug_skip:
                print(f"[SKIP NON-JSON] {line!r}", file=sys.stderr)
            return {}

    def wait_for_request(self, request_id: str, max_wait_sec: int) -> Dict[str, Any]:
        t0 = time.time()
        while True:
            if time.time() - t0 > max_wait_sec:
                raise TimeoutError(f"timeout waiting response for request_id={request_id}")

            msg = self.read_json_line()
            if msg is None:
                continue
            if not msg:
                continue

            if msg.get("request_id") == request_id:
                return msg

            if self.debug_skip:
                print(f"[SKIP] {msg}", file=sys.stderr)

    def setup(self, cfg: LLMSetupConfig) -> str:
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
        self.send_json(req)
        resp = self.wait_for_request(req_id, max_wait_sec=60)

        err = resp.get("error", {}) or {}
        if isinstance(err, dict) and err.get("code", 0) != 0:
            raise RuntimeError(f"llm.setup failed: err={err} full={resp}")

        work_id = resp.get("work_id")
        if isinstance(work_id, str) and work_id:
            return work_id
        return "llm"

    def inference(
        self,
        work_id: str,
        user_prompt: str,
        input_object: str,
        soft_prefix_b64: Optional[str],
        soft_prefix_len: int,
        max_wait_sec: int,
    ) -> str:
        req_id = f"infer_{int(time.time()*1000)}"

        data_obj: Dict[str, Any] = {"delta": user_prompt, "index": 0, "finish": True}
        if soft_prefix_b64 is not None:
            data_obj["soft_prefix"] = {"len": int(soft_prefix_len), "data_b64": soft_prefix_b64}

        req = {
            "request_id": req_id,
            "work_id": work_id,
            "action": "inference",
            "object": input_object,   # llm.utf-8.stream
            "data": data_obj,
        }
        self.send_json(req)

        # response_format is non-stream, but just in case handle stream-ish dict too
        t0 = time.time()
        out_chunks: List[str] = []

        while True:
            if time.time() - t0 > max_wait_sec:
                raise TimeoutError(f"timeout waiting inference result request_id={req_id}")

            msg = self.read_json_line()
            if msg is None:
                continue
            if not msg:
                continue

            if msg.get("request_id") != req_id:
                if self.debug_skip:
                    print(f"[SKIP] {msg}", file=sys.stderr)
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

            # otherwise keep waiting

        return "".join(out_chunks).strip()

    def exit(self, work_id: str) -> None:
        # exit response may or may not come; do not wait
        req_id = f"exit_{int(time.time()*1000)}"
        self.send_json({"request_id": req_id, "work_id": work_id, "action": "exit"})


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


def ffmpeg_convert_for_tinyplay(in_wav: Path, out_wav: Path, ar_hz: int, channels: int, sample_fmt: str) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-i", str(in_wav),
        "-ar", str(ar_hz),
        "-ac", str(channels),
        "-sample_fmt", sample_fmt,
        str(out_wav),
    ]
    subprocess.run(cmd, check=True)


def tinyplay_play(wav_path: Path, card: int, device: int) -> None:
    cmd = ["tinyplay", f"-D{card}", f"-d{device}", str(wav_path)]
    subprocess.run(cmd, check=True)


# -----------------------------
# Main
# -----------------------------

def default_model_for_preset(preset: str) -> str:
    if preset == "qwen":
        # match your working sample
        return "qwen2.5-0.5B-prefill-20e"
    if preset == "tinyswallow":
        return "TinySwallow-1.5B-Instruct-ax630c"
    return "qwen2.5-0.5B-prefill-20e"


def main() -> int:
    ap = argparse.ArgumentParser(description="Simple Module LLM -> TTS wav -> ffmpeg -> tinyplay (JSONL protocol)")

    ap.add_argument("--llm-host", default="127.0.0.1", help="llm-sys host (on-device: 127.0.0.1, from PC: device IP)")
    ap.add_argument("--llm-port", type=int, default=10001, help="llm-sys TCP port (default 10001)")
    ap.add_argument("--llm-timeout", type=int, default=240, help="socket timeout seconds (default 240)")

    ap.add_argument("--preset", choices=["qwen", "tinyswallow"], default="qwen")
    ap.add_argument("--llm-model", default="", help="override model name (if empty, use preset default)")

    ap.add_argument("--system-prompt", default="あなたは親切で簡潔な日本語アシスタントです。短く自然な日本語で答えてください。")
    ap.add_argument("--user", default="こんにちは。自己紹介を一文でお願いします。")
    ap.add_argument("--max-token-len", type=int, default=128)

    # SoftPrefix (constant fill)
    ap.add_argument("--softprefix-val", type=float, default=None, help="enable soft_prefix with constant value (e.g. 0.01)")
    ap.add_argument("--softprefix-len", type=int, default=1, help="P: prefix token length (default 1)")
    ap.add_argument("--softprefix-h", type=int, default=896, help="H: tokens_embed_size (default 896)")

    # TTS
    ap.add_argument("--openai-base", default="http://127.0.0.1:8000/v1", help="OpenAI-compatible base_url")
    ap.add_argument("--tts-model", default="melotts-ja-jp")
    ap.add_argument("--tts-speed", type=float, default=1.0)
    ap.add_argument("--out-raw", default="/tmp/llm_tts_raw.wav")
    ap.add_argument("--out-play", default="/tmp/llm_tts_32k_stereo_s16.wav")
    ap.add_argument("--no-tts", action="store_true", help="skip TTS/ffmpeg/tinyplay (LLM only)")
    ap.add_argument("--no-play", action="store_true", help="do not run tinyplay")
    ap.add_argument("--tinyplay-card", type=int, default=0)
    ap.add_argument("--tinyplay-device", type=int, default=1)

    ap.add_argument("--debug-skip", action="store_true", help="print skipped messages (stderr)")

    args = ap.parse_args()

    model_name = args.llm_model.strip() or default_model_for_preset(args.preset)

    cfg = LLMSetupConfig(
        model=model_name,
        system_prompt=args.system_prompt,
        response_format="llm.utf-8",        # non-stream output
        input_object="llm.utf-8.stream",    # stream input (for soft_prefix)
        max_token_len=int(args.max_token_len),
        enoutput=True,
        enkws=False,
    )

    question_list = ["植物に関する詩を書いてください", "何かいいことがありましたか？", "植物の気持ちになって言葉を紡いでください"]

    soft_b64: Optional[str] = None
    if args.softprefix_val is not None:
        soft_b64 = make_soft_prefix_b64_constant(int(args.softprefix_len), int(args.softprefix_h), float(args.softprefix_val))

    print(f"[INFO] LLM host={args.llm_host}:{args.llm_port}")
    print(f"[INFO] model={cfg.model}")
    print(f"[INFO] response_format={cfg.response_format}, input_object={cfg.input_object}")
    if soft_b64 is None:
        print("[INFO] soft_prefix: disabled")
    else:
        print(f"[INFO] soft_prefix: enabled P={args.softprefix_len} H={args.softprefix_h} val={args.softprefix_val}")

    while:
        # ---- LLM ----
        with JSONLClient(args.llm_host, args.llm_port, args.llm_timeout, debug_skip=args.debug_skip) as cli:
            work_id = cli.setup(cfg)
            print(f"[INFO] work_id={work_id}")

            t0 = time.time()
            out_text = cli.inference(
                work_id=work_id,
                user_prompt=random.choice(question_list),
                input_object=cfg.input_object,
                soft_prefix_b64=soft_b64,
                soft_prefix_len=int(args.softprefix_len),
                max_wait_sec=int(args.llm_timeout),
            )
            dt = time.time() - t0

            print("\n========== LLM OUTPUT ==========")
            print(out_text)
            print("================================")
            print(f"[INFO] inference time: {dt:.2f}s\n")

            # optional
            cli.exit(work_id)

        if args.no_tts:
            print("[INFO] --no-tts specified. Done.")
            return 0

        # ---- TTS ----
        raw_path = Path(args.out_raw)
        play_path = Path(args.out_play)

        print(f"[INFO] TTS base={args.openai_base}")
        print(f"[INFO] TTS model={args.tts_model} speed={args.tts_speed}")
        print(f"[INFO] Writing wav: {raw_path}")
        tts_generate_wav(args.openai_base, args.tts_model, out_text, raw_path, float(args.tts_speed))

        print(f"[INFO] Converting for tinyplay: {play_path}")
        ffmpeg_convert_for_tinyplay(raw_path, play_path, ar_hz=32000, channels=2, sample_fmt="s16")

        if args.no_play:
            print("[INFO] --no-play specified. Done.")
            return 0

        print(f"[INFO] tinyplay: card={args.tinyplay_card}, device={args.tinyplay_device}")
        tinyplay_play(play_path, card=int(args.tinyplay_card), device=int(args.tinyplay_device))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
