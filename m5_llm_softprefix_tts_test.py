#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
M5Stack Module LLM (Linux) integration test:
- LLM via StackFlow llm-sys TCP socket (default 127.0.0.1:10001)
  * JSON frame = 10-byte length header + UTF-8 JSON body  (see docs/notes)  :contentReference[oaicite:2]{index=2}
  * llm.setup / llm.inference message formats follow M5Stack StackFlow API docs :contentReference[oaicite:3]{index=3}
- TTS via OpenAI-compatible API (default http://127.0.0.1:8000/v1)
  * POST /v1/audio/speech returns audio (wav) :contentReference[oaicite:4]{index=4}

SoftPrefix injection:
- This script sends a "softprefix val" in multiple candidate keys.
  Adjust keys to match your patched llm_framework if needed.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple


# -----------------------------
# TCP framing helpers (llm-sys)
# -----------------------------

def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise EOFError("socket closed")
        buf.extend(chunk)
    return bytes(buf)


def _read_len10(sock: socket.socket, max_resync: int = 1024) -> int:
    """
    Read 10-byte ASCII length header.
    Most implementations use 10-digit decimal (zero-padded).
    If we hit a stray newline/garbage, we try to resync by shifting 1 byte.
    """
    hdr = bytearray(_recv_exact(sock, 10))
    # Fast path
    try:
        s = bytes(hdr).decode("utf-8")
        if s.isdigit():
            return int(s)
    except Exception:
        pass

    # Resync: shift until header becomes 10 digits
    for _ in range(max_resync):
        hdr.pop(0)
        hdr.extend(_recv_exact(sock, 1))
        try:
            s = bytes(hdr).decode("utf-8")
            if s.isdigit():
                return int(s)
        except Exception:
            continue

    raise ValueError(f"Failed to parse/resync 10-byte length header: {bytes(hdr)!r}")


def send_frame(sock: socket.socket, obj: Dict[str, Any]) -> None:
    body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    header = f"{len(body):010d}".encode("utf-8")
    sock.sendall(header + body)


def recv_frame(sock: socket.socket, timeout_s: float) -> Dict[str, Any]:
    sock.settimeout(timeout_s)
    n = _read_len10(sock)
    body = _recv_exact(sock, n)
    try:
        return json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as e:
        raise RuntimeError(f"JSON decode failed: {e} / body={body[:200]!r}...") from e


def recv_until(
    sock: socket.socket,
    predicate,
    timeout_total_s: float,
    per_read_timeout_s: float = 5.0
) -> Dict[str, Any]:
    deadline = time.time() + timeout_total_s
    while True:
        remain = deadline - time.time()
        if remain <= 0:
            raise TimeoutError("Timed out waiting for expected response")
        msg = recv_frame(sock, timeout_s=min(per_read_timeout_s, remain))
        if predicate(msg):
            return msg


# -----------------------------
# StackFlow llm-sys client
# -----------------------------

@dataclass
class LLMSetupConfig:
    model: str
    system_prompt: str
    response_format: str = "llm.utf-8"         # non-stream output (easy to compare) :contentReference[oaicite:5]{index=5}
    input_format: str = "llm.utf-8.stream"     # stream input (delta/index/finish + extra fields) :contentReference[oaicite:6]{index=6}
    max_token_len: int = 256
    enoutput: bool = False
    enkws: bool = False


class StackFlowClient:
    def __init__(self, host: str, port: int, timeout_s: float = 5.0) -> None:
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self.sock: Optional[socket.socket] = None

    def connect(self) -> None:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(self.timeout_s)
        s.connect((self.host, self.port))
        self.sock = s

    def close(self) -> None:
        if self.sock:
            try:
                self.sock.close()
            finally:
                self.sock = None

    def _must_sock(self) -> socket.socket:
        if not self.sock:
            raise RuntimeError("Not connected")
        return self.sock

    def sys_ping(self) -> Dict[str, Any]:
        req = {
            "request_id": "sys_ping",
            "work_id": "sys",
            "action": "ping",
            "object": "None",
            "data": "None",
        }
        send_frame(self._must_sock(), req)
        return recv_until(
            self._must_sock(),
            lambda m: m.get("request_id") == "sys_ping" and m.get("work_id") == "sys",
            timeout_total_s=5.0,
        )

    def sys_lsmode(self) -> List[Dict[str, Any]]:
        req = {
            "request_id": "sys_lsmode",
            "work_id": "sys",
            "action": "lsmode",
            "object": "None",
            "data": "None",
        }
        send_frame(self._must_sock(), req)
        resp = recv_until(
            self._must_sock(),
            lambda m: m.get("request_id") == "sys_lsmode" and m.get("work_id") == "sys",
            timeout_total_s=10.0,
        )
        data = resp.get("data", [])
        if not isinstance(data, list):
            return []
        return data

    def llm_setup(self, cfg: LLMSetupConfig) -> str:
        """
        Returns work_id like "llm.1003"
        """
        req_id = f"llm_setup_{int(time.time())}"
        req = {
            "request_id": req_id,
            "work_id": "llm",
            "action": "setup",
            "object": "llm.setup",
            "data": {
                "model": cfg.model,
                "response_format": cfg.response_format,
                "input": cfg.input_format,
                "enoutput": bool(cfg.enoutput),
                "enkws": bool(cfg.enkws),
                "max_token_len": int(cfg.max_token_len),
                "prompt": cfg.system_prompt,
            },
        }
        send_frame(self._must_sock(), req)

        resp = recv_until(
            self._must_sock(),
            lambda m: m.get("request_id") == req_id and str(m.get("work_id", "")).startswith("llm."),
            timeout_total_s=30.0,
        )
        err = resp.get("error", {}) or {}
        if err.get("code", 0) != 0:
            raise RuntimeError(f"llm.setup failed: {err}")

        work_id = resp.get("work_id")
        if not isinstance(work_id, str) or not work_id.startswith("llm."):
            raise RuntimeError(f"Unexpected llm.setup response: {resp}")
        return work_id

    def llm_infer_stream_input(
        self,
        work_id: str,
        user_text: str,
        softprefix_val: Optional[float] = None,
        timeout_total_s: float = 120.0,
    ) -> str:
        """
        Sends inference as llm.utf-8.stream input (delta/index/finish),
        waits for output (either llm.utf-8 string OR llm.utf-8.stream delta chunks).
        """
        req_id = f"llm_infer_{int(time.time())}"

        data_obj: Dict[str, Any] = {
            "delta": user_text,
            "index": 0,
            "finish": True,
        }

        # --- SoftPrefix injection: send multiple candidate keys ---
        # Adjust this block to match your patched llm_framework key names.
        if softprefix_val is not None:
            data_obj.update({
                "soft_prefix_val": float(softprefix_val),
                "softprefix_val": float(softprefix_val),
                "prefix_val": float(softprefix_val),
                "test_embed_val": float(softprefix_val),
                "soft_prefix": {"val": float(softprefix_val)},
                "enable_soft_prefix": True,
            })

        req = {
            "request_id": req_id,
            "work_id": work_id,
            "action": "inference",
            "object": "llm.utf-8.stream",
            "data": data_obj,
        }

        send_frame(self._must_sock(), req)

        # Many implementations first send an "ack" (data push ok) then later the generation result.
        # We'll collect responses for this work_id until finish or a full string arrives.
        out_chunks: List[str] = []
        deadline = time.time() + timeout_total_s

        while True:
            remain = deadline - time.time()
            if remain <= 0:
                raise TimeoutError("Timed out waiting for LLM inference output")

            msg = recv_frame(self._must_sock(), timeout_s=min(5.0, remain))

            if msg.get("work_id") != work_id:
                continue

            err = msg.get("error", {}) or {}
            if err.get("code", 0) != 0:
                raise RuntimeError(f"LLM inference error: {err} / msg={msg}")

            data = msg.get("data")

            # Non-stream output (recommended in your repo README) :contentReference[oaicite:7]{index=7}
            if isinstance(data, str):
                out_chunks.append(data)
                break

            # Stream output
            if isinstance(data, dict):
                delta = data.get("delta")
                if isinstance(delta, str) and delta:
                    out_chunks.append(delta)
                if data.get("finish") is True:
                    break

            # Other messages: ignore

        return "".join(out_chunks).strip()


# -----------------------------
# OpenAI-compatible TTS client
# -----------------------------

def http_post_json(url: str, payload: Dict[str, Any], timeout_s: float = 60.0) -> bytes:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=data,
        headers={
            "Content-Type": "application/json",
            # OpenAI互換なので Bearer はダミーでも通ることが多い（必要なら差し替え）
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


def tts_generate_wav(
    base_url: str,
    model_id: str,
    text: str,
    out_wav_path: Path,
    speed: float = 1.0,
) -> None:
    # POST {base_url}/audio/speech
    url = base_url.rstrip("/") + "/audio/speech"
    payload = {
        "model": model_id,
        "input": text,
        "response_format": "wav",
        "speed": float(speed),
    }
    audio_bytes = http_post_json(url, payload, timeout_s=120.0)
    out_wav_path.write_bytes(audio_bytes)


def ffmpeg_convert_for_tinyplay(
    in_wav: Path,
    out_wav: Path,
    ar_hz: int = 32000,
    channels: int = 2,
    sample_fmt: str = "s16",
) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-i", str(in_wav),
        "-ar", str(ar_hz),
        "-ac", str(channels),
        "-sample_fmt", sample_fmt,
        str(out_wav),
    ]
    subprocess.run(cmd, check=True)


def tinyplay_play(wav_path: Path, card: int = 0, device: int = 1) -> None:
    cmd = ["tinyplay", f"-D{card}", f"-d{device}", str(wav_path)]
    subprocess.run(cmd, check=True)


# -----------------------------
# Model selection helpers
# -----------------------------

def pick_model_by_preset(models: List[Dict[str, Any]], preset: str) -> Optional[str]:
    """
    Try to pick an installed model name from sys.lsmode list.
    """
    # sys.lsmode item may have keys: model / mode / id
    candidates: List[str] = []
    for it in models:
        for k in ("model", "mode", "id", "name"):
            v = it.get(k)
            if isinstance(v, str):
                candidates.append(v)

    preset_l = preset.lower()
    if preset_l == "qwen":
        for name in candidates:
            if "qwen" in name.lower():
                return name
    if preset_l in ("tinyswallow", "tiny_swallow", "tiny-swalllow", "tiny-swallow"):
        for name in candidates:
            if "tinyswallow" in name.lower():
                return name

    return None


# -----------------------------
# Main
# -----------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="LLM (SoftPrefix) -> TTS wav -> ffmpeg -> tinyplay test on M5Stack Module LLM")

    ap.add_argument("--llm-host", default="127.0.0.1", help="llm-sys host (on-device: 127.0.0.1, from PC: device IP or m5stack-LLM.local)")
    ap.add_argument("--llm-port", type=int, default=10001, help="llm-sys TCP port (default 10001)")
    ap.add_argument("--openai-base", default="http://127.0.0.1:8000/v1", help="OpenAI-compatible base_url (default on-device)")

    ap.add_argument("--preset", choices=["qwen", "tinyswallow"], default="qwen", help="LLM preset model selector")
    ap.add_argument("--llm-model", default="", help="Override LLM model name explicitly (if empty, auto-pick via sys.lsmode)")

    ap.add_argument("--system-prompt", default="あなたは親切で簡潔な日本語アシスタントです。短く自然な日本語で答えてください。", help="System prompt used at llm.setup")
    ap.add_argument("--user", default="こんにちは。自己紹介を一文でお願いします。", help="User input text")

    ap.add_argument("--max-token-len", type=int, default=256, help="max_token_len for llm.setup")

    ap.add_argument("--softprefix-val", type=float, default=None, help="Enable SoftPrefix injection by sending this float value (e.g. 0.01). Omit for baseline.")

    ap.add_argument("--tts-model", default="melotts-ja-jp", help="TTS model id (e.g. melotts-ja-jp) :contentReference[oaicite:8]{index=8}")
    ap.add_argument("--tts-speed", type=float, default=1.0, help="TTS speed (default 1.0)")

    ap.add_argument("--out-raw", default="/tmp/llm_tts_raw.wav", help="Output raw wav path")
    ap.add_argument("--out-play", default="/tmp/llm_tts_32k_stereo_s16.wav", help="Output converted wav path for tinyplay")

    ap.add_argument("--no-play", action="store_true", help="Do not run tinyplay")
    ap.add_argument("--tinyplay-card", type=int, default=0, help="tinyplay card number (default 0)")
    ap.add_argument("--tinyplay-device", type=int, default=1, help="tinyplay device number (default 1)")

    args = ap.parse_args()

    raw_path = Path(args.out_raw)
    play_path = Path(args.out_play)

    cli = StackFlowClient(args.llm_host, args.llm_port)

    try:
        cli.connect()
        ping = cli.sys_ping()
        # ping response is mostly error.code==0 :contentReference[oaicite:9]{index=9}
        if (ping.get("error") or {}).get("code", 0) != 0:
            print(f"[WARN] sys.ping error: {ping}", file=sys.stderr)

        lsmode = cli.sys_lsmode()
        model_name = args.llm_model.strip()
        if not model_name:
            picked = pick_model_by_preset(lsmode, args.preset)
            if picked:
                model_name = picked
            else:
                # Fallback defaults from docs / common setup:
                if args.preset == "qwen":
                    model_name = "qwen2.5-0.5b"  # doc example :contentReference[oaicite:10]{index=10}
                else:
                    model_name = "TinySwallow-1.5B-Instruct-ax630c"  # common mode name :contentReference[oaicite:11]{index=11}

        print(f"[INFO] Using LLM model: {model_name}")
        if args.softprefix_val is None:
            print("[INFO] SoftPrefix: DISABLED (baseline)")
        else:
            print(f"[INFO] SoftPrefix: ENABLED val={args.softprefix_val}")

        cfg = LLMSetupConfig(
            model=model_name,
            system_prompt=args.system_prompt,
            response_format="llm.utf-8",         # non-stream output (recommended in repo) :contentReference[oaicite:12]{index=12}
            input_format="llm.utf-8.stream",     # stream input (delta/index/finish + softprefix fields) :contentReference[oaicite:13]{index=13}
            max_token_len=args.max_token_len,
            enoutput=False,
            enkws=False,
        )
        work_id = cli.llm_setup(cfg)
        print(f"[INFO] LLM work_id: {work_id}")

        llm_text = cli.llm_infer_stream_input(
            work_id=work_id,
            user_text=args.user,
            softprefix_val=args.softprefix_val,
            timeout_total_s=180.0,
        )

        print("\n========== LLM OUTPUT ==========")
        print(llm_text)
        print("================================\n")

    finally:
        cli.close()

    # --- TTS ---
    # Note: OpenAI-compatible API supports /v1/models and /v1/chat/completions etc :contentReference[oaicite:14]{index=14}
    print(f"[INFO] TTS model: {args.tts_model}")
    print(f"[INFO] Writing wav: {raw_path}")
    tts_generate_wav(
        base_url=args.openai_base,
        model_id=args.tts_model,
        text=llm_text,
        out_wav_path=raw_path,
        speed=args.tts_speed,
    )

    print(f"[INFO] Converting for tinyplay: {play_path}")
    ffmpeg_convert_for_tinyplay(
        in_wav=raw_path,
        out_wav=play_path,
        ar_hz=32000,
        channels=2,
        sample_fmt="s16",
    )

    if args.no_play:
        print("[INFO] --no-play specified. Done.")
        return 0

    print(f"[INFO] tinyplay: card={args.tinyplay_card}, device={args.tinyplay_device}")
    tinyplay_play(play_path, card=args.tinyplay_card, device=args.tinyplay_device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
