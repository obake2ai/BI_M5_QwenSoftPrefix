#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
M5Stack Module LLM (Linux) integration test:
- LLM via StackFlow llm-sys TCP socket (default 127.0.0.1:10001)
  * Frame: 10-byte length header + UTF-8 JSON body (TCP framing)
- TTS via OpenAI-compatible API (default http://127.0.0.1:8000/v1)
  * POST /v1/audio/speech returns audio (wav)

Fix for MemoryError:
- Robust framed receiver with:
  * max frame size guard
  * buffered resync (shift by 1 byte until a valid header+JSON body is found)
  * recv() chunk size cap
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, List


# -----------------------------
# Robust TCP framing (llm-sys)
# -----------------------------

DEFAULT_MAX_FRAME_BYTES = 8 * 1024 * 1024        # 8MB: plenty for lsmode etc, prevents runaway
DEFAULT_MAX_BUFFER_BYTES = 16 * 1024 * 1024      # buffer cap for resync safety
SOCK_RECV_CHUNK = 4096                           # per recv() read size


def _parse_len10_header(hdr10: bytes, max_frame_bytes: int) -> Optional[int]:
    """
    Parse 10-byte ASCII length header.
    Some firmwares may pad with spaces/NUL; accept strip() variants.
    Return None if invalid/out-of-range.
    """
    try:
        s = hdr10.decode("utf-8")
    except UnicodeDecodeError:
        return None

    s2 = s.strip("\x00 \r\n\t")
    if not s2.isdigit():
        return None

    try:
        n = int(s2)
    except ValueError:
        return None

    if n < 0 or n > max_frame_bytes:
        return None

    return n


@dataclass
class Len10JsonFramedSocket:
    """
    Buffered framed JSON receiver:
    - keeps rxbuf across reads
    - tries to resync by shifting 1 byte when header/body is not plausible
    """
    sock: socket.socket
    max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES
    max_buffer_bytes: int = DEFAULT_MAX_BUFFER_BYTES
    debug: bool = False
    rxbuf: bytearray = field(default_factory=bytearray)

    def send_json(self, obj: Dict[str, Any]) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        header = f"{len(body):010d}".encode("utf-8")
        self.sock.sendall(header + body)

    def recv_json(self, timeout_s: float) -> Dict[str, Any]:
        """
        Receive one JSON object (dict) framed as len10 + json.
        Raises TimeoutError/EOFError on failure.
        """
        deadline = time.time() + timeout_s

        while True:
            parsed = self._try_parse_one()
            if parsed is not None:
                return parsed

            remain = deadline - time.time()
            if remain <= 0:
                raise TimeoutError("Timed out waiting for a framed JSON response")

            self.sock.settimeout(min(2.0, remain))
            chunk = self.sock.recv(SOCK_RECV_CHUNK)
            if not chunk:
                raise EOFError("socket closed")

            self.rxbuf.extend(chunk)

            # Prevent runaway buffer growth if something goes very wrong
            if len(self.rxbuf) > self.max_buffer_bytes:
                drop = len(self.rxbuf) - self.max_buffer_bytes
                if self.debug:
                    print(f"[DEBUG] rxbuf overflow: drop {drop} bytes", file=sys.stderr)
                del self.rxbuf[:drop]

    def _try_parse_one(self) -> Optional[Dict[str, Any]]:
        """
        Try to parse one frame from rxbuf.
        If frame candidate fails (bad header / bad JSON), shift by 1 byte and retry.
        Returns dict on success, None if not enough bytes.
        """
        while len(self.rxbuf) >= 10:
            hdr = bytes(self.rxbuf[:10])
            n = _parse_len10_header(hdr, self.max_frame_bytes)
            if n is None:
                if self.debug:
                    # show a compact view of the bad header
                    try:
                        hs = hdr.decode("utf-8", errors="replace")
                    except Exception:
                        hs = repr(hdr)
                    print(f"[DEBUG] bad header -> shift 1 byte | hdr={hs!r} raw={hdr!r}", file=sys.stderr)
                del self.rxbuf[:1]
                continue

            if len(self.rxbuf) < 10 + n:
                return None  # need more data

            body = bytes(self.rxbuf[10:10 + n])

            # Try JSON decode WITHOUT consuming; on failure, resync by shifting 1 byte.
            try:
                obj = json.loads(body.decode("utf-8"))
            except Exception as e:
                if self.debug:
                    preview = body[:80]
                    print(f"[DEBUG] JSON decode failed -> shift 1 byte | n={n} err={e} body[:80]={preview!r}",
                          file=sys.stderr)
                del self.rxbuf[:1]
                continue

            if not isinstance(obj, dict):
                # Protocol should be an object; if not, wrap it (keeps robustness).
                obj = {"_json": obj}

            # Consume frame on success
            del self.rxbuf[:10 + n]
            return obj

        return None


# -----------------------------
# StackFlow llm-sys client
# -----------------------------

@dataclass
class LLMSetupConfig:
    model: str
    system_prompt: str
    response_format: str = "llm.utf-8"         # non-stream output
    input_format: str = "llm.utf-8.stream"     # stream input
    max_token_len: int = 256
    enoutput: bool = False
    enkws: bool = False


class StackFlowClient:
    def __init__(
        self,
        host: str,
        port: int,
        timeout_s: float = 5.0,
        max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES,
        debug_frame: bool = False,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self.max_frame_bytes = max_frame_bytes
        self.debug_frame = debug_frame

        self.sock: Optional[socket.socket] = None
        self.fsock: Optional[Len10JsonFramedSocket] = None

    def connect(self) -> None:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(self.timeout_s)
        # reduce latency for small frames
        try:
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except Exception:
            pass
        s.connect((self.host, self.port))
        self.sock = s
        self.fsock = Len10JsonFramedSocket(
            sock=s,
            max_frame_bytes=self.max_frame_bytes,
            debug=self.debug_frame,
        )

    def close(self) -> None:
        if self.sock:
            try:
                self.sock.close()
            finally:
                self.sock = None
                self.fsock = None

    def _must_fsock(self) -> Len10JsonFramedSocket:
        if not self.fsock:
            raise RuntimeError("Not connected")
        return self.fsock

    def _recv_until(self, predicate, timeout_total_s: float, per_read_timeout_s: float = 5.0) -> Dict[str, Any]:
        deadline = time.time() + timeout_total_s
        while True:
            remain = deadline - time.time()
            if remain <= 0:
                raise TimeoutError("Timed out waiting for expected response")
            msg = self._must_fsock().recv_json(timeout_s=min(per_read_timeout_s, remain))
            if predicate(msg):
                return msg

    def sys_ping(self) -> Dict[str, Any]:
        req = {
            "request_id": "sys_ping",
            "work_id": "sys",
            "action": "ping",
            "object": "None",
            "data": "None",
        }
        self._must_fsock().send_json(req)
        return self._recv_until(
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
        self._must_fsock().send_json(req)
        resp = self._recv_until(
            lambda m: m.get("request_id") == "sys_lsmode" and m.get("work_id") == "sys",
            timeout_total_s=10.0,
        )
        data = resp.get("data", [])
        if isinstance(data, list):
            return data
        return []

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
        self._must_fsock().send_json(req)

        resp = self._recv_until(
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
        if softprefix_val is not None:
            v = float(softprefix_val)
            data_obj.update({
                "soft_prefix_val": v,
                "softprefix_val": v,
                "prefix_val": v,
                "test_embed_val": v,
                "soft_prefix": {"val": v},
                "enable_soft_prefix": True,
            })

        req = {
            "request_id": req_id,
            "work_id": work_id,
            "action": "inference",
            "object": "llm.utf-8.stream",
            "data": data_obj,
        }

        self._must_fsock().send_json(req)

        out_chunks: List[str] = []
        deadline = time.time() + timeout_total_s

        while True:
            remain = deadline - time.time()
            if remain <= 0:
                raise TimeoutError("Timed out waiting for LLM inference output")

            msg = self._must_fsock().recv_json(timeout_s=min(5.0, remain))

            if msg.get("work_id") != work_id:
                # other unit messages; ignore
                continue

            err = msg.get("error", {}) or {}
            if err.get("code", 0) != 0:
                raise RuntimeError(f"LLM inference error: {err} / msg={msg}")

            data = msg.get("data")

            # Non-stream output
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

    ap.add_argument("--llm-host", default="127.0.0.1", help="llm-sys host (on-device: 127.0.0.1)")
    ap.add_argument("--llm-port", type=int, default=10001, help="llm-sys TCP port (default 10001)")
    ap.add_argument("--openai-base", default="http://127.0.0.1:8000/v1", help="OpenAI-compatible base_url (default on-device)")

    ap.add_argument("--preset", choices=["qwen", "tinyswallow"], default="qwen", help="LLM preset model selector")
    ap.add_argument("--llm-model", default="", help="Override LLM model name explicitly (if empty, auto-pick via sys.lsmode)")

    ap.add_argument("--system-prompt", default="あなたは親切で簡潔な日本語アシスタントです。短く自然な日本語で答えてください。", help="System prompt used at llm.setup")
    ap.add_argument("--user", default="こんにちは。自己紹介を一文でお願いします。", help="User input text")

    ap.add_argument("--max-token-len", type=int, default=256, help="max_token_len for llm.setup")

    ap.add_argument("--softprefix-val", type=float, default=None, help="Enable SoftPrefix injection by sending this float value (e.g. 0.01). Omit for baseline.")

    ap.add_argument("--tts-model", default="melotts-ja-jp", help="TTS model id (e.g. melotts-ja-jp)")
    ap.add_argument("--tts-speed", type=float, default=1.0, help="TTS speed (default 1.0)")

    ap.add_argument("--out-raw", default="/tmp/llm_tts_raw.wav", help="Output raw wav path")
    ap.add_argument("--out-play", default="/tmp/llm_tts_32k_stereo_s16.wav", help="Output converted wav path for tinyplay")

    ap.add_argument("--no-play", action="store_true", help="Do not run tinyplay")
    ap.add_argument("--tinyplay-card", type=int, default=0, help="tinyplay card number (default 0)")
    ap.add_argument("--tinyplay-device", type=int, default=1, help="tinyplay device number (default 1)")

    # NEW: frame safety / debug
    ap.add_argument("--max-frame-bytes", type=int, default=DEFAULT_MAX_FRAME_BYTES, help="Max TCP frame size for llm-sys (default 8MB)")
    ap.add_argument("--debug-frame", action="store_true", help="Enable debug logs for TCP frame resync")

    args = ap.parse_args()

    raw_path = Path(args.out_raw)
    play_path = Path(args.out_play)

    cli = StackFlowClient(
        args.llm_host,
        args.llm_port,
        timeout_s=5.0,
        max_frame_bytes=int(args.max_frame_bytes),
        debug_frame=bool(args.debug_frame),
    )

    llm_text = ""
    try:
        cli.connect()
        ping = cli.sys_ping()
        if (ping.get("error") or {}).get("code", 0) != 0:
            print(f"[WARN] sys.ping error: {ping}", file=sys.stderr)

        lsmode = cli.sys_lsmode()

        model_name = args.llm_model.strip()
        if not model_name:
            picked = pick_model_by_preset(lsmode, args.preset)
            if picked:
                model_name = picked
            else:
                # fallback defaults
                if args.preset == "qwen":
                    model_name = "qwen2.5-0.5b"
                else:
                    model_name = "TinySwallow-1.5B-Instruct-ax630c"

        print(f"[INFO] Using LLM model: {model_name}")
        if args.softprefix_val is None:
            print("[INFO] SoftPrefix: DISABLED (baseline)")
        else:
            print(f"[INFO] SoftPrefix: ENABLED val={args.softprefix_val}")

        cfg = LLMSetupConfig(
            model=model_name,
            system_prompt=args.system_prompt,
            response_format="llm.utf-8",
            input_format="llm.utf-8.stream",
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
