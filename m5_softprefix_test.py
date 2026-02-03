#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import socket
import json
import base64
import struct
import time
import difflib
import hashlib
import argparse
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path


# ========= 設定 =========
HOST = "127.0.0.1"
PORT = 10001

MODEL = "qwen2.5-0.5B-prefill-20e"
SYSTEM_PROMPT = "あなたは優秀な日本語アシスタントです。"

# 比較したいユーザー入力
USER_PROMPT = "こんにちは。植物に関する詩を描いて。"

# 非streamで返す（重要）
RESPONSE_FORMAT = "llm.utf-8"

# soft_prefix を渡すため、入力は stream 形式で送る
INPUT_OBJECT_FOR_INFER = "llm.utf-8.stream"

# まず短め推奨（比較が目的なら 128〜256 で十分）
MAX_TOKEN_LEN = 128

# soft_prefix の形状
P = 1         # prefix token数（1でOK）
H = 896       # tokens_embed_size（あなたの環境に合わせる）

# 値を小→大へ（必要なら増減してOK）
# ※ baseline（prefix無し）とは別に「prefixあり val=0.0」も入れて、"prefixを入れたこと自体の影響" を見られるようにする
VALS = [0.0, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1.0, 2.0]

SOCK_TIMEOUT_SEC = 240

# 結果保存ファイル
OUT_JSON = "compare_results.json"
# ========= /設定 =========


def f32_to_bf16_u16(x: float) -> int:
    """float32 -> bf16 (truncate) -> u16"""
    u32 = struct.unpack("<I", struct.pack("<f", x))[0]
    return (u32 >> 16) & 0xFFFF


def make_soft_prefix_b64_constant(P: int, H: int, val: float) -> str:
    """bf16 little-endian u16 を P*H 個並べて base64"""
    u16 = f32_to_bf16_u16(val)
    raw = struct.pack("<H", u16) * (P * H)
    return base64.b64encode(raw).decode("ascii")


def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def preview(s: str, n: int = 120) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    one = s.replace("\n", "\\n")
    return (one[:n] + "…") if len(one) > n else one


class LLMClient:
    def __init__(self, host: str, port: int, timeout: int):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock: Optional[socket.socket] = None
        self.r = None
        self.w = None

    def __enter__(self):
        self.sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        self.sock.settimeout(self.timeout)
        # readlineで取りこぼさない
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

    def read_json_line(self) -> Dict[str, Any]:
        line = self.r.readline()
        if line == "":
            raise RuntimeError("socket closed (EOF)")
        line = line.strip()
        if not line:
            return {}
        return json.loads(line)

    def wait_response(self, request_id: str, max_wait_sec: int = 240) -> Dict[str, Any]:
        """同一接続内で request_id に一致する応答を待つ"""
        t0 = time.time()
        while True:
            if time.time() - t0 > max_wait_sec:
                raise TimeoutError(f"timeout waiting response for request_id={request_id}")

            msg = self.read_json_line()
            if not msg:
                continue

            if msg.get("request_id") == request_id:
                return msg

            # デバッグしたい場合はここを有効化
            # print("[SKIP]", msg)

    def setup(self) -> str:
        req_id = f"setup_{int(time.time()*1000)}"
        setup_req = {
            "request_id": req_id,
            "work_id": "llm",
            "action": "setup",
            "object": "llm.setup",
            "data": {
                "model": MODEL,
                "response_format": RESPONSE_FORMAT,   # ★非stream
                "input": INPUT_OBJECT_FOR_INFER,      # ★入力は stream
                "enoutput": True,
                "max_token_len": MAX_TOKEN_LEN,
                "prompt": SYSTEM_PROMPT,
            },
        }
        self.send_json(setup_req)
        resp = self.wait_response(req_id, max_wait_sec=60)
        err = resp.get("error", {})
        if isinstance(err, dict) and err.get("code", 0) != 0:
            raise RuntimeError(f"setup failed: {err} full={resp}")
        work_id = resp.get("work_id", "llm")
        return work_id

    def inference(
        self,
        work_id: str,
        user_prompt: str,
        soft_prefix_b64: Optional[str] = None,
        soft_prefix_len: int = 0,
    ) -> Tuple[str, float, Dict[str, Any]]:
        req_id = f"infer_{int(time.time()*1000)}"
        # 入力は stream 形式の data を送る（soft_prefixもここに載せる）
        data_obj: Dict[str, Any] = {"delta": user_prompt, "index": 0, "finish": True}
        if soft_prefix_b64 is not None:
            data_obj["soft_prefix"] = {"len": int(soft_prefix_len), "data_b64": soft_prefix_b64}

        infer_req = {
            "request_id": req_id,
            "work_id": work_id,
            "action": "inference",
            "object": INPUT_OBJECT_FOR_INFER,  # ★入力は stream
            "data": data_obj,
        }

        t0 = time.time()
        self.send_json(infer_req)

        # response_format は非stream なので、1発で全文が返る想定
        resp = self.wait_response(req_id, max_wait_sec=SOCK_TIMEOUT_SEC)
        dt = time.time() - t0

        err = resp.get("error", {})
        if isinstance(err, dict) and err.get("code", 0) != 0:
            raise RuntimeError(f"inference failed: {err} full={resp}")

        out = resp.get("data", "")
        if not isinstance(out, str):
            # 念のため
            out = json.dumps(out, ensure_ascii=False)
        return out, dt, resp

    def exit(self, work_id: str) -> None:
        req_id = f"exit_{int(time.time()*1000)}"
        self.send_json({"request_id": req_id, "work_id": work_id, "action": "exit"})
        # exit 応答は来たり来なかったりするので待たない


@dataclass
class CaseResult:
    name: str
    has_prefix: bool
    P: int
    H: int
    val: Optional[float]
    elapsed_sec: float
    out_len: int
    out_sha1: str
    out_preview: str
    out_text: str


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def unified_diff_head(a: str, b: str, n_lines: int = 80) -> str:
    """差分を長くしすぎないため、先頭 n_lines だけの unified diff"""
    a_lines = a.splitlines(keepends=True)[:n_lines]
    b_lines = b.splitlines(keepends=True)[:n_lines]
    diff = difflib.unified_diff(a_lines, b_lines, fromfile="baseline", tofile="case", lineterm="")
    return "\n".join(list(diff)[:200])  # さらに上限


def main():
    ap = argparse.ArgumentParser(description="LLM (SoftPrefix) -> TTS wav -> ffmpeg -> tinyplay test on M5Stack Module LLM")
    ap.add_argument("--openai-base", default="http://127.0.0.1:8000/v1", help="OpenAI-compatible base_url (default on-device)")
    ap.add_argument("--tts-model", default="melotts-ja-jp", help="TTS model id (e.g. melotts-ja-jp)")
    ap.add_argument("--tts-speed", type=float, default=1.0, help="TTS speed (default 1.0)")
    ap.add_argument("--no-play", action="store_true", help="Do not run tinyplay")

    ap.add_argument("--out-raw", default="/tmp/llm_tts_raw.wav", help="Output raw wav path")
    ap.add_argument("--out-play", default="/tmp/llm_tts_32k_stereo_s16.wav", help="Output converted wav path for tinyplay")
    raw_path = Path(args.out_raw)
    play_path = Path(args.out_play)

    results: List[CaseResult] = []

    with LLMClient(HOST, PORT, SOCK_TIMEOUT_SEC) as cli:
        work_id = cli.setup()
        print("[SETUP OK] work_id:", work_id)

        # ---- baseline ----
        base_text, base_dt, _ = cli.inference(work_id, USER_PROMPT, soft_prefix_b64=None)
        baseline = CaseResult(
            name="baseline(no_prefix)",
            has_prefix=False,
            P=0,
            H=H,
            val=None,
            elapsed_sec=base_dt,
            out_len=len(base_text),
            out_sha1=sha1_text(base_text),
            out_preview=preview(base_text),
            out_text=base_text,
        )
        results.append(baseline)

        print("\n=== BASELINE ===")
        print(f"time={base_dt:.2f}s len={baseline.out_len} sha1={baseline.out_sha1}")
        print(baseline.out_text)
        print("==============\n")

        # ---- prefix cases ----
        for v in VALS:
            sp_b64 = make_soft_prefix_b64_constant(P, H, v)
            out, dt, _ = cli.inference(work_id, USER_PROMPT, soft_prefix_b64=sp_b64, soft_prefix_len=P)

            cr = CaseResult(
                name=f"prefix(P={P},val={v})",
                has_prefix=True,
                P=P,
                H=H,
                val=v,
                elapsed_sec=dt,
                out_len=len(out),
                out_sha1=sha1_text(out),
                out_preview=preview(out),
                out_text=out,
            )
            results.append(cr)

            sim = similarity(baseline.out_text, out)
            print(f"=== CASE val={v} === time={dt:.2f}s len={cr.out_len} sim={sim:.3f} sha1={cr.out_sha1}")
            print("preview:", cr.out_preview)
            # diffも少しだけ出す（長くなりすぎるので先頭だけ）
            d = unified_diff_head(baseline.out_text, out)
            if d.strip():
                print("--- diff(head) ---")
                print(d)
            else:
                print("--- diff(head) --- (no diff in head)")
            print("")

            # --- TTS ---
            if not args.no_play:
                print(f"[INFO] TTS model: {args.tts_model}")
                print(f"[INFO] Writing wav: {raw_path}")
                tts_generate_wav(
                    base_url=args.openai_base,
                    model_id=args.tts_model,
                    text=out,
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
                print(f"[INFO] tinyplay: card={args.tinyplay_card}, device={args.tinyplay_device}")
                tinyplay_play(play_path, card=args.tinyplay_card, device=args.tinyplay_device)

            else:
                print("[INFO] --no-play specified. Done.")
                return 0


        # 終了
        cli.exit(work_id)

    # 保存（全文も入るのでサイズ注意）
    payload = {
        "meta": {
            "host": HOST,
            "port": PORT,
            "model": MODEL,
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt": USER_PROMPT,
            "response_format": RESPONSE_FORMAT,
            "input_object_for_infer": INPUT_OBJECT_FOR_INFER,
            "max_token_len": MAX_TOKEN_LEN,
            "P": P,
            "H": H,
            "vals": VALS,
            "saved_at_unix": int(time.time()),
        },
        "results": [asdict(r) for r in results],
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # ざっくりサマリ
    baseline_sha = results[0].out_sha1
    changed = sum(1 for r in results[1:] if r.out_sha1 != baseline_sha)
    print(f"\nSaved: {OUT_JSON}")
    print(f"Changed cases vs baseline: {changed}/{len(results)-1}")
    if changed == 0:
        print("NOTE: 全ケースが baseline と同一です。soft_prefix がサーバ側で適用されていない可能性があります。")


if __name__ == "__main__":
    main()
