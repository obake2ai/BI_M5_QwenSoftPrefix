#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import base64
import json
import random
import re
import socket
import struct
import subprocess
import time
from typing import Any, Dict, List, Optional


# ========= SoftPrefix shape（あなたの環境に合わせる）=========
P_DEFAULT = 1
H_DEFAULT = 896

# ========= LLM I/O =========
LLM_RESPONSE_FORMAT = "llm.utf-8"          # 非streamで返す
LLM_INPUT_OBJECT = "llm.utf-8.stream"      # soft_prefixを載せるため入力はstream

# ========= TTS I/O (StackFlow unit) =========
TTS_INPUT_OBJECT = "tts.utf-8"             # lsmode上の input_type

# ========= OpenAI-compatible TTS (melotts-ja-jp) =========
OPENAI_TTS_DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"   # on device (localhost)
OPENAI_TTS_AUDIO_PATH = "/audio/speech"


def f32_to_bf16_u16(x: float) -> int:
    """float32 -> bf16 (truncate) -> u16"""
    u32 = struct.unpack("<I", struct.pack("<f", x))[0]
    return (u32 >> 16) & 0xFFFF


def make_soft_prefix_b64_constant(p: int, h: int, val: float) -> str:
    """bf16 little-endian u16 を P*H 個並べて base64"""
    u16 = f32_to_bf16_u16(val)
    raw = struct.pack("<H", u16) * (p * h)
    return base64.b64encode(raw).decode("ascii")


def sanitize_llm_output_to_2_3_words(text: str) -> str:
    """
    LLM出力から「2〜3単語（半角スペース区切り）」を抽出
    """
    if not text:
        return ""

    t = text.strip()
    # 先頭行だけ使う（説明が続くのを抑止）
    t = t.splitlines()[0].strip()

    # かぎ括弧/引用符除去
    t = t.replace("「", "").replace("」", "").replace("『", "").replace("』", "")
    t = t.replace('"', "").replace("'", "")

    # 日本語/英数/スペース/長音など以外をスペース化
    t = re.sub(r"[^0-9A-Za-z\u3040-\u30FF\u4E00-\u9FFFー々\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()

    toks = [x for x in t.split(" ") if x]
    if len(toks) >= 3:
        return " ".join(toks[:3])
    if len(toks) == 2:
        return " ".join(toks)
    if len(toks) == 1:
        return toks[0]
    return ""


def build_user_prompt(prev_phrase: str) -> str:
    """
    前回の2〜3語を条件に、次の2〜3語を生成させる
    """
    return (
        "次の条件で、日本語の短いフレーズを生成してください。\n"
        f"前回: {prev_phrase}\n\n"
        "条件:\n"
        "- 出力は必ず「2語」または「3語」\n"
        "- 単語は半角スペースで区切る\n"
        "- 説明文、記号、句読点、引用符は付けない\n"
        "- できれば名詞や形容詞中心\n\n"
        "出力例: 春 風 芽\n"
        "出力:"
    )


def _norm_sentence_end(text: str) -> str:
    """TTS向け：末尾に句点が無ければ付与"""
    if text and text[-1] not in ".。!?？！":
        return text + "。"
    return text


def run_ssh_bash_script(
    host: str,
    ssh_user: str,
    ssh_port: int,
    identity_file: Optional[str],
    ssh_options: List[str],
    script: str,
    timeout_sec: float,
    enable_mux: bool,
) -> None:
    """
    Mac側から `ssh user@host bash -s` で複数行スクリプトを流し込んで実行する
    - JSONや日本語を含むため、コマンド1行渡しより安全
    - openai TTS を毎回呼ぶので、デフォルトで SSH multiplex (ControlMaster) を使えるようにする
    """
    cmd: List[str] = ["ssh", "-p", str(ssh_port)]

    # Multiplexing（同一ホストへのSSHを高速化 + パスワード入力を1回にできる可能性）
    if enable_mux:
        # %r/%h/%p は ssh が展開。ControlPathはローカル(Mac)側のパス。
        cmd += [
            "-o", "ControlMaster=auto",
            "-o", "ControlPersist=60s",
            "-o", "ControlPath=/tmp/ssh-m5stack-llm-%r@%h-%p",
        ]

    # keep alive（Wi-Fi等で切れにくくする）
    cmd += [
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=3",
    ]

    # user provided -o options
    for opt in ssh_options:
        cmd += ["-o", opt]

    if identity_file:
        cmd += ["-i", identity_file]

    cmd += [f"{ssh_user}@{host}", "bash", "-s"]

    subprocess.run(
        cmd,
        input=script.encode("utf-8"),
        check=True,
        timeout=timeout_sec,
    )


def melotts_openai_wav_ffmpeg_tinyplay_over_ssh(
    *,
    host: str,
    ssh_user: str,
    ssh_port: int,
    identity_file: Optional[str],
    ssh_options: List[str],
    enable_mux: bool,
    openai_base_url: str,
    model: str,
    text: str,
    speed: float,
    target_rate: int,
    target_channels: int,
    playcard: int,
    playdevice: int,
    timeout_sec: float,
) -> None:
    """
    device側で：
      1) OpenAI互換API /v1/audio/speech で wav を生成して /tmp に保存
      2) ffmpeg で (32000Hz / 2ch / s16) に変換（logo.wav と同条件に揃える想定）
      3) tinyplay -D<card> -d<device> で再生
    をSSH経由で実行する。

    ※ text は日本語を含むので、payloadは base64 にしてリモートpythonで復元する。
    """
    text = _norm_sentence_end(text)

    rid = f"{time.time_ns()}_{random.randint(0, 9999):04d}"
    raw_path = f"/tmp/melotts_raw_{rid}.wav"
    out_path = f"/tmp/melotts_play_{rid}.wav"

    tts_url = openai_base_url.rstrip("/") + OPENAI_TTS_AUDIO_PATH

    payload = {
        "model": model,
        "input": text,
        "response_format": "wav",
        "speed": float(speed),
    }
    payload_b64 = base64.b64encode(json.dumps(payload, ensure_ascii=False).encode("utf-8")).decode("ascii")

    # bashスクリプト（リモート側）
    script = f"""set -euo pipefail
RAW="{raw_path}"
OUT="{out_path}"

# ツール存在チェック（無ければ入れる）
command -v python3 >/dev/null 2>&1 || (echo "python3 not found" >&2; exit 2)
command -v tinyplay >/dev/null 2>&1 || (echo "tinyplay not found" >&2; exit 3)
command -v ffmpeg >/dev/null 2>&1 || (apt update && apt install -y ffmpeg)

python3 - <<'PY'
import base64, json, urllib.request
payload = json.loads(base64.b64decode("{payload_b64}").decode("utf-8"))
data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
req = urllib.request.Request(
    "{tts_url}",
    data=data,
    headers={{"Content-Type": "application/json", "Authorization": "Bearer sk-"}},
)
with urllib.request.urlopen(req, timeout=60) as r:
    audio = r.read()
open("{raw_path}", "wb").write(audio)
PY

# ロゴ音と同等条件に寄せる（あなたの環境ではこれで再生OKだった）
ffmpeg -y -hide_banner -loglevel error -i "$RAW" -ar {int(target_rate)} -ac {int(target_channels)} -sample_fmt s16 "$OUT"

tinyplay -D{int(playcard)} -d{int(playdevice)} "$OUT"

rm -f "$RAW" "$OUT"
"""

    run_ssh_bash_script(
        host=host,
        ssh_user=ssh_user,
        ssh_port=ssh_port,
        identity_file=identity_file,
        ssh_options=ssh_options,
        script=script,
        timeout_sec=timeout_sec,
        enable_mux=enable_mux,
    )


class StackFlowClient:
    """
    Module LLM (llm-sys) TCP JSONL クライアント
    - readline/makefile はタイムアウト後に壊れやすいので recv+自前split
    """
    def __init__(self, host: str, port: int, connect_timeout: float = 5.0, recv_timeout: float = 0.5):
        self.host = host
        self.port = port
        self.connect_timeout = connect_timeout
        self.recv_timeout = recv_timeout

        self.sock: Optional[socket.socket] = None
        self._buf = b""
        self._rid_counter = 0

    def connect(self) -> None:
        self.close()
        self.sock = socket.create_connection((self.host, self.port), timeout=self.connect_timeout)
        self.sock.settimeout(self.recv_timeout)
        self._buf = b""

    def close(self) -> None:
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
        self.sock = None
        self._buf = b""

    def _next_rid(self, prefix: str) -> str:
        self._rid_counter += 1
        return f"{prefix}_{time.time_ns()}_{self._rid_counter}"

    def send_json(self, obj: Dict[str, Any]) -> None:
        if not self.sock:
            raise RuntimeError("socket is not connected")
        data = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")
        self.sock.sendall(data)

    def recv_json(self) -> Optional[Dict[str, Any]]:
        if not self.sock:
            raise RuntimeError("socket is not connected")

        # 既に1行溜まっているなら返す
        if b"\n" in self._buf:
            line, self._buf = self._buf.split(b"\n", 1)
            line = line.strip()
            if not line:
                return None
            try:
                return json.loads(line.decode("utf-8", errors="replace"))
            except Exception:
                # reboot時の文字列等が来ることがあるので無視
                return None

        try:
            chunk = self.sock.recv(4096)
        except (socket.timeout, TimeoutError):
            return None

        if not chunk:
            raise RuntimeError("socket closed (EOF)")

        self._buf += chunk
        return None

    def wait_response(self, request_id: str, timeout_sec: float, label: str = "") -> Dict[str, Any]:
        t0 = time.time()
        last_print = 0.0
        while True:
            if time.time() - t0 > timeout_sec:
                raise TimeoutError(f"timeout waiting response for request_id={request_id}")

            msg = self.recv_json()
            if msg is None:
                if label and (time.time() - last_print) > 10.0:
                    last_print = time.time()
                    print(f"[WAIT] {label}: {time.time() - t0:.1f}s ...")
                continue

            # 対象外でもエラーは見える化
            err = msg.get("error", {})
            if isinstance(err, dict) and err.get("code", 0) != 0 and msg.get("request_id") != request_id:
                print("[RECV-ERR]", msg)

            if msg.get("request_id") == request_id:
                err2 = msg.get("error", {})
                if isinstance(err2, dict) and err2.get("code", 0) != 0:
                    print("[RECV-ERR]", msg)
                return msg

    # ---- sys ----
    def sys_lsmode(self) -> List[Dict[str, Any]]:
        rid = self._next_rid("lsmode")
        self.send_json({"request_id": rid, "work_id": "sys", "action": "lsmode"})
        resp = self.wait_response(rid, timeout_sec=10.0, label="sys.lsmode")
        data = resp.get("data", [])
        return data if isinstance(data, list) else []

    def sys_ping(self) -> bool:
        rid = self._next_rid("ping")
        self.send_json({"request_id": rid, "work_id": "sys", "action": "ping"})
        resp = self.wait_response(rid, timeout_sec=3.0, label="")
        err = resp.get("error", {})
        return isinstance(err, dict) and err.get("code", 0) == 0

    def sys_reset(self) -> None:
        rid = self._next_rid("reset")
        try:
            self.send_json({"request_id": rid, "work_id": "sys", "action": "reset"})
            _ = self.wait_response(rid, timeout_sec=2.0, label="")
        except Exception:
            pass

    def reset_and_wait_ready(self, max_wait_sec: float = 30.0) -> None:
        print("[INFO] sys.reset on start...")
        try:
            self.sys_reset()
        finally:
            self.close()

        t0 = time.time()
        while True:
            if time.time() - t0 > max_wait_sec:
                raise TimeoutError("reset done but system not ready (ping timeout)")

            try:
                self.connect()
                if self.sys_ping():
                    time.sleep(1.0)
                    return
            except Exception:
                pass

            self.close()
            time.sleep(0.5)

    # ---- audio ----
    def audio_setup(self, playdevice: int = 1, play_volume: float = 0.7) -> str:
        rid = self._next_rid("audio_setup")
        req = {
            "request_id": rid,
            "work_id": "audio",
            "action": "setup",
            "object": "audio.setup",
            "data": {
                "capcard": 0,
                "capdevice": 0,
                "capVolume": 0.5,
                "playcard": 0,
                "playdevice": int(playdevice),
                "playVolume": float(play_volume),
            },
        }
        self.send_json(req)
        resp = self.wait_response(rid, timeout_sec=20.0, label="audio.setup")
        err = resp.get("error", {})
        if isinstance(err, dict) and err.get("code", 0) != 0:
            raise RuntimeError(f"audio.setup failed: {resp}")
        return str(resp.get("work_id", "audio"))

    def audio_setup_best_effort(self, play_volume: float = 0.7) -> Optional[str]:
        for attempt in range(1, 6):
            for dev in (1, 0):
                try:
                    wid = self.audio_setup(playdevice=dev, play_volume=play_volume)
                    return wid
                except Exception as e:
                    print(f"[WARN] audio.setup failed (attempt {attempt}/5, playdevice={dev}): {repr(e)}")
                    time.sleep(0.8)
        return None

    # ---- llm ----
    def llm_setup(self, model: str, system_prompt: str, max_token_len: int) -> str:
        rid = self._next_rid("setup_llm")
        req = {
            "request_id": rid,
            "work_id": "llm",
            "action": "setup",
            "object": "llm.setup",
            "data": {
                "model": model,
                "response_format": LLM_RESPONSE_FORMAT,
                "input": LLM_INPUT_OBJECT,
                "enoutput": True,
                "max_token_len": int(max_token_len),
                "prompt": system_prompt,
            },
        }
        self.send_json(req)
        resp = self.wait_response(rid, timeout_sec=180.0, label="llm.setup")
        err = resp.get("error", {})
        if isinstance(err, dict) and err.get("code", 0) != 0:
            raise RuntimeError(f"llm.setup failed: {resp}")
        return str(resp.get("work_id", "llm"))

    def llm_setup_with_retry(self, model: str, system_prompt: str, max_token_len: int) -> str:
        last_err: Optional[Exception] = None
        for attempt in range(1, 3):
            try:
                return self.llm_setup(model, system_prompt, max_token_len)
            except Exception as e:
                last_err = e
                print(f"[WARN] llm.setup failed (attempt {attempt}/2). sys.reset and retry... err={repr(e)}")
                self.reset_and_wait_ready(max_wait_sec=40.0)
        raise RuntimeError(f"llm.setup retry exhausted: {repr(last_err)}")

    def llm_inference(self, llm_work_id: str, user_prompt: str,
                      soft_prefix_b64: Optional[str], soft_prefix_len: int,
                      timeout_sec: float) -> str:
        rid = self._next_rid("infer_llm")
        data_obj: Dict[str, Any] = {"delta": user_prompt, "index": 0, "finish": True}
        if soft_prefix_b64 is not None:
            data_obj["soft_prefix"] = {"len": int(soft_prefix_len), "data_b64": soft_prefix_b64}

        req = {
            "request_id": rid,
            "work_id": llm_work_id,
            "action": "inference",
            "object": LLM_INPUT_OBJECT,
            "data": data_obj,
        }
        self.send_json(req)
        resp = self.wait_response(rid, timeout_sec=timeout_sec, label="llm.inference")
        err = resp.get("error", {})
        if isinstance(err, dict) and err.get("code", 0) != 0:
            raise RuntimeError(f"llm.inference failed: {resp}")

        out = resp.get("data", "")
        if not isinstance(out, str):
            out = json.dumps(out, ensure_ascii=False)
        return out

    # ---- tts (StackFlow unit) ----
    def tts_setup(self, model: str, response_format: str) -> str:
        rid = self._next_rid("setup_tts")
        req = {
            "request_id": rid,
            "work_id": "tts",
            "action": "setup",
            "object": "tts.setup",
            "data": {
                "model": model,
                "response_format": response_format,
                "input": TTS_INPUT_OBJECT,
                "enoutput": True,
                "enkws": False,
            },
        }
        self.send_json(req)
        resp = self.wait_response(rid, timeout_sec=60.0, label="tts.setup")
        err = resp.get("error", {})
        if isinstance(err, dict) and err.get("code", 0) != 0:
            raise RuntimeError(f"tts.setup failed: {resp}")
        return str(resp.get("work_id", "tts"))

    def tts_inference(self, tts_work_id: str, text: str, timeout_sec: float = 8.0) -> None:
        text = _norm_sentence_end(text)
        rid = self._next_rid("infer_tts")
        req = {
            "request_id": rid,
            "work_id": tts_work_id,
            "action": "inference",
            "object": "tts.utf-8",
            "data": text,
        }
        self.send_json(req)
        try:
            _ = self.wait_response(rid, timeout_sec=timeout_sec, label="tts.inference")
        except TimeoutError:
            print("[WARN] tts.inference ack timeout (audio may still be playing). continue...")

    def exit_work(self, work_id: str) -> None:
        rid = self._next_rid("exit")
        try:
            self.send_json({"request_id": rid, "work_id": work_id, "action": "exit"})
        except Exception:
            pass


def pick_llm_mode(lsmode: List[Dict[str, Any]], fallback: str) -> str:
    for x in lsmode:
        if x.get("type") == "llm" and isinstance(x.get("mode"), str):
            return x["mode"]
    for x in lsmode:
        if x.get("type") == "llm" and isinstance(x.get("model"), str):
            return x["model"]
    return fallback


def pick_tts_response_format(lsmode: List[Dict[str, Any]]) -> str:
    for x in lsmode:
        if x.get("type") == "tts":
            outs = x.get("output_type", [])
            if isinstance(outs, list) and "sys.play.0_1" in outs:
                return "sys.play.0_1"
    return "tts.wav"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="192.168.3.146")
    ap.add_argument("--port", type=int, default=10001)

    ap.add_argument("--sp-strength", type=float, required=True,
                    help="SoftPrefix強度(絶対値)。毎回ランダム符号で±にします。")
    ap.add_argument("--interval", type=float, default=0.3)

    ap.add_argument("--p", type=int, default=P_DEFAULT)
    ap.add_argument("--h", type=int, default=H_DEFAULT)

    ap.add_argument("--max-token-len", type=int, default=24)
    ap.add_argument("--system-prompt", default="あなたは日本語の短いフレーズだけを出力する生成器です。")
    ap.add_argument("--seed", default="春 風", help="最初の『前回フレーズ』（2語以上推奨）")

    # ---- TTS options ----
    ap.add_argument("--tts", action="store_true", help="StackFlowのTTSユニットを使って再生する")
    ap.add_argument("--tts-openai", action="store_true",
                    help="OpenAI互換API(melotts)でWAV生成→ffmpeg変換→tinyplay再生（SSH経由）")

    ap.add_argument("--tts-required", action="store_true",
                    help="TTS再生が失敗したら終了する（デフォルトは生成だけ継続）")
    ap.add_argument("--tts-timeout", type=float, default=8.0)

    ap.add_argument("--audio-volume", type=float, default=0.7)
    ap.add_argument("--no-reset", action="store_true", help="起動時の sys.reset を行わない")

    # ---- OpenAI(TTS) over SSH params ----
    ap.add_argument("--ssh-user", default="root")
    ap.add_argument("--ssh-port", type=int, default=22)
    ap.add_argument("--ssh-identity", default=None, help="ssh -i の鍵ファイル（必要なら）")
    ap.add_argument("--ssh-option", action="append", default=[],
                    help="ssh の追加オプション。例: --ssh-option 'StrictHostKeyChecking=no'（複数可）")
    ap.add_argument("--ssh-no-mux", action="store_true",
                    help="SSH multiplex(ControlMaster)を無効化（デフォルトは有効）")

    ap.add_argument("--tts-openai-base-url", default=OPENAI_TTS_DEFAULT_BASE_URL,
                    help="デバイス側OpenAI互換APIのbase_url（デフォルトはデバイス内 localhost）")
    ap.add_argument("--tts-openai-model", default="melotts-ja-jp")
    ap.add_argument("--tts-openai-speed", type=float, default=1.0)
    ap.add_argument("--tts-openai-target-rate", type=int, default=32000)
    ap.add_argument("--tts-openai-target-ch", type=int, default=2)
    ap.add_argument("--tts-openai-playcard", type=int, default=0)
    ap.add_argument("--tts-openai-playdevice", type=int, default=1)
    ap.add_argument("--tts-openai-timeout", type=float, default=120.0,
                    help="SSH+生成+変換+再生 全体のタイムアウト秒")

    args = ap.parse_args()

    cli = StackFlowClient(args.host, args.port)

    cli.connect()
    if not args.no_reset:
        cli.reset_and_wait_ready(max_wait_sec=40.0)

    lsmode = cli.sys_lsmode()
    llm_mode = pick_llm_mode(lsmode, fallback="qwen2.5-0.5B-prefill-20e")
    print("[lsmode.llm_mode]", llm_mode)

    llm_work_id = cli.llm_setup_with_retry(
        llm_mode,
        system_prompt=args.system_prompt,
        max_token_len=args.max_token_len
    )
    print("[LLM SETUP OK] work_id:", llm_work_id)

    lsmode = cli.sys_lsmode()

    audio_work_id: Optional[str] = None
    tts_work_id: Optional[str] = None
    tts_resp = pick_tts_response_format(lsmode)
    print("[lsmode refreshed] tts_resp =", tts_resp)

    need_audio = bool(args.tts or args.tts_openai)
    if need_audio:
        audio_work_id = cli.audio_setup_best_effort(play_volume=args.audio_volume)
        if audio_work_id:
            print("[AUDIO SETUP OK] work_id:", audio_work_id)
        else:
            print("[WARN] audio.setup failed repeatedly. Playback volume/device may be wrong.")

    if args.tts:
        candidates = [
            "melotts-ja-jp",
            "model-melotts-ja-jp",
        ]
        print("[TTS(StackFlow)] candidates=", candidates, "response_format=", tts_resp)

        last_err: Optional[Exception] = None
        for m in candidates:
            try:
                tts_work_id = cli.tts_setup(model=m, response_format=tts_resp)
                print("[TTS SETUP OK] work_id:", tts_work_id, "model:", m, "resp:", tts_resp)
                last_err = None
                break
            except Exception as e:
                print("[WARN] tts.setup failed for model:", m, "err=", repr(e))
                last_err = e

        if tts_work_id is None and args.tts_required:
            raise RuntimeError(f"TTS required but setup failed: {repr(last_err)}")

        if tts_work_id is None:
            print("[WARN] Japanese TTS setup failed. Continue WITHOUT StackFlow-TTS.")

    prev_phrase = sanitize_llm_output_to_2_3_words(args.seed) or "春 風"

    i = 0
    try:
        while True:
            i += 1

            sp_val = args.sp_strength * (1.0 if random.random() < 0.5 else -1.0)
            sp_b64 = make_soft_prefix_b64_constant(args.p, args.h, sp_val)

            user_prompt = build_user_prompt(prev_phrase)

            raw = cli.llm_inference(
                llm_work_id=llm_work_id,
                user_prompt=user_prompt,
                soft_prefix_b64=sp_b64,
                soft_prefix_len=args.p,
                timeout_sec=240.0,
            )
            curr = sanitize_llm_output_to_2_3_words(raw)

            if curr and len(curr.split(" ")) < 2:
                retry_prompt = (
                    "出力が1語でした。必ず2語か3語にしてください。"
                    "半角スペース区切りで、余計な文字なし。\n"
                    f"前回: {prev_phrase}\n"
                    "出力:"
                )
                raw2 = cli.llm_inference(
                    llm_work_id=llm_work_id,
                    user_prompt=retry_prompt,
                    soft_prefix_b64=sp_b64,
                    soft_prefix_len=args.p,
                    timeout_sec=240.0,
                )
                curr2 = sanitize_llm_output_to_2_3_words(raw2)
                if curr2:
                    curr = curr2

            if not curr:
                curr = prev_phrase

            print(f"[{i:06d}] sp={sp_val:+.3g} prev='{prev_phrase}' -> curr='{curr}'")

            speak_text = f"{prev_phrase}。{curr}。"

            if args.tts_openai:
                try:
                    melotts_openai_wav_ffmpeg_tinyplay_over_ssh(
                        host=args.host,
                        ssh_user=args.ssh_user,
                        ssh_port=args.ssh_port,
                        identity_file=args.ssh_identity,
                        ssh_options=args.ssh_option,
                        enable_mux=(not args.ssh_no_mux),
                        openai_base_url=args.tts_openai_base_url,
                        model=args.tts_openai_model,
                        text=speak_text,
                        speed=args.tts_openai_speed,
                        target_rate=args.tts_openai_target_rate,
                        target_channels=args.tts_openai_target_ch,
                        playcard=args.tts_openai_playcard,
                        playdevice=args.tts_openai_playdevice,
                        timeout_sec=args.tts_openai_timeout,
                    )
                except Exception as e:
                    print("[WARN] OpenAI-TTS playback failed:", repr(e))
                    if args.tts_required:
                        raise

            elif tts_work_id:
                cli.tts_inference(tts_work_id, speak_text, timeout_sec=args.tts_timeout)

            prev_phrase = curr
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n[INFO] stopped by user (Ctrl+C)")

    finally:
        try:
            if tts_work_id:
                cli.exit_work(tts_work_id)
        except Exception:
            pass
        try:
            cli.exit_work(llm_work_id)
        except Exception:
            pass
        try:
            if audio_work_id and "." in audio_work_id:
                cli.exit_work(audio_work_id)
        except Exception:
            pass
        cli.close()


if __name__ == "__main__":
    main()
