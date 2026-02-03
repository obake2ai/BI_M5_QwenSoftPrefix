#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compare_baseline_vs_softprefix_nostream_tts.py

- Soft-prefix の val を振って LLM 出力を比較
- 生成テキストを Module LLM 側の TTS で読み上げ（本体スピーカ再生）
- Python3.13 の socket.makefile + timeout で発生しがちな
  "OSError: cannot read from timed out object" を回避するため、
  JSONL を raw socket + select + 自前バッファで受信します。

重要:
- LLM/TTS の setup 応答で返ってくる work_id (例: "llm.1001", "tts.1001") を
  inference で必ず使います（これが崩れると error -4 が出やすい）。
"""

import base64
import difflib
import hashlib
import json
import re
import select
import socket
import struct
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple


# ========= 設定 =========
HOST = "192.168.3.146"
PORT = 10001

SYSTEM_PROMPT = "あなたは優秀な日本語アシスタントです。"
USER_PROMPT = "こんにちは。植物に関する詩を描いて。"

# LLM: 非streamで返す（比較が目的ならこれが楽）
RESPONSE_FORMAT = "llm.utf-8"
# soft_prefix を渡すため入力は stream 形式で送る
INPUT_OBJECT_FOR_INFER = "llm.utf-8.stream"
MAX_TOKEN_LEN = 128

# soft_prefix 形状
P = 1
H = 896
VALS = [0.0, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1.0, 2.0]

# 通信
CONNECT_TIMEOUT_SEC = 10.0
RECV_POLL_SEC = 0.2           # select待ち
SOCK_TIMEOUT_SEC = 240.0      # inference待ち上限（LLMは長いので）
SETUP_TIMEOUT_SEC = 240.0     # setupが遅い個体もあるので長め

# 結果保存ファイル
OUT_JSON = "compare_results.json"

# ========= Device TTS =========
ENABLE_DEVICE_TTS = True

# Audio unit（スピーカ/マイク設定）。TTS を sys.play.* で鳴らす前提でも
# 初期化しておくと安定しやすい。
ENABLE_AUDIO_SETUP = True
AUDIO_CAPCARD = 0
AUDIO_CAPDEVICE = 0
AUDIO_CAPVOL = 0.5
AUDIO_PLAYCARD = 0
AUDIO_PLAYDEVICE = 1
AUDIO_PLAYVOL = 0.7

# 読み上げ対象
SPEAK_BASELINE = True
SPEAK_EACH_CASE = True
SPEAK_ONLY_CHANGED = False

# 句点補完（TTSは文末句点が必要なことがある）
TTS_AUTO_APPEND_PUNCT = True
TTS_DEFAULT_PUNCT = "。"

# 長文だとTTSが重いので分割
TTS_MAX_CHARS_PER_UTTER = 220
TTS_TIMEOUT_SEC = 120.0
TTS_GAP_SEC = 0.05

# TTS モード優先指定（空なら自動で "melotts-ja-jp" を探す）
TTS_PREFERRED_MODE = "melotts-ja-jp"

# 可能なら device 内蔵再生（sys.play.*）を使う
TTS_PREFER_SYS_PLAY = True

# （拡張）エフェクト: 現行 StackFlow API には標準搭載されていないため
# llm-sys 側の実装がある場合のみ動きます。未対応なら自動で外して継続。
TTS_EFFECTS: Dict[str, Any] = {}

# ========= /設定 =========


# ----------------- util -----------------

def f32_to_bf16_u16(x: float) -> int:
    """float32 -> bf16 (truncate) -> u16"""
    u32 = struct.unpack("<I", struct.pack("<f", x))[0]
    return (u32 >> 16) & 0xFFFF


def make_soft_prefix_b64_constant(p_tokens: int, h: int, val: float) -> str:
    """bf16 little-endian u16 を p_tokens*h 個並べて base64"""
    u16 = f32_to_bf16_u16(val)
    raw = struct.pack("<H", u16) * (p_tokens * h)
    return base64.b64encode(raw).decode("ascii")


def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def preview(s: str, n: int = 120) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    one = s.replace("\n", "\\n")
    return (one[:n] + "…") if len(one) > n else one


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def unified_diff_head(a: str, b: str, n_lines: int = 80) -> str:
    """差分を長くしすぎないため、先頭 n_lines だけの unified diff"""
    a_lines = a.splitlines(keepends=True)[:n_lines]
    b_lines = b.splitlines(keepends=True)[:n_lines]
    diff = difflib.unified_diff(a_lines, b_lines, fromfile="baseline", tofile="case", lineterm="")
    return "\n".join(list(diff)[:200])


def sanitize_for_tts(text: str) -> str:
    """TTS向けに余計な記号・コードブロック等を軽く除去"""
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"```.*?```", "", t, flags=re.S)     # コードブロック除去
    t = re.sub(r"https?://\S+", "", t)              # URL除去
    t = re.sub(r"\n{3,}", "\n\n", t).strip()        # 空行圧縮
    return t.strip()


def ensure_sentence_end_punct(text: str, default_punct: str) -> str:
    if not text:
        return text
    if text[-1] in ".。!?？！":
        return text
    return text + default_punct


def split_for_tts(text: str, max_chars: int) -> List[str]:
    """長文を読み上げやすい長さに分割（簡易）"""
    t = sanitize_for_tts(text)
    if not t:
        return []

    parts: List[str] = []
    for line in t.split("\n"):
        line = line.strip()
        if not line:
            continue

        segs = re.split(r"([。.!?！？])", line)  # 句点等で分割（区切り文字を残す）
        buf = ""
        for i in range(0, len(segs), 2):
            chunk = segs[i].strip()
            punct = segs[i + 1] if i + 1 < len(segs) else ""
            piece = (chunk + punct).strip()
            if not piece:
                continue

            if len(buf) + len(piece) + 1 <= max_chars:
                buf = (buf + " " + piece).strip()
            else:
                if buf:
                    parts.append(buf)
                buf = piece

        if buf:
            parts.append(buf)

    if TTS_AUTO_APPEND_PUNCT:
        parts = [ensure_sentence_end_punct(p, TTS_DEFAULT_PUNCT) for p in parts]

    # 念のため max_chars 超えは強制分割
    final_parts: List[str] = []
    for p in parts:
        if len(p) <= max_chars:
            final_parts.append(p)
        else:
            for j in range(0, len(p), max_chars):
                sub = p[j:j + max_chars].strip()
                if sub:
                    if TTS_AUTO_APPEND_PUNCT:
                        sub = ensure_sentence_end_punct(sub, TTS_DEFAULT_PUNCT)
                    final_parts.append(sub)
    return final_parts


# ----------------- protocol client -----------------

class StackFlowClient:
    """
    StackFlow (Module LLM) JSONL over TCP client.

    - 送信: 1行JSON + '\\n'
    - 受信: 1行JSON
    """

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.sock: Optional[socket.socket] = None
        self._buf = bytearray()
        self._seq = 0

    def __enter__(self):
        self.sock = socket.create_connection((self.host, self.port), timeout=CONNECT_TIMEOUT_SEC)
        # recv側は select で待つので、ここは blocking のまま（timeout=None）
        self.sock.settimeout(None)
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if self.sock:
                self.sock.close()
        except Exception:
            pass
        self.sock = None

    def _next_req_id(self, prefix: str) -> str:
        self._seq += 1
        return f"{prefix}_{time.time_ns()}_{self._seq}"

    def send_json(self, obj: Dict[str, Any]) -> None:
        if not self.sock:
            raise RuntimeError("socket not connected")
        data = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8", errors="ignore")
        self.sock.sendall(data)

    def _recv_into_buffer(self, timeout_sec: float) -> bool:
        """
        受信してバッファに積む。
        戻り値: データを受け取ったら True、timeout なら False。
        """
        if not self.sock:
            raise RuntimeError("socket not connected")

        r, _, _ = select.select([self.sock], [], [], timeout_sec)
        if not r:
            return False

        chunk = self.sock.recv(64 * 1024)
        if chunk == b"":
            raise RuntimeError("socket closed (EOF)")

        self._buf.extend(chunk)
        return True

    def recv_json_line(self, timeout_sec: float) -> Optional[Dict[str, Any]]:
        """
        timeout_sec 以内に 1行分の JSON が来れば dict を返す。
        来なければ None。
        """
        deadline = time.time() + timeout_sec
        while True:
            nl = self._buf.find(b"\n")
            if nl != -1:
                line = bytes(self._buf[:nl]).decode("utf-8", errors="replace").strip()
                del self._buf[:nl + 1]
                if not line:
                    return None
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    print("[WARN] non-json line:", repr(line[:200]))
                    return None

            # バッファに改行が無いので追加受信
            now = time.time()
            if now >= deadline:
                return None
            self._recv_into_buffer(min(RECV_POLL_SEC, deadline - now))

    def wait_response(self, request_id: str, timeout_sec: float, label: str = "") -> Dict[str, Any]:
        """
        指定 request_id の応答が来るまで待つ。
        途中で他 request_id のエラーはログに出す。
        """
        t0 = time.time()
        last_log = 0.0
        while True:
            dt = time.time() - t0
            if dt >= timeout_sec:
                raise TimeoutError(f"timeout waiting response for request_id={request_id}")

            msg = self.recv_json_line(timeout_sec=RECV_POLL_SEC)
            if msg is None:
                # 10秒おきに待機表示
                if dt - last_log >= 10.0 and label:
                    print(f"[WAIT] {label}: {dt:.1f}s ...")
                    last_log = dt
                continue

            # 他の応答も来るので、エラーだけは見える化
            err = msg.get("error")
            if isinstance(err, dict) and err.get("code", 0) != 0:
                print("[RECV-ERR]", msg)

            if msg.get("request_id") == request_id:
                return msg

    # ---------------- SYS ----------------

    def sys_ping(self) -> None:
        req_id = self._next_req_id("ping")
        self.send_json({"request_id": req_id, "work_id": "sys", "action": "ping"})
        resp = self.wait_response(req_id, timeout_sec=5.0, label="sys.ping")
        err = resp.get("error", {})
        if isinstance(err, dict) and err.get("code", 0) != 0:
            raise RuntimeError(f"sys.ping failed: {err} full={resp}")

    def sys_lsmode(self) -> List[Dict[str, Any]]:
        req_id = self._next_req_id("lsmode")
        self.send_json({"request_id": req_id, "work_id": "sys", "action": "lsmode"})
        resp = self.wait_response(req_id, timeout_sec=10.0, label="sys.lsmode")
        err = resp.get("error", {})
        if isinstance(err, dict) and err.get("code", 0) != 0:
            raise RuntimeError(f"sys.lsmode failed: {err} full={resp}")
        data = resp.get("data", [])
        return data if isinstance(data, list) else []

    def sys_hwinfo(self) -> Dict[str, Any]:
        req_id = self._next_req_id("hwinfo")
        self.send_json({"request_id": req_id, "work_id": "sys", "action": "hwinfo"})
        resp = self.wait_response(req_id, timeout_sec=5.0, label="sys.hwinfo")
        err = resp.get("error", {})
        if isinstance(err, dict) and err.get("code", 0) != 0:
            raise RuntimeError(f"sys.hwinfo failed: {err} full={resp}")
        data = resp.get("data", {})
        return data if isinstance(data, dict) else {}

    def sys_reset_and_wait(self, timeout_sec: float = 40.0) -> None:
        # reset 開始
        req_id = self._next_req_id("reset")
        self.send_json({"request_id": req_id, "work_id": "sys", "action": "reset"})
        self.wait_response(req_id, timeout_sec=10.0, label="sys.reset(start)")

        # reset 完了は request_id="0" で返る（仕様）
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            msg = self.recv_json_line(timeout_sec=RECV_POLL_SEC)
            if not msg:
                continue
            if msg.get("work_id") == "sys" and msg.get("request_id") == "0":
                break

        # 念のため ping で復帰確認（reset中は落ちる/遅れる個体対策）
        for _ in range(50):
            try:
                self.sys_ping()
                return
            except Exception:
                time.sleep(0.2)

        raise TimeoutError("sys.reset completed message not received / ping not recovered")

    # ---------------- AUDIO ----------------

    def audio_setup(self) -> str:
        req_id = self._next_req_id("setup_audio")
        payload = {
            "capcard": AUDIO_CAPCARD,
            "capdevice": AUDIO_CAPDEVICE,
            "capVolume": AUDIO_CAPVOL,
            "playcard": AUDIO_PLAYCARD,
            "playdevice": AUDIO_PLAYDEVICE,
            "playVolume": AUDIO_PLAYVOL,
        }
        self.send_json({
            "request_id": req_id,
            "work_id": "audio",
            "action": "setup",
            "object": "audio.setup",
            "data": payload,
        })
        resp = self.wait_response(req_id, timeout_sec=60.0, label="audio.setup")
        err = resp.get("error", {})
        if isinstance(err, dict) and err.get("code", 0) != 0:
            raise RuntimeError(f"audio.setup failed: {err} full={resp}")
        return str(resp.get("work_id", "audio"))

    # ---------------- LLM ----------------

    @staticmethod
    def _pick_llm_mode(lsmodes: List[Dict[str, Any]]) -> str:
        for x in lsmodes:
            if x.get("type") != "llm":
                continue
            # FW差異: "model" or "mode"
            m = x.get("mode") if isinstance(x.get("mode"), str) else None
            if not m and isinstance(x.get("model"), str):
                m = x.get("model")
            if m:
                return m
        raise RuntimeError("No LLM mode found in sys.lsmode")

    def llm_setup(self, llm_mode: str) -> str:
        req_id = self._next_req_id("setup_llm")
        data = {
            # 公式は model キーだが、FW差異に備えて mode も併記
            "model": llm_mode,
            "mode": llm_mode,
            "response_format": RESPONSE_FORMAT,
            "input": INPUT_OBJECT_FOR_INFER,   # stream入力
            "enoutput": True,
            "max_token_len": MAX_TOKEN_LEN,
            "prompt": SYSTEM_PROMPT,
        }
        self.send_json({
            "request_id": req_id,
            "work_id": "llm",
            "action": "setup",
            "object": "llm.setup",
            "data": data,
        })
        resp = self.wait_response(req_id, timeout_sec=SETUP_TIMEOUT_SEC, label="llm.setup")
        err = resp.get("error", {})
        if isinstance(err, dict) and err.get("code", 0) != 0:
            raise RuntimeError(f"llm.setup failed: {err} full={resp}")
        work_id = resp.get("work_id")
        if not isinstance(work_id, str) or not work_id:
            raise RuntimeError(f"llm.setup returned invalid work_id: {work_id!r} full={resp}")
        return work_id

    def llm_inference(
        self,
        llm_work_id: str,
        user_prompt: str,
        soft_prefix_b64: Optional[str],
        soft_prefix_len: int,
        timeout_sec: float,
        retry_on_push_false: bool = True,
        max_retry_sec: float = 30.0,
    ) -> Tuple[str, float, Dict[str, Any]]:
        """
        error -4 (inference data push false) が出る場合があるので、
        短時間だけリトライする。
        """
        start_all = time.time()
        attempt = 0
        delay = 0.2

        while True:
            attempt += 1
            req_id = self._next_req_id("infer_llm")

            data_obj: Dict[str, Any] = {"delta": user_prompt, "index": 0, "finish": True}
            if soft_prefix_b64 is not None:
                data_obj["soft_prefix"] = {"len": int(soft_prefix_len), "data_b64": soft_prefix_b64}

            self.send_json({
                "request_id": req_id,
                "work_id": llm_work_id,  # ★必ず llm.setup の戻り work_id を使う
                "action": "inference",
                "object": INPUT_OBJECT_FOR_INFER,
                "data": data_obj,
            })

            t0 = time.time()
            resp = self.wait_response(req_id, timeout_sec=timeout_sec, label="llm.inference")
            dt = time.time() - t0

            err = resp.get("error", {})
            if isinstance(err, dict) and err.get("code", 0) == 0:
                out = resp.get("data", "")
                if not isinstance(out, str):
                    out = json.dumps(out, ensure_ascii=False)
                return out, dt, resp

            code = err.get("code") if isinstance(err, dict) else None

            # -4: inference data push false
            if retry_on_push_false and code == -4 and (time.time() - start_all) < max_retry_sec:
                time.sleep(delay)
                delay = min(2.0, delay * 1.4)
                continue

            raise RuntimeError(f"llm.inference failed: {err} full={resp}")

    # ---------------- TTS ----------------

    @staticmethod
    def _as_list(x: Any) -> List[str]:
        if isinstance(x, list):
            return [str(i) for i in x if isinstance(i, (str, int, float))]
        if isinstance(x, str):
            return [x]
        return []

    @staticmethod
    def _pick_tts_candidates(lsmodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        tts = [x for x in lsmodes if x.get("type") == "tts"]
        # まず "melotts-ja-jp" を優先
        preferred: List[Dict[str, Any]] = []
        others: List[Dict[str, Any]] = []
        for x in tts:
            mode = x.get("mode") if isinstance(x.get("mode"), str) else x.get("model")
            mode = mode if isinstance(mode, str) else ""
            if mode == TTS_PREFERRED_MODE or "ja" in mode.lower() or "japanese" in mode.lower():
                preferred.append(x)
            else:
                others.append(x)
        return preferred + others

    @staticmethod
    def _pick_sys_play(output_types: List[str]) -> Optional[str]:
        for ot in output_types:
            if isinstance(ot, str) and ot.startswith("sys.play"):
                return ot
        return None

    def tts_setup(self, tts_mode: str, tts_input: str, response_format: str, effects: Optional[Dict[str, Any]]) -> str:
        req_id = self._next_req_id("setup_tts")
        data: Dict[str, Any] = {
            "model": tts_mode,            # 公式のキー
            "mode": tts_mode,             # FW差異向け
            "response_format": response_format,
            "input": tts_input,
            "enoutput": False,            # sys.play で鳴らす想定なのでUARTへ音声返さない
            "enkws": False,
        }
        if effects:
            data["effects"] = effects

        self.send_json({
            "request_id": req_id,
            "work_id": "tts",
            "action": "setup",
            "object": "tts.setup",
            "data": data,
        })
        resp = self.wait_response(req_id, timeout_sec=SETUP_TIMEOUT_SEC, label="tts.setup")
        err = resp.get("error", {})
        if isinstance(err, dict) and err.get("code", 0) != 0:
            # effects を付けて失敗したら外して再試行
            if effects:
                print(f"[WARN] tts.setup failed with effects; retry without effects. err={err}")
                data.pop("effects", None)
                req_id2 = self._next_req_id("setup_tts_retry")
                self.send_json({
                    "request_id": req_id2,
                    "work_id": "tts",
                    "action": "setup",
                    "object": "tts.setup",
                    "data": data,
                })
                resp2 = self.wait_response(req_id2, timeout_sec=SETUP_TIMEOUT_SEC, label="tts.setup(retry)")
                err2 = resp2.get("error", {})
                if isinstance(err2, dict) and err2.get("code", 0) != 0:
                    raise RuntimeError(f"tts.setup failed: {err2} full={resp2}")
                work_id = resp2.get("work_id")
                if not isinstance(work_id, str) or not work_id:
                    raise RuntimeError(f"tts.setup returned invalid work_id: {work_id!r} full={resp2}")
                return work_id
            raise RuntimeError(f"tts.setup failed: {err} full={resp}")

        work_id = resp.get("work_id")
        if not isinstance(work_id, str) or not work_id:
            raise RuntimeError(f"tts.setup returned invalid work_id: {work_id!r} full={resp}")
        return work_id

    def tts_speak(self, tts_work_id: str, text: str, timeout_sec: float) -> None:
        chunks = split_for_tts(text, TTS_MAX_CHARS_PER_UTTER)
        if not chunks:
            return

        for idx, ch in enumerate(chunks):
            req_id = self._next_req_id(f"infer_tts_{idx}")
            self.send_json({
                "request_id": req_id,
                "work_id": tts_work_id,      # ★必ず tts.setup の戻り work_id を使う
                "action": "inference",
                "object": "tts.utf-8",
                "data": ch,
            })
            resp = self.wait_response(req_id, timeout_sec=timeout_sec, label="tts.inference")
            err = resp.get("error", {})
            if isinstance(err, dict) and err.get("code", 0) != 0:
                raise RuntimeError(f"tts.inference failed: {err} full={resp}")
            time.sleep(TTS_GAP_SEC)

    def exit_work(self, work_id: str) -> None:
        # exit は応答が来ないこともあるので、送るだけ
        req_id = self._next_req_id("exit")
        self.send_json({"request_id": req_id, "work_id": work_id, "action": "exit"})


# ---------------- result ----------------

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


# ---------------- main ----------------

def main():
    results: List[CaseResult] = []

    with StackFlowClient(HOST, PORT) as cli:
        # sys.ping
        cli.sys_ping()
        cli.sys_reset_and_wait()   # ★追加

        # sys.lsmode
        lsmodes = cli.sys_lsmode()

        # LLM mode
        llm_mode = cli._pick_llm_mode(lsmodes)
        print("[lsmode.llm_mode]", llm_mode)

        # Audio setup（任意）
        audio_work_id: Optional[str] = None
        if ENABLE_AUDIO_SETUP:
            try:
                audio_work_id = cli.audio_setup()
                print("[AUDIO SETUP OK] work_id:", audio_work_id)
            except Exception as e:
                print("[WARN] audio.setup failed (continue):", repr(e))
                audio_work_id = None

        # LLM setup
        llm_work_id = cli.llm_setup(llm_mode)
        print("[LLM SETUP OK] work_id:", llm_work_id)

        # TTS setup（任意）
        tts_work_id: Optional[str] = None
        tts_pick_info: Optional[Dict[str, Any]] = None
        if ENABLE_DEVICE_TTS:
            candidates = cli._pick_tts_candidates(lsmodes)
            for ent in candidates:
                mode = ent.get("mode") if isinstance(ent.get("mode"), str) else ent.get("model")
                mode = mode if isinstance(mode, str) else ""
                if not mode:
                    continue

                input_types = cli._as_list(ent.get("input_type"))
                output_types = cli._as_list(ent.get("output_type"))
                tts_input = input_types[0] if input_types else "tts.utf-8"

                # 再生を優先：sys.play.* があればそれを response_format にする
                sys_play = cli._pick_sys_play(output_types) if TTS_PREFER_SYS_PLAY else None
                resp_fmt = sys_play if sys_play else "tts.base64.wav"

                print("[lsmode.tts_try]", {"mode": mode, "input": tts_input, "response_format": resp_fmt})

                try:
                    tts_work_id = cli.tts_setup(
                        tts_mode=mode,
                        tts_input=tts_input,
                        response_format=resp_fmt,
                        effects=TTS_EFFECTS if TTS_EFFECTS else None,
                    )
                    tts_pick_info = {"mode": mode, "input": tts_input, "response_format": resp_fmt}
                    print("[TTS SETUP OK] work_id:", tts_work_id)
                    break
                except Exception as e:
                    print("[WARN] tts.setup failed for mode:", mode, "err=", repr(e))
                    tts_work_id = None
                    continue

            if not tts_work_id:
                # 状況把握のため hwinfo を出す
                try:
                    hw = cli.sys_hwinfo()
                    print("[INFO] sys.hwinfo", hw)
                except Exception:
                    pass
                print("[WARN] TTS setup failed for all candidates; continue without TTS.")

        # ---- baseline ----
        base_text, base_dt, _ = cli.llm_inference(
            llm_work_id=llm_work_id,
            user_prompt=USER_PROMPT,
            soft_prefix_b64=None,
            soft_prefix_len=0,
            timeout_sec=SOCK_TIMEOUT_SEC,
            retry_on_push_false=True,
            max_retry_sec=60.0,
        )

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

        if tts_work_id and SPEAK_BASELINE:
            print("[TTS] speak baseline on device...")
            cli.tts_speak(tts_work_id, baseline.out_text, timeout_sec=TTS_TIMEOUT_SEC)

        # ---- prefix cases ----
        for v in VALS:
            sp_b64 = make_soft_prefix_b64_constant(P, H, v)
            out, dt, _ = cli.llm_inference(
                llm_work_id=llm_work_id,
                user_prompt=USER_PROMPT,
                soft_prefix_b64=sp_b64,
                soft_prefix_len=P,
                timeout_sec=SOCK_TIMEOUT_SEC,
                retry_on_push_false=True,
                max_retry_sec=60.0,
            )

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
            d = unified_diff_head(baseline.out_text, out)
            if d.strip():
                print("--- diff(head) ---")
                print(d)
            else:
                print("--- diff(head) --- (no diff in head)")
            print("")

            if tts_work_id and SPEAK_EACH_CASE:
                if SPEAK_ONLY_CHANGED and cr.out_sha1 == baseline.out_sha1:
                    continue
                print(f"[TTS] speak case val={v} on device...")
                cli.tts_speak(tts_work_id, cr.out_text, timeout_sec=TTS_TIMEOUT_SEC)

        # 終了（存在する work_id のみ）
        if tts_work_id:
            cli.exit_work(tts_work_id)
        if llm_work_id:
            cli.exit_work(llm_work_id)
        if audio_work_id:
            cli.exit_work(audio_work_id)

    # 保存
    payload = {
        "meta": {
            "host": HOST,
            "port": PORT,
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt": USER_PROMPT,
            "response_format": RESPONSE_FORMAT,
            "input_object_for_infer": INPUT_OBJECT_FOR_INFER,
            "max_token_len": MAX_TOKEN_LEN,
            "P": P,
            "H": H,
            "vals": VALS,
            "saved_at_unix": int(time.time()),
            "tts": {
                "enabled": ENABLE_DEVICE_TTS,
                "picked": tts_pick_info,
                "speak_baseline": SPEAK_BASELINE,
                "speak_each_case": SPEAK_EACH_CASE,
                "speak_only_changed": SPEAK_ONLY_CHANGED,
                "effects": TTS_EFFECTS,
            },
            "audio": {
                "enabled": ENABLE_AUDIO_SETUP,
                "capcard": AUDIO_CAPCARD,
                "capdevice": AUDIO_CAPDEVICE,
                "capVolume": AUDIO_CAPVOL,
                "playcard": AUDIO_PLAYCARD,
                "playdevice": AUDIO_PLAYDEVICE,
                "playVolume": AUDIO_PLAYVOL,
            },
        },
        "results": [asdict(r) for r in results],
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    baseline_sha = results[0].out_sha1 if results else ""
    changed = sum(1 for r in results[1:] if r.out_sha1 != baseline_sha)
    print(f"\nSaved: {OUT_JSON}")
    print(f"Changed cases vs baseline: {changed}/{max(0, len(results)-1)}")
    if results and changed == 0:
        print("NOTE: 全ケースが baseline と同一です。soft_prefix がサーバ側で適用されていない可能性があります。")


if __name__ == "__main__":
    main()
