import socket, json, base64, struct, sys, time
from typing import Any, Dict, Optional, Tuple

HOST = "192.168.3.132"
PORT = 10001

# ===== 設定 =====
P = 1
H = 896
VAL = 0.02
PROMPT_TEXT = "こんにちは。植物に関する詩を描いて。"
SOCK_TIMEOUT_SEC = 15.0
# ===============

class JsonLineClient:
    def __init__(self, sock: socket.socket):
        self.sock = sock
        self.sock.settimeout(SOCK_TIMEOUT_SEC)
        self._buf = ""

    def send_obj(self, obj: Dict[str, Any], show_bytes: bool = False) -> int:
        payload = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")
        if show_bytes:
            print("SEND_BYTES:", len(payload))
        self.sock.sendall(payload)
        return len(payload)

    def recv_line(self) -> str:
        # バッファに改行が来るまで受信（余りは保持）
        while "\n" not in self._buf:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise RuntimeError("socket closed by peer")
            self._buf += chunk.decode("utf-8", errors="replace")
        line, self._buf = self._buf.split("\n", 1)
        return line.strip()

    def recv_json(self) -> Dict[str, Any]:
        line = self.recv_line()
        return json.loads(line)

def f32_to_bf16_u16(x: float) -> int:
    u32 = struct.unpack("<I", struct.pack("<f", x))[0]
    return (u32 >> 16) & 0xFFFF

def make_soft_prefix_b64(P: int, H: int, val: float) -> str:
    u16 = f32_to_bf16_u16(val)
    raw = b"".join(struct.pack("<H", u16) for _ in range(P * H))
    return base64.b64encode(raw).decode("ascii")

def get_error(resp: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    err = resp.get("error")
    if isinstance(err, dict) and isinstance(err.get("code"), int) and err.get("code") != 0:
        return err
    return None

def normalize_data_field(d: Any) -> Any:
    # data が JSON文字列で返ることもあるので復元を試す
    if isinstance(d, str):
        s = d.strip()
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                return json.loads(s)
            except Exception:
                return d
        return d
    return d

def extract_work_id(setup_resp: Dict[str, Any], fallback: str = "llm") -> str:
    # 念のため複数パターンを試す（環境差吸収）
    wid = setup_resp.get("work_id")
    if isinstance(wid, str) and wid:
        return wid

    d = setup_resp.get("data")
    if isinstance(d, dict):
        wid2 = d.get("work_id")
        if isinstance(wid2, str) and wid2:
            return wid2

    return fallback

def recv_stream_until_finish(cli: JsonLineClient, first_resp: Optional[Dict[str, Any]] = None) -> None:
    # すでに1つレスポンスを読んでいる場合はそれも処理
    def handle(resp: Dict[str, Any]) -> bool:
        err = get_error(resp)
        if err:
            print("\n[ERROR]", err, file=sys.stderr)
            return True  # stop

        d = normalize_data_field(resp.get("data"))
        # stream: data が dict で delta/finish を持つ
        if isinstance(d, dict):
            delta = d.get("delta", "")
            if isinstance(delta, str) and delta:
                print(delta, end="", flush=True)
            if d.get("finish") is True:
                print()
                return True  # stop
            return False

        # non-stream: data が文字列のこともある
        if isinstance(d, str):
            if d not in ("", "None", "null"):
                print(d, end="", flush=True)
            return False

        # その他
        # print(f"\n[DEBUG] resp={resp}\n", file=sys.stderr)
        return False

    if first_resp is not None:
        if handle(first_resp):
            return

    while True:
        resp = cli.recv_json()
        if handle(resp):
            return

def build_inference_request(
    work_id: str,
    prompt_text: str,
    soft_prefix: Optional[Dict[str, Any]],
    data_mode: str,
) -> Dict[str, Any]:
    """
    data_mode:
      - "json_string": data に inner dict を json.dumps した「文字列」を入れる（推奨）
      - "object": data に inner dict をそのまま入れる（従来互換）
    """
    inner = {"delta": prompt_text, "index": 0, "finish": True}
    if soft_prefix is not None:
        inner["soft_prefix"] = soft_prefix  # {"len":P, "data_b64":...}

    if data_mode == "json_string":
        data_field: Any = json.dumps(inner, ensure_ascii=False)
    elif data_mode == "object":
        data_field = inner
    else:
        raise ValueError("unknown data_mode")

    return {
        "request_id": "llm_002",
        "work_id": work_id,
        "action": "inference",
        "object": "llm.utf-8.stream",
        "data": data_field,
    }

def try_inference(cli: JsonLineClient, work_id: str, prompt_text: str, soft_prefix_obj: Optional[Dict[str, Any]]) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    戻り値:
      (success, first_response)
    """
    # 最優先：data を JSON文字列として送る（push条件が厳しい環境向け）
    for mode in ("json_string", "object"):
        req = build_inference_request(work_id, prompt_text, soft_prefix_obj, data_mode=mode)
        payload_bytes = (json.dumps(req, ensure_ascii=False) + "\n").encode("utf-8")
        print(f"[TRY] inference mode={mode} bytes={len(payload_bytes)} prefix={'yes' if soft_prefix_obj else 'no'}")

        cli.sock.sendall(payload_bytes)

        # 多くの環境では「即エラー」か「ストリームの最初の1フレーム」が返る
        first = cli.recv_json()
        print("[RAW RESP]", first)

        err = get_error(first)
        if err:
            # push false など：解決策を切り分けるため、次のモードへ
            print("ERROR:", err, file=sys.stderr)
            continue

        # エラーでなければ成功扱い。ストリーム処理へ渡す
        return True, first

    return False, None

def safe_exit(cli: JsonLineClient, work_id: str) -> None:
    # exit は失敗しても致命ではないので握り潰す
    try:
        cli.send_obj({"request_id": "llm_exit", "work_id": work_id, "action": "exit"})
        resp = cli.recv_json()
        print("[EXIT RESP]", resp)
    except Exception as e:
        print("[EXIT ERROR]", e, file=sys.stderr)

def main():
    sp_b64 = make_soft_prefix_b64(P, H, VAL)
    print("base64_len:", len(sp_b64))

    with socket.create_connection((HOST, PORT), timeout=SOCK_TIMEOUT_SEC) as sock:
        cli = JsonLineClient(sock)

        # ---- setup ----
        setup_req = {
            "request_id": "llm_001",
            "work_id": "llm",
            "action": "setup",
            "object": "llm.setup",
            "data": {
                "model": "qwen2.5-0.5B-prefill-20e",
                "response_format": "llm.utf-8.stream",
                "input": "llm.utf-8.stream",
                "enoutput": True,
                "max_token_len": 1023,
                "prompt": "あなたは優秀な日本語アシスタントです。"
            }
        }
        cli.send_obj(setup_req)
        setup_resp = cli.recv_json()
        print("[SETUP RESP]", setup_resp)

        err = get_error(setup_resp)
        if err:
            print("SETUP ERROR:", err, file=sys.stderr)
            return

        work_id = extract_work_id(setup_resp, fallback="llm")
        print("work_id:", work_id)

        # ---- inference (prefix付き) ----
        soft_prefix_obj = {"len": P, "data_b64": sp_b64}

        ok, first = try_inference(cli, work_id, PROMPT_TEXT, soft_prefix_obj)
        if ok:
            recv_stream_until_finish(cli, first_resp=first)
            safe_exit(cli, work_id)
            return

        # ---- prefix付きが全滅なら、prefix無しでも動くか切り分け ----
        print("\n[INFO] prefix付きが push できないため、prefix無しで切り分けします。\n", file=sys.stderr)
        ok2, first2 = try_inference(cli, work_id, PROMPT_TEXT, soft_prefix_obj=None)
        if ok2:
            recv_stream_until_finish(cli, first_resp=first2)
            safe_exit(cli, work_id)

            print(
                "\n[DIAGNOSIS] prefix無しは動くが prefix付きは push false。\n"
                "→ 現状の llm-sys / inference 受付が、data内の追加キー（soft_prefix）を弾いている可能性が高いです。\n"
                "→ 対策は「soft_prefix を別オブジェクトで事前送信してサーバ側に保持」などの方式に変える必要があります。\n",
                file=sys.stderr
            )
            return

        print(
            "\n[DIAGNOSIS] prefix無しでも push できません。\n"
            "→ setup が実は失敗見え/ work_id不一致 / inference形式不一致の可能性。\n"
            "→ M5側で journalctl を見て原因箇所を特定してください。\n",
            file=sys.stderr
        )
        safe_exit(cli, work_id)

if __name__ == "__main__":
    main()
