import socket, json, sys

HOST = "192.168.3.132"
PORT = 10001
SOCK_TIMEOUT = 180

def send_json(w, obj):
    w.write(json.dumps(obj, ensure_ascii=False) + "\n")
    w.flush()

def read_json_line(r):
    line = r.readline()
    if line == "":
        raise RuntimeError("socket closed (EOF)")
    line = line.strip()
    if not line:
        return None
    return json.loads(line)

with socket.create_connection((HOST, PORT), timeout=SOCK_TIMEOUT) as sock:
    sock.settimeout(SOCK_TIMEOUT)
    r = sock.makefile("r", encoding="utf-8", newline="\n")
    w = sock.makefile("w", encoding="utf-8", newline="\n")

    setup_req = {
        "request_id": "llm_001",
        "work_id": "llm",
        "action": "setup",
        "object": "llm.setup",
        "data": {
            "model": "qwen2.5-0.5B-prefill-20e",
            "response_format": "llm.utf-8",   # ★ stream なし
            "input": "llm.utf-8",             # ★ stream なし
            "enoutput": True,
            "max_token_len": 128,             # まず短め
            "prompt": "あなたは優秀な日本語アシスタントです。"
        }
    }
    send_json(w, setup_req)
    setup_resp = read_json_line(r)
    print("[SETUP]", setup_resp)

    err = setup_resp.get("error", {})
    if isinstance(err, dict) and err.get("code", 0) != 0:
        print("SETUP FAILED:", err)
        sys.exit(1)

    work_id = setup_resp.get("work_id", "llm")
    print("work_id:", work_id)

    infer_req = {
        "request_id": "llm_002",
        "work_id": work_id,
        "action": "inference",
        "object": "llm.utf-8",  # ★ stream なし
        "data": "こんにちは。植物に関する短い詩を日本語で。"
    }
    send_json(w, infer_req)

    # ★ 非ストリームなら、基本1発で返る想定
    resp = read_json_line(r)
    print("[RESP]", resp)

    send_json(w, {"request_id": "llm_exit", "work_id": work_id, "action": "exit"})
