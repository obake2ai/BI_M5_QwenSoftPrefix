import socket, json, sys

HOST = "192.168.3.132"
PORT = 10001

SOCK_TIMEOUT = 180  # 念のため長め

def send_json(w, obj):
    w.write(json.dumps(obj, ensure_ascii=False) + "\n")
    w.flush()

def read_json_line(r):
    """readline() なので、TCPの1パケットに複数行入っても取りこぼさない"""
    line = r.readline()
    if line == "":
        raise RuntimeError("socket closed (EOF)")
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        print("[WARN] non-json line:", repr(line), file=sys.stderr)
        return None

with socket.create_connection((HOST, PORT), timeout=SOCK_TIMEOUT) as sock:
    sock.settimeout(SOCK_TIMEOUT)

    # Text mode makefile（readlineが安全）
    r = sock.makefile("r", encoding="utf-8", newline="\n")
    w = sock.makefile("w", encoding="utf-8", newline="\n")

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
            "max_token_len": 256,  # まず短めで安定確認（長くしたいなら後で戻す）
            "prompt": "あなたは優秀な日本語アシスタントです。"
        }
    }
    send_json(w, setup_req)

    setup_resp = None
    while setup_resp is None:
        setup_resp = read_json_line(r)

    print("[SETUP]", setup_resp)
    err = setup_resp.get("error", {})
    if isinstance(err, dict) and err.get("code", 0) != 0:
        raise SystemExit(f"SETUP FAILED: {err}")

    work_id = setup_resp.get("work_id", "llm")
    print("work_id:", work_id)

    # inference
    infer_req = {
        "request_id": "llm_002",
        "work_id": work_id,
        "action": "inference",
        "object": "llm.utf-8.stream",
        "data": {"delta": "こんにちは。植物に関する詩を描いて。", "index": 0, "finish": True}
    }
    send_json(w, infer_req)

    # receive loop
    while True:
        resp = read_json_line(r)
        if resp is None:
            continue

        err = resp.get("error", {})
        if isinstance(err, dict) and err.get("code", 0) != 0:
            print("[ERROR]", err, resp)
            break

        data = resp.get("data")

        # streamの本体は dict で来る
        if isinstance(data, dict):
            delta = data.get("delta", "")
            if delta:
                print(delta, end="", flush=True)
            if data.get("finish") is True:
                print()
                break
        else:
            # "None" 等のACKが来ることがあるので無視でOK
            pass

    # exit
    send_json(w, {"request_id": "llm_exit", "work_id": work_id, "action": "exit"})
    # exit応答は来ない/来るが環境次第なので、来たら表示
    try:
        ex = read_json_line(r)
        if ex:
            print("[EXIT]", ex)
    except Exception:
        pass
