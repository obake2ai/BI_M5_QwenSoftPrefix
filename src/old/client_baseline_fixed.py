import socket, json

HOST="192.168.3.132"
PORT=10001

def send(sock, obj):
    sock.sendall((json.dumps(obj, ensure_ascii=False)+"\n").encode("utf-8"))

def recv(sock):
    buf=""
    while "\n" not in buf:
        chunk = sock.recv(4096)
        if not chunk:
            raise RuntimeError("socket closed")
        buf += chunk.decode("utf-8", errors="replace")
    return json.loads(buf.split("\n",1)[0].strip())

with socket.create_connection((HOST, PORT)) as sock:
    setup_req = {
        "request_id":"llm_001",
        "work_id":"llm",
        "action":"setup",
        "object":"llm.setup",
        "data":{
            "model":"qwen2.5-0.5B-prefill-20e",
            "response_format":"llm.utf-8.stream",
            "input":"llm.utf-8.stream",
            "enoutput": True,
            "max_token_len": 1023,
            "prompt":"あなたは優秀な日本語アシスタントです。"
        }
    }
    send(sock, setup_req)
    r = recv(sock)
    print("[SETUP]", r)

    err = r.get("error", {})
    if isinstance(err, dict) and err.get("code", 0) != 0:
        raise SystemExit(f"SETUP FAILED: {err}")

    work_id = r.get("work_id", "llm")
    print("work_id:", work_id)

    send(sock, {
        "request_id":"llm_002",
        "work_id": work_id,
        "action":"inference",
        "object":"llm.utf-8.stream",
        "data":{"delta":"こんにちは。植物に関する詩を描いて。", "index":0, "finish":True}
    })

    while True:
        r = recv(sock)
        err = r.get("error", {})
        if isinstance(err, dict) and err.get("code", 0) != 0:
            print("[ERROR]", err, r)
            break
        d = r.get("data")
        if isinstance(d, dict):
            print(d.get("delta",""), end="", flush=True)
            if d.get("finish") is True:
                print()
                break
        else:
            # エラー/ACKなどで data が文字列のケース
            # print("[INFO] data=", repr(d), r)
            pass
