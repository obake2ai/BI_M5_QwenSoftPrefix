import socket, json

HOST = "192.168.3.132"   # m5stack側のIP
PORT = 10001

def send_json(sock, obj):
    data = json.dumps(obj, ensure_ascii=False) + "\n"
    sock.sendall(data.encode("utf-8"))

def recv_line(sock):
    buf = ""
    while "\n" not in buf:
        buf += sock.recv(4096).decode("utf-8")
    return buf.strip()

with socket.create_connection((HOST, PORT)) as sock:
    # setup
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
    send_json(sock, setup_req)
    setup_resp = json.loads(recv_line(sock))
    llm_work_id = setup_resp.get("work_id")
    print("work_id:", llm_work_id)

    # inference
    send_json(sock, {
        "request_id": "llm_002",
        "work_id": llm_work_id,
        "action": "inference",
        "object": "llm.utf-8.stream",
        "data": {"delta": "こんにちは。植物に関する詩を描いて。", "index": 0, "finish": True}
    })

    while True:
        r = json.loads(recv_line(sock))
        d = r.get("data", {})
        print(d.get("delta", ""), end="", flush=True)
        if d.get("finish"):
            print()
            break

    # exit
    send_json(sock, {"request_id": "llm_exit", "work_id": llm_work_id, "action": "exit"})
    print(json.loads(recv_line(sock)))
