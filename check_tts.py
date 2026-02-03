#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import time

from check_softprefix_nostream_tts import StackFlowClient, pick_tts_response_format

def list_tts_models(lsmode):
    models = []
    for x in lsmode:
        if x.get("type") == "tts":
            for k in ("mode", "model"):
                v = x.get(k)
                if isinstance(v, str) and v:
                    models.append(v)
    uniq = []
    seen = set()
    for m in models:
        if m not in seen:
            seen.add(m)
            uniq.append(m)
    return uniq

HOST="192.168.3.146"
PORT=10001

cli = StackFlowClient(HOST, PORT)
cli.connect()
cli.reset_and_wait_ready(max_wait_sec=40.0)

# ★追加：reset直後はセッションが不安定なことがあるので張り直す
cli.close()
time.sleep(0.8)
cli.connect()

# 念のため ping
if not cli.sys_ping():
    raise RuntimeError("sys.ping failed after reconnect")

lsmode = cli.sys_lsmode()
print(json.dumps(lsmode, ensure_ascii=False, indent=2))

tts_resp = pick_tts_response_format(lsmode)
print("tts_resp =", tts_resp)

# audioは再生に必要
wid_audio = cli.audio_setup_best_effort(play_volume=0.7)
print("audio_work_id =", wid_audio)

cands = ["melotts-ja-jp", "model-melotts-ja-jp", "single_speaker_fast", "single_speaker_english_fast"]

print("tts candidates from lsmode =", cands)

ok = False
last = None

for m in (cands + ["melotts-ja-jp"]):
    try:
        wid_tts = cli.tts_setup(model=m, response_format=tts_resp)
        print("TTS SETUP OK:", wid_tts, "model=", m)

        # ここ大事：英語モデルなら英語で喋らせる（日本語だと無音/変な音の可能性）
        if "english" in m:
            cli.tts_inference(wid_tts, "This is a test.", timeout_sec=8.0)  # '.'で終える
        else:
            cli.tts_inference(wid_tts, "テストです。", timeout_sec=8.0)

        time.sleep(2.0)  # 再生のラグ対策
        ok = True
        last = None
        break

    except Exception as e:
        last = e
        print("TTS SETUP FAILED model=", m, "err=", repr(e))

if not ok:
    print("ALL FAILED:", repr(last))
