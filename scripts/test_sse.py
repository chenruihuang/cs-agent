# scripts/test_sse.py
import json, urllib.request, time
req = urllib.request.Request(
    "http://localhost:8000/chat",
    data=json.dumps({"message": "16岁打工能签合同吗"}).encode(),
    headers={"Content-Type": "application/json"},
)

t0 = time.time()
with urllib.request.urlopen(req) as resp:
    for raw in resp:                      # urllib 逐行读
        line = raw.decode().strip()
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":              # ← 关键：跳过结束标记
            print(f"\n<DONE>")
            break
        payload = json.loads(data)
        if payload.get("type") == "docs":
            print(f"来源页: {payload['pages']}")
        elif payload.get("type") == "token":
            print(f"{payload['content']}", end="", flush=True)