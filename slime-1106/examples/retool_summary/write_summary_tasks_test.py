import os, time, json, requests, uuid, random, sys

HOST = os.getenv("DB_HOST", "127.0.0.1")
PORT = int(os.getenv("DB_PORT", "18888"))
KEY_SUFFIX = os.getenv("KEY_SUFFIX")
INTERVAL = float(os.getenv("INTERVAL", "3"))
PROMPT = os.getenv("PROMPT", "请总结下面对话，输出要点与结论")
COUNT = int(os.getenv("COUNT", "0"))  # 0 表示无限循环

if not KEY_SUFFIX:
    print("ERROR: KEY_SUFFIX is required (must match agent-summary).")
    sys.exit(1)

BASE = f"http://{HOST}:{PORT}"
LIST_KEY = f"summary_queue_{KEY_SUFFIX}"

def health():
    try:
        r = requests.get(f"{BASE}/health", timeout=3)
        print("DB health:", r.text)
    except Exception as e:
        print("DB health check failed:", e)

def commit_once(idx: int):
    task_id = f"T{int(time.time())}_{idx}_{uuid.uuid4().hex[:8]}"
    # 构造一点变化的“对话”
    pats = [
        "医生: 近两日有无发热？\n病人: 无发热，有轻微头痛。",
        "医生: 是否咳嗽？\n病人: 偶有干咳，无痰。",
        "医生: 是否有胸闷气短？\n病人: 运动后轻微气促，休息可缓解。",
        "医生: 是否腹泻恶心？\n病人: 无腹泻，无恶心。"
    ]
    history = "\n".join(random.sample(pats, k=2))
    token_count = random.randint(300, 1200)

    task_data = {
        "taskId": task_id,
        "originalPrompt": PROMPT,
        "conversationHistory": [{"role": "assistant", "content": history}],
        "tokenCount": token_count,
        "metadata": {}
    }
    payload = {
        "listKey": LIST_KEY,
        "taskData": json.dumps(task_data, ensure_ascii=False)
    }
    try:
        r = requests.post(f"{BASE}/taskCommit", json=payload, timeout=5)
        ok = r.ok and r.json().get("success")
        print(f"[commit] idx={idx} task_id={task_id} ok={ok}")
    except Exception as e:
        print(f"[commit] idx={idx} error: {e}")

def main():
    health()
    i = 0
    while True:
        i += 1
        commit_once(i)
        time.sleep(INTERVAL)
        if COUNT > 0 and i >= COUNT:
            break

if __name__ == "__main__":
    main()
