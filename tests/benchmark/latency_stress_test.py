import asyncio
import json
import time
import websockets
import numpy as np

GATEWAY_WS_URL = "ws://localhost:8000/v1/audio/stream"

async def simulate_voice_client(client_id: int):
    print(f"[Client-{client_id}] Connecting to WebSocket Gateway at {GATEWAY_WS_URL}...")
    try:
        async with websockets.connect(GATEWAY_WS_URL) as ws:
            print(f"[Client-{client_id}] Connected!")
            
            # 1. Send simulated 80ms PCM audio chunks (1280 float32 samples @ 16kHz)
            t_start = time.perf_counter()
            pcm_chunk = np.zeros(1280, dtype=np.float32).tobytes()
            
            for _ in range(5):
                await ws.send(pcm_chunk)
                await asyncio.sleep(0.08)

            # 2. Trigger text/speech end event
            user_text = f"Namaste, what is the weather today in Bengaluru? (Client {client_id})"
            print(f"[Client-{client_id}] Sending final prompt: '{user_text}'")
            t_prompt_sent = time.perf_counter()
            
            await ws.send(json.dumps({"type": "TEXT_PROMPT", "text": user_text}))

            # Metrics tracking
            t_llm_start = None
            t_first_token = None
            t_first_sentence = None
            t_first_audio = None
            
            while True:
                response = await ws.recv()
                t_recv = time.perf_counter()
                
                if isinstance(response, str):
                    data = json.loads(response)
                    msg_type = data.get("type")
                    
                    if msg_type == "LLM_START" and t_llm_start is None:
                        t_llm_start = t_recv
                    elif msg_type == "LLM_TOKEN" and t_first_token is None:
                        t_first_token = t_recv
                        ttft_ms = (t_first_token - t_prompt_sent) * 1000
                        print(f"  ⚡ [Client-{client_id}] Time to First Token (TTFT): {ttft_ms:.2f} ms")
                    elif msg_type == "LLM_SENTENCE" and t_first_sentence is None:
                        t_first_sentence = t_recv
                    elif msg_type == "RESPONSE_DONE":
                        print(f"  ✅ [Client-{client_id}] Response stream completed!")
                        break
                elif isinstance(response, bytes) and t_first_audio is None:
                    t_first_audio = t_recv
                    ttfa_ms = (t_first_audio - t_prompt_sent) * 1000
                    print(f"  🔊 [Client-{client_id}] Time to First Audio (TTFA): {ttfa_ms:.2f} ms")

    except Exception as e:
        print(f"❌ [Client-{client_id}] Connection error: {e}")

async def run_benchmark(num_concurrent_clients: int = 5):
    print(f"==================================================")
    print(f"Starting Latency Stress Benchmark with {num_concurrent_clients} concurrent streams")
    print(f"==================================================")
    tasks = [simulate_voice_client(i) for i in range(num_concurrent_clients)]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(run_benchmark(num_concurrent_clients=3))
