# tools/test_endpoint_concurrent.py
import asyncio
import time
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import httpx

# =======================
# Config
# =======================
CSV_PATH = "data/future_unseen_examples.csv"
BASE_URL = "http://localhost:8000"
ENDPOINT = "/predict"           # or "/predict/minimal"
DURATION_SECS = 60              # total test duration
BATCH_SIZE = 5000                # requests per wave (will recycle rows if CSV < BATCH_SIZE)
CONCURRENCY = 100               # max concurrent in-flight requests
TIMEOUT_SECS = 30               # per-request timeout

HOUSE_ROW_COLS: List[str] = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
    "waterfront", "view", "condition", "grade", "sqft_above",
    "sqft_basement", "yr_built", "yr_renovated", "zipcode",
    "lat", "long", "sqft_living15", "sqft_lot15"
]
MIN_REQUIRED = ["sqft_living", "bedrooms", "bathrooms", "zipcode"]


def build_row_full(src_row: Dict[str, Any]) -> Dict[str, Any]:
    row = {k: src_row.get(k) for k in HOUSE_ROW_COLS if k in src_row}
    if "zipcode" in row and row["zipcode"] is not None:
        row["zipcode"] = str(row["zipcode"])
    return row


def build_row_minimal(src_row: Dict[str, Any]) -> Dict[str, Any]:
    row = {k: src_row.get(k) for k in MIN_REQUIRED}
    missing = [k for k, v in row.items() if v is None]
    if missing:
        raise ValueError(f"Row missing required minimal fields: {missing}")
    row["zipcode"] = str(row["zipcode"])
    return row


def replicate_payloads(base_payloads: List[Dict[str, Any]], target: int) -> List[Dict[str, Any]]:
    """Repeat payloads cyclically until length == target."""
    if not base_payloads:
        return []
    if len(base_payloads) >= target:
        return base_payloads[:target]
    result = []
    i = 0
    while len(result) < target:
        result.append(base_payloads[i % len(base_payloads)])
        i += 1
    return result


async def send_one(
    client: httpx.AsyncClient,
    url: str,
    payload: Dict[str, Any],
    sem: asyncio.Semaphore,
    latencies_ms: List[float],
    success_counter: List[int],
    failure_counter: List[int]
):
    async with sem:
        t0 = time.perf_counter()
        try:
            resp = await client.post(url, json=payload, timeout=TIMEOUT_SECS)
            resp.raise_for_status()
            js = resp.json()
            server_ms = js.get("latency_ms")
            latency_ms = float(server_ms) if server_ms is not None else (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(latency_ms)
            success_counter[0] += 1
        except httpx.HTTPStatusError as ex:
            failure_counter[0] += 1
            print(f"[HTTP ERROR] {ex.response.status_code}: {ex.response.text.strip()}")
        except httpx.RequestError as ex:
            failure_counter[0] += 1
            print(f"[REQUEST ERROR] {ex}")
        except Exception as ex:
            failure_counter[0] += 1
            print(f"[UNKNOWN ERROR] {ex}")



async def main_async():
    csv_path = Path(CSV_PATH)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        print("No rows in CSV.")
        return

    # Build base payloads from all rows you have (100)
    base_payloads: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        if ENDPOINT == "/predict":
            base_payloads.append({"row": build_row_full(row_dict)})
        else:
            base_payloads.append({"row": build_row_minimal(row_dict)})

    # Replicate to reach BATCH_SIZE=400
    payloads = replicate_payloads(base_payloads, BATCH_SIZE)

    url = f"{BASE_URL}{ENDPOINT}"
    sem = asyncio.Semaphore(CONCURRENCY)
    latencies_ms: List[float] = []
    success_counter = [0]
    failure_counter = [0]

    print(f"[info] CSV rows: {len(df)} → sending {BATCH_SIZE} requests per wave (recycling rows).")
    print(f"[info] CONCURRENCY={CONCURRENCY}, DURATION={DURATION_SECS}s → {url}")

    start = time.perf_counter()
    wave_idx = 0

    async with httpx.AsyncClient() as client:
        while (time.perf_counter() - start) < DURATION_SECS:
            wave_idx += 1
            wave_start = time.perf_counter()

            tasks = [
                asyncio.create_task(send_one(client, url, p, sem, latencies_ms, success_counter, failure_counter))
                for p in payloads
            ]
            await asyncio.gather(*tasks, return_exceptions=False)

            wave_time = time.perf_counter() - wave_start
            rps = BATCH_SIZE / wave_time if wave_time > 0 else float("inf")
            print(f"[wave {wave_idx}] {BATCH_SIZE} requests in {wave_time:.2f}s "
                  f"({rps:.1f} req/s) succ={success_counter[0]} fail={failure_counter[0]}")

    duration = time.perf_counter() - start
    total = success_counter[0] + failure_counter[0]
    avg_lat = (sum(latencies_ms) / len(latencies_ms)) if latencies_ms else 0.0
    

    print("\n[summary]")
    print(f"  Duration:     {duration:.2f}s")
    print(f"  Total sent:   {total}")
    print(f"  Success:      {success_counter[0]}")
    print(f"  Failures:     {failure_counter[0]}")
    print(f"  Overall RPS:  {total / duration:.1f}")
    print(f"  Avg latency:  {avg_lat:.1f} ms")    


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
