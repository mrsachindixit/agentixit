import concurrent.futures
import time


def get_customer_profile(customer_id: str) -> dict:
    time.sleep(0.8)
    return {"customer_id": customer_id, "tier": "gold", "region": "APAC"}


def get_latest_invoice(customer_id: str) -> dict:
    time.sleep(1.2)
    return {"customer_id": customer_id, "invoice_id": "INV-2048", "status": "paid"}


def get_support_summary(customer_id: str) -> dict:
    time.sleep(0.6)
    return {"customer_id": customer_id, "open_tickets": 1}


def run_parallel(customer_id: str, timeout_seconds: float = 1.0) -> dict:
    tools = {
        "profile": lambda: get_customer_profile(customer_id),
        "invoice": lambda: get_latest_invoice(customer_id),
        "support": lambda: get_support_summary(customer_id),
    }

    started = time.perf_counter()
    results: dict = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_map = {executor.submit(fn): name for name, fn in tools.items()}
        done, not_done = concurrent.futures.wait(
            future_map.keys(), timeout=timeout_seconds, return_when=concurrent.futures.ALL_COMPLETED
        )

        for future in done:
            name = future_map[future]
            try:
                results[name] = future.result()
            except Exception as e:
                results[name] = {"error": str(e)}

        for future in not_done:
            name = future_map[future]
            results[name] = {"error": f"timeout>{timeout_seconds}s"}
            future.cancel()

    elapsed = time.perf_counter() - started
    print(f"Parallel elapsed: {elapsed:.2f}s")
    return results


if __name__ == "__main__":
    print("Timeout = 1.0s")
    print(run_parallel("cust-123", timeout_seconds=1.0))

    print("\n" + "=" * 70)

    print("Timeout = 2.0s")
    print(run_parallel("cust-123", timeout_seconds=2.0))
