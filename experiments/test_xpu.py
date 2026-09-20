import torch
import time

def test_xpu():
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        print("XPU not available.")
        return

    print("Testing Intel Arc XPU capabilities...")
    device = torch.device("xpu")
    print(f"Device name: {torch.xpu.get_device_name(0)}")

    for dtype in [torch.bfloat16, torch.float16, torch.float32]:
        try:
            print(f"\nTesting dtype: {dtype} on XPU...")
            x = torch.randn(128, 3072, dtype=dtype, device=device)
            w = torch.randn(3072, 3072, dtype=dtype, device=device)
            torch.xpu.synchronize()
            t0 = time.perf_counter()
            for _ in range(20):
                y = torch.matmul(x, w)
            torch.xpu.synchronize()
            t1 = time.perf_counter()
            print(f"Success! 20 matmuls took {(t1 - t0)*1000:.2f}ms")
        except Exception as e:
            print(f"Failed with {dtype}: {e}")

if __name__ == "__main__":
    test_xpu()
