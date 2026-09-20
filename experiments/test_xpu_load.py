import torch
from zip2zip import Zip2ZipModel, Zip2ZipTokenizer

def test_load_to_xpu():
    model_id = "epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1"
    print("Loading model on CPU with bfloat16...")
    model = Zip2ZipModel.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    print("Loaded on CPU. Checking memory before moving to XPU...")
    free_mem = torch.xpu.get_device_properties(0).total_memory - torch.xpu.memory_allocated(0)
    print(f"XPU free memory: {free_mem / (1024**3):.2f} GB")
    
    try:
        print("Moving model to XPU...")
        model.to("xpu")
        print("Successfully moved to XPU!")
        allocated = torch.xpu.memory_allocated(0) / (1024**3)
        print(f"XPU allocated: {allocated:.2f} GB")
    except Exception as e:
        print(f"Moving to XPU failed: {e}")

if __name__ == "__main__":
    test_load_to_xpu()
