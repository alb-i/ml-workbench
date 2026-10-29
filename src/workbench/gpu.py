"""GPU stats via nvidia-ml-py (pynvml)."""

import pynvml

_initialized = False


def gpu_stats() -> list[dict]:
    """Utilisation, memory and temperature per GPU; empty list if NVML is not available."""
    global _initialized
    try:
        if not _initialized:
            pynvml.nvmlInit()
            _initialized = True
        stats = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            stats.append(
                {
                    "index": i,
                    "name": pynvml.nvmlDeviceGetName(h),
                    "util": pynvml.nvmlDeviceGetUtilizationRates(h).gpu,
                    "mem_used_gb": round(mem.used / 2**30, 2),
                    "mem_total_gb": round(mem.total / 2**30, 2),
                    "temp_c": pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU),
                }
            )
        return stats
    except pynvml.NVMLError:
        return []
