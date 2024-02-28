
import time
import pynvml
 
def get_max_gpu_memory_usage():
    pynvml.nvmlInit()
    
    try:
        gpu_count = pynvml.nvmlDeviceGetCount()
        max_memory_usage = [0] * gpu_count
        mean_memory_usage = [0] * gpu_count
        cnt = 0
 
        while True:
            for i in range(gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory_used = memory_info.used
 
                if memory_used > max_memory_usage[i]:
                    max_memory_usage[i] = memory_used

                mean_memory_usage[i] += memory_used

            cnt += 1
            # Delay to control the monitoring interval
            time.sleep(0.1)

    except KeyboardInterrupt:
        for i in range(gpu_count):
            mean_memory_usage[i] /= cnt
        return max_memory_usage, mean_memory_usage
    finally:
        pynvml.nvmlShutdown()
 
if __name__ == "__main__":
    max_memory_usage, mean_memory_usage = get_max_gpu_memory_usage()
    
    for i, (max_memory, mean_memory) in enumerate(zip(max_memory_usage, mean_memory_usage)):
        print(f"Maximum GPU {i} memory usage: {max_memory / 1024 / 1024} MB")
        print(f"Average GPU {i} memory usage: {mean_memory / 1024 / 1024} MB")
