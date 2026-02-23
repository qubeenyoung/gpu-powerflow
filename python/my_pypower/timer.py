from time import perf_counter
import sys

class BlockTimer:
    def __init__(self, enabled, tag, op_name, iter_idx):
        self.enabled = enabled
        self.tag = tag
        self.op = op_name
        self.iter = iter_idx
        self.start = None

    def __enter__(self):
        if self.enabled:
            self.start = perf_counter()
        return self

    def __exit__(self, *args):  
        if self.enabled and self.start is not None:
            elapsed = perf_counter() - self.start
            print(f"{self.tag} {self.op} {self.iter} {elapsed:.6f}",
                    file=sys.stdout, flush=True)
            
def log_time(op_name: str, iter_idx: int, start_time: float):
    """경과시간 출력"""
    elapsed = perf_counter() - start_time
    print(f"{op_name} {iter_idx} {elapsed:.6f}", file=sys.stdout, flush=True)