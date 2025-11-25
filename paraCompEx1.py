# PRAM search implementations and step counts
from math import ceil, log2
import random
import string
from typing import Tuple, Optional, List, Dict
import matplotlib.pyplot as plt



def generate_unique_string(n: int) -> str:
    """Generate a string of length n with unique characters.
    Uses ASCII letters + digits; if n exceeds available symbols, extend by unique unicode code points.
    """
    base = list(string.ascii_letters + string.digits)
    if n <= len(base):
        random.shuffle(base)
        return ''.join(base[:n])
    # Extend deterministically with unique code points beyond base
    extra = [chr(0x2500 + i) for i in range(n - len(base))]  # box-drawing chars
    random.shuffle(base)
    return ''.join(base + extra)


def seq_search(s: str, target: str) -> Tuple[int, int]:
    """Sequential linear search.
    Returns (index or -1, steps) where steps = number of comparisons performed.
    """
    steps = 0
    for i, ch in enumerate(s):
        steps += 1
        if ch == target:
            return i, steps
    return -1, steps


def pram_erew_search(s: str, target: str, p: int) -> Tuple[int, int]:
    """EREW PRAM search simulation.
    - Distribute indices to processors in blocks (disjoint reads).
    - Each processor checks its assigned index in 1 step when p >= n; else ceil(n/p) steps.
    - Combine with EREW reduction to select found index: O(log p) steps.
    Returns (index or -1, parallel_steps).
    """
    n = len(s)
    # Search phase steps
    search_steps = ceil(n / p)
    # Determine found index (functional result)
    idx = s.find(target)
    # Reduction phase: if found, reduce min over processors; else reduce OR over flags
    reduction_steps = ceil(log2(p)) if p > 1 else 0
    return idx, search_steps + reduction_steps


def pram_crew_search(s: str, target: str, p: int) -> Tuple[int, int]:
    """CREW PRAM search simulation.
    - Concurrent reads permitted; with p >= n, all indices can be checked in 1 step.
    - Exclusive writes: final aggregation via reduction O(log p).
    Returns (index or -1, parallel_steps).
    """
    n = len(s)
    search_steps = ceil(n / p)
    idx = s.find(target)
    reduction_steps = ceil(log2(p)) if p > 1 else 0
    return idx, search_steps + reduction_steps


def pram_crcw_search(s: str, target: str, p: int, rule: str = 'priority') -> Tuple[int, int]:
    """CRCW PRAM search simulation.
    - Concurrent reads and writes allowed.
    - With priority rule and unique characters, the matching processor writes its index in the same step.
    Parallel steps:
      - search + write: 1 step when p >= n; else ceil(n/p) steps.
    Returns (index or -1, parallel_steps).
    """
    n = len(s)
    idx = s.find(target)
    steps = ceil(n / p)
    # Under CRCW priority, aggregation is done in the same step.
    return idx, steps


def speedups(n: int, p: int, target_present: bool = True) -> Dict[str, Tuple[int, float, int]]:
    """Compute indices, parallel steps, and speedups for all models.
    Returns a dict mapping model -> (index, speedup, steps).
    """
    s = generate_unique_string(n)
    target = s[random.randrange(n)] if target_present else '#'

    idx_seq, t_seq = seq_search(s, target)
    idx_erew, t_erew = pram_erew_search(s, target, p)
    idx_crew, t_crew = pram_crew_search(s, target, p)
    idx_crcw, t_crcw = pram_crcw_search(s, target, p)

    assert idx_seq == idx_erew == idx_crew == idx_crcw or idx_seq == -1, "Inconsistent search indices"

    def spd(ts: int) -> float:
        return (t_seq / ts) if ts > 0 else float('inf')

    return {
        'Sequential': (idx_seq, 1.0, t_seq),
        'EREW': (idx_erew, spd(t_erew), t_erew),
        'CREW': (idx_crew, spd(t_crew), t_crew),
        'CRCW': (idx_crcw, spd(t_crcw), t_crcw),
    }

if __name__ == "__main__":
 # Visualize speedup for PRAM models vs sequential

# Parameters
    n = 256   # string length (unique characters)
    p = n     # processors

    results = speedups(n, p, target_present=True)

    # Print summary
    for model, (idx, sp, steps) in results.items():
        print(f"{model:10s} -> index={idx}, steps={steps}, speedup={sp:.2f}")

    # Prepare bar chart
    labels = ["EREW", "CREW", "CRCW"]
    spd_vals = [results[m][1] for m in labels]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, spd_vals, color=["#1f77b4", "#ff7f0e", "#2ca02c"]) 
    ax.set_ylabel("Speedup vs Sequential (T_seq / T_model)")
    ax.set_title(f"PRAM Search Speedup (n={n}, p={p})")
    for i, v in enumerate(spd_vals):
        ax.text(i, v + 0.02, f"{v:.1f}", ha='center', va='bottom', fontsize=9)
    plt.show()   