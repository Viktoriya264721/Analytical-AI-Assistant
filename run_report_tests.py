import random
import subprocess
import sys

MONTHS = [
    "2025-06",
    "2025-08",
    "2025-10",
    "2025-11",
    "2025-12",
    "2026-02",
    "2026-04",
]

MODEL = "sonnet"
JUDGE_MODEL = "gpt-4o-mini"
RUNS = 3

shuffled = MONTHS[:]
random.shuffle(shuffled)
print(f"Порядок запуску: {' → '.join(shuffled)}")

for i, month in enumerate(shuffled, 1):
    print(f"\n{'='*72}")
    print(f"  [{i}/{len(MONTHS)}] Місяць: {month}")
    print(f"{'='*72}\n")

    result = subprocess.run(
        [sys.executable, "tests/test_report.py",
         "--month", month,
         "--model", MODEL,
         "--judge-model", JUDGE_MODEL,
         "--runs", str(RUNS)],
    )

    if result.returncode != 0:
        print(f"\n  ПОМИЛКА для місяця {month} (код: {result.returncode})")
