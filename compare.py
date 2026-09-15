"""Compare the C++ and Python gradient-descent outputs.

Builds + runs gradient-descent-cpp, runs gradient-descent-python/main.py,
parses `final: m = ..., b = ...` (C++) and matching lines (Python),
then reports whether they agree within tolerance.
"""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
CPP_DIR = ROOT / "gradient-descent-cpp"
PY_MAIN = ROOT / "gradient-descent-python" / "main.py"
TOL = 1e-2


def run_cpp():
    subprocess.run(["make"], cwd=CPP_DIR, check=True, capture_output=True)
    out = subprocess.run(
        [str(CPP_DIR / "gd")], capture_output=True, text=True, check=True
    ).stdout
    subprocess.run(["make", "clean"], cwd=CPP_DIR, check=True, capture_output=True)
    return out


def run_python():
    venv_py = ROOT / ".venv" / "bin" / "python"
    exe = str(venv_py) if venv_py.exists() else sys.executable
    out = subprocess.run([exe, str(PY_MAIN)], capture_output=True, text=True).stdout
    return out


def parse_params(text):
    """Grab m, b from `final:` line, plus sklearn line if present."""
    final = re.search(r"final:\s*m\s*=\s*([\d.eE+-]+),\s*b\s*=\s*([\d.eE+-]+)", text)
    sk = re.search(r"sklearn:\s*m\s*=\s*([\d.eE+-]+),\s*b\s*=\s*([\d.eE+-]+)", text)
    m, b = (float(final.group(1)), float(final.group(2))) if final else (None, None)
    sm, sb = (float(sk.group(1)), float(sk.group(2))) if sk else (None, None)
    return (m, b), (sm, sb)


def main():
    cpp_out = run_cpp()
    py_out = run_python()

    print("================ C++ ================")
    print(cpp_out.strip())
    print("================ Python ================")
    print(py_out.strip())

    (cpp_m, cpp_b), _ = parse_params(cpp_out)
    (py_m, py_b), (sk_m, sk_b) = parse_params(py_out)

    def fmt(v):
        return f"{v:10.6f}" if v is not None else "       n/a"

    print("\n================ Comparison ================")
    print(f"  {'method':<10}  {'m':>10}  {'b':>10}")
    print(f"  {'----------':<10}  {'----------':>10}  {'----------':>10}")
    print(f"  {'C++ GD':<10}  {fmt(cpp_m)}  {fmt(cpp_b)}")
    print(f"  {'Python GD':<10}  {fmt(py_m)}  {fmt(py_b)}")
    print(f"  {'sklearn':<10}  {fmt(sk_m)}  {fmt(sk_b)}")
    print()

    ok = True
    for name, a, b in [("m (C++ vs Py GD)", cpp_m, py_m),
                       ("b (C++ vs Py GD)", cpp_b, py_b),
                       ("m (Py GD vs sklearn)", py_m, sk_m),
                       ("b (Py GD vs sklearn)", py_b, sk_b)]:
        if a is None or b is None:
            print(f"  {name:<22} could not parse -> FAIL")
            ok = False
        else:
            d = abs(a - b)
            status = "OK" if d < TOL else "FAIL"
            if d >= TOL:
                ok = False
            print(f"  {name:<22} |diff| = {d:.6f} -> {status}")

    print(f"\n  MATCH: {ok} (tol = {TOL})")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
