"""Repository consistency checks for HARVI-ECG (no patient-level data needed).

Run from the repository root:  python checks/consistency.py
Exit code 0 = all checks passed. Used by .github/workflows/consistency.yml.

Checks
  1. every Python file parses
  2. README.md is exactly what docs/make_readme.py generates from results/
  3. the data embedded in docs/index.html equals docs/site_data.json
  4. the website loads no external resources (scripts, styles, images, fonts)
  5. every local link in the README, REPRODUCE and the website resolves
  6. the report copy served by the website is identical to report/harvi_results_report.pdf
  7. CITATION.cff is valid against the official CFF schema (needs `cffconvert`)
"""
import ast, json, os, re, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
results = []


def check(name):
    def deco(fn):
        try:
            msg = fn()
            results.append((True, name, msg or ""))
        except Exception as e:  # noqa: BLE001 - report every failure, keep going
            results.append((False, name, str(e)))
        return fn
    return deco


def files(ext, skip=(".git", "__pycache__")):
    for dp, dn, fs in os.walk("."):
        dn[:] = [d for d in dn if d not in skip]
        for f in fs:
            if f.endswith(ext):
                yield os.path.join(dp, f)[2:]


@check("Python files parse")
def _():
    bad = []
    for p in files(".py"):
        try:
            ast.parse(open(p, encoding="utf-8").read(), p)
        except SyntaxError as e:
            bad.append(f"{p}:{e.lineno}")
    assert not bad, "syntax errors: " + ", ".join(bad)
    return f"{sum(1 for _ in files('.py'))} files"


@check("README.md regenerates from results/")
def _():
    before = open("README.md", "rb").read()
    try:
        r = subprocess.run([sys.executable, "docs/make_readme.py"], capture_output=True, text=True)
        assert r.returncode == 0, "docs/make_readme.py failed: " + r.stderr[-400:]
        after = open("README.md", "rb").read()
    finally:
        open("README.md", "wb").write(before)  # never leave the checkout modified
    if after != before:
        a, b = before.decode().splitlines(), after.decode().splitlines()
        i = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
        raise AssertionError(f"README.md differs from the generator output from line {i + 1}; "
                             "edit docs/make_readme.py (or results/) and re-run it instead of editing README.md")


INDEX = open("docs/index.html", encoding="utf-8").read()


@check("website data == docs/site_data.json")
def _():
    m = re.search(r'<script id="data" type="application/json">(.*?)</script>', INDEX, re.S)
    assert m, "no embedded data block in docs/index.html"
    assert json.loads(m.group(1)) == json.load(open("docs/site_data.json", encoding="utf-8")), \
        "embedded data and docs/site_data.json differ"


@check("website loads no external resources")
def _():
    ext = re.findall(r'<(?:script|img|source|iframe|link)\b[^>]*\b(?:src|href)="((?:https?:)?//[^"]+)"', INDEX)
    ext += re.findall(r'(?:@import|url\()\s*["\']?((?:https?:)?//[^"\')\s]+)', INDEX)
    assert not ext, "external resources: " + ", ".join(sorted(set(ext)))


@check("local links resolve")
def _():
    missing = []
    def scan(path, text, base):
        refs = re.findall(r'(?:src|href)="([^"]+)"', text) + re.findall(r"\]\(([^)\s]+)\)", text)
        for ref in refs:
            if re.match(r"^(?:[a-z]+:|#|//)", ref) or ref.startswith("${"):
                continue
            target = os.path.normpath(os.path.join(base, ref.split("#")[0].split("?")[0]))
            if not os.path.exists(target):
                missing.append(f"{path} -> {ref}")
    for p in ["README.md", "REPRODUCE.md"] + sorted(files(".md")):
        if os.path.exists(p) and not p.startswith(".github"):
            scan(p, open(p, encoding="utf-8").read(), os.path.dirname(p))
    scan("docs/index.html", INDEX, "docs")
    assert not missing, "; ".join(sorted(set(missing)))


@check("website report copy == report/harvi_results_report.pdf")
def _():
    a, b = "report/harvi_results_report.pdf", "docs/report/harvi_results_report.pdf"
    assert open(a, "rb").read() == open(b, "rb").read(), f"{b} is out of date; copy {a} over it"


@check("CITATION.cff valid (CFF schema)")
def _():
    r = subprocess.run(["cffconvert", "--validate", "-i", "CITATION.cff"], capture_output=True, text=True)
    assert r.returncode == 0, (r.stdout + r.stderr).strip()[-600:]


w = max(len(n) for _, n, _ in results)
for ok, name, msg in results:
    print(f"{'PASS' if ok else 'FAIL'}  {name:<{w}}  {msg}")
sys.exit(0 if all(ok for ok, _, _ in results) else 1)
