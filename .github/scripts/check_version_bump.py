#!/usr/bin/env python3
"""
Check version bump between two commits for pyproject.toml and src/quanteval/__init__.py.

Writes two outputs to the GitHub Actions runner via the GITHUB_OUTPUT file:
- should_release: 'true' or 'false'
- version: the new version string when should_release is 'true'

This script is intended to be invoked from a workflow step which sets
the environment variables `GITHUB_BEFORE` and `GITHUB_AFTER`.
"""
import os
import re
import subprocess
import sys


def read_at(sha, path):
    # If sha is empty or all zeros, fallback to reading from the working tree
    if not sha or set(sha) == {"0"}:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return ""
    try:
        out = subprocess.check_output(["git", "show", f"{sha}:{path}"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8")
    except subprocess.CalledProcessError:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return ""


def extract_version_pyproject(text: str) -> str:
    if not text:
        return ""
    m = re.search(r"^version\s*=\s*['\"]([^'\"]+)['\"]", text, re.M)
    if m:
        return m.group(1)
    m = re.search(r"^\s*version\s*=\s*['\"]([^'\"]+)['\"]", text, re.M)
    return m.group(1) if m else ""


def extract_version_init(text: str) -> str:
    if not text:
        return ""
    m = re.search(r"__version__\s*=\s*['\"]([^'\"]+)['\"]", text)
    return m.group(1) if m else ""


def write_output(pairs: dict):
    out_path = os.environ.get("GITHUB_OUTPUT")
    for k, v in pairs.items():
        line = f"{k}={v}\n"
        if out_path:
            try:
                with open(out_path, "a") as f:
                    f.write(line)
            except Exception:
                pass
        # Also print for visibility
        print(line.strip())


def main():
    before = os.environ.get("GITHUB_BEFORE", "")
    after = os.environ.get("GITHUB_AFTER", os.environ.get("GITHUB_SHA", ""))

    py = "pyproject.toml"
    init = "src/quanteval/__init__.py"

    old_py = read_at(before, py)
    new_py = read_at(after, py)
    old_init = read_at(before, init)
    new_init = read_at(after, init)

    old_py_v = extract_version_pyproject(old_py)
    new_py_v = extract_version_pyproject(new_py)
    old_init_v = extract_version_init(old_init)
    new_init_v = extract_version_init(new_init)

    print("old_py_v=", old_py_v)
    print("new_py_v=", new_py_v)
    print("old_init_v=", old_init_v)
    print("new_init_v=", new_init_v)

    should_release = False
    version = ""
    if new_py_v and new_init_v and new_py_v == new_init_v:
        if new_py_v != old_py_v and new_init_v != old_init_v:
            should_release = True
            version = new_py_v

    if should_release:
        write_output({"should_release": "true", "version": version})
    else:
        write_output({"should_release": "false"})


if __name__ == "__main__":
    main()
