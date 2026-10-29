#!/usr/bin/env python3
"""Publish the GLiDE MLX port (run on the Mac from the repo root, with an HF write login for netprojects and gh logged in).

    python scripts/publish.py --models ~/local-llms/glide            # dry run: checks everything, prints the plan
    python scripts/publish.py --models ~/local-llms/glide --go       # do it

Each step is skipped when already done, so a re-run resumes:
  1. Hugging Face: create netprojects/GLiDE-mlx-8bit and -4bit (public) and upload each build with its card as one commit.
  2. GitHub: create internet-projects/glide-mlx (public), push main, tag v0.1.0, create the release.
"""
import argparse
import fnmatch
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(HERE)
GH_REPO, TAG = "internet-projects/glide-mlx", "v0.1.0"
BUILDS = [("netprojects/GLiDE-mlx-8bit", "mlx-q8", "hf_cards/q8.md"), ("netprojects/GLiDE-mlx-4bit", "mlx-q4", "hf_cards/q4.md")]
SKIP = ["README.md", ".DS_Store", "._*", "*.pyc", "__pycache__/*"]
NEED = ["config.json", "model.safetensors.index.json", "readout.safetensors", "decision_config.json", "tokenizer.json",
        "chat_template.jinja", "LICENSE", "NOTICE", "licenses/Qwen3.8-27B-LICENSE"]
COMMIT_MSG = "GLiDE v1.2 (fastino/GLiDE d199b6d) MLX {bits}: weights, card, Fastino Community License files"


def sh(*cmd, check=True):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if check and r.returncode:
        sys.exit(f"failed: {' '.join(cmd)}\n{r.stdout}{r.stderr}")
    return r


def files_of(folder):
    out = []
    for root, _, names in os.walk(folder):
        for n in names:
            rel = os.path.relpath(os.path.join(root, n), folder)
            if not any(fnmatch.fnmatch(rel, p) or fnmatch.fnmatch(n, p) for p in SKIP):
                out.append(rel)
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="folder holding mlx-q8/ and mlx-q4/")
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args()
    from huggingface_hub import CommitOperationAdd, HfApi
    api = HfApi()

    who = api.whoami()
    orgs = [o.get("name") for o in who.get("orgs", [])]
    role = ((who.get("auth") or {}).get("accessToken") or {}).get("role")
    print(f"HF login: {who.get('name')} (orgs: {', '.join(orgs)}), token role: {role}")
    if "netprojects" not in orgs:
        sys.exit("this HF login is not in the netprojects org")
    if role == "read":
        sys.exit("read-only HF token: log in with the netprojects-write token (hf auth login)")
    if sh("gh", "auth", "status", check=False).returncode:
        sys.exit("gh is not logged in (gh auth login)")
    if sh("git", "status", "--porcelain", "--untracked-files=no").stdout.strip():
        sys.exit("uncommitted changes in the repo")
    plan = []
    for repo, sub, card in BUILDS:
        folder = os.path.join(a.models, sub)
        fs = files_of(folder)
        missing = [n for n in NEED if n not in fs]
        if missing or not any(f.endswith(".safetensors") and f.startswith("model-") for f in fs):
            sys.exit(f"{folder} is incomplete: missing {missing}")
        size = sum(os.path.getsize(os.path.join(folder, f)) for f in fs) / 1e9
        exists = api.repo_exists(repo)
        done = exists and any(c.title.startswith(COMMIT_MSG.split(" MLX")[0]) for c in api.list_repo_commits(repo))
        plan.append(f"HF {repo}: " + ("already published" if done else f"{'create, ' if not exists else ''}upload {len(fs)} files ({size:.1f} GB) + {card}"))
    gh_exists = sh("gh", "repo", "view", GH_REPO, check=False).returncode == 0
    plan.append(f"GitHub {GH_REPO}: " + ("exists; push main and tag" if gh_exists else "create (public), push main, tag") + f" {TAG}, release")
    print("plan:\n  " + "\n  ".join(plan))
    if not a.go:
        print("dry run: nothing changed. Re-run with --go.")
        return

    for repo, sub, card in BUILDS:
        folder = os.path.join(a.models, sub)
        api.create_repo(repo, private=False, exist_ok=True)
        if any(c.title.startswith(COMMIT_MSG.split(" MLX")[0]) for c in api.list_repo_commits(repo)):
            print(f"HF {repo}: already published")
            continue
        ops = [CommitOperationAdd(path_in_repo=f, path_or_fileobj=os.path.join(folder, f)) for f in files_of(folder)]
        ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card))
        print(f"HF {repo}: uploading {len(ops)} files ...", flush=True)
        oid = api.create_commit(repo, operations=ops, commit_message=COMMIT_MSG.format(bits="8-bit" if "8bit" in repo else "4-bit")).oid
        print(f"HF {repo}: published {oid[:7]}")

    sh("git", "branch", "-M", "main")
    if not gh_exists:
        sh("gh", "repo", "create", GH_REPO, "--public", "--description",
           "GLiDE (Fastino's decision model) on Apple Silicon with MLX: 8-bit and 4-bit builds, /v1/systemone server. Not affiliated with Fastino.")
    if sh("git", "remote", "get-url", "origin", check=False).returncode:
        sh("git", "remote", "add", "origin", f"https://github.com/{GH_REPO}.git")
    sh("git", "push", "-u", "origin", "main")
    if sh("git", "rev-parse", "-q", "--verify", f"refs/tags/{TAG}", check=False).returncode:
        sh("git", "tag", "-a", TAG, "-m", "GLiDE v1.2 MLX port: 8-bit and 4-bit builds")
    sh("git", "push", "origin", f"refs/tags/{TAG}")
    if sh("gh", "release", "view", TAG, "-R", GH_REPO, check=False).returncode:
        sh("gh", "release", "create", TAG, "-R", GH_REPO, "--title", f"{TAG} — GLiDE v1.2 on MLX", "--notes",
           "First release. 8-bit: 202/204 same top answer as Fastino's code, JevBench public 215/231. "
           "4-bit: 197/204, 213/231. Weights: huggingface.co/netprojects/GLiDE-mlx-8bit and -4bit. "
           "Built with Fastino GLiDE; Fastino Community License. Not affiliated with Fastino.")
    print(f"GitHub: https://github.com/{GH_REPO} main + {TAG} + release")
    print("done.")


if __name__ == "__main__":
    main()
