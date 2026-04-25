from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_folder


def main() -> None:
    parser = argparse.ArgumentParser(description="Push local artifact folder to HF Hub.")
    parser.add_argument("--repo-id", required=True, help="team/the-examiner")
    parser.add_argument("--local-dir", required=True, help="Folder to upload")
    parser.add_argument("--repo-type", default="model", choices=["model", "dataset", "space"])
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN missing.")

    local_dir = Path(args.local_dir)
    if not local_dir.exists():
        raise FileNotFoundError(f"{local_dir} does not exist.")

    create_repo(repo_id=args.repo_id, repo_type=args.repo_type, token=token, exist_ok=True)
    upload_folder(
        repo_id=args.repo_id,
        folder_path=str(local_dir),
        repo_type=args.repo_type,
        token=token,
        ignore_patterns=["*.mp4", "*.avi", "*.mov", "*.webm"],
    )

    api = HfApi(token=token)
    files = api.list_repo_files(repo_id=args.repo_id, repo_type=args.repo_type)
    blocked_ext = (".mp4", ".avi", ".mov", ".webm")
    if any(path.lower().endswith(blocked_ext) for path in files):
        raise RuntimeError("MSR-9 violation: video file found in HF repo.")


if __name__ == "__main__":
    main()
