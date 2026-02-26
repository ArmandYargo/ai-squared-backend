# dev_ingest_docs.py
from __future__ import annotations

import time
from pathlib import Path
from typing import List, Set

from agent.rag import RagStore


def _gather_kb_files(kb_dir: Path) -> List[Path]:
    exts = (".pdf", ".txt", ".md", ".docx")
    files = [p for p in kb_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    return sorted(files)


def _resolved_path_set(paths: List[Path]) -> Set[str]:
    return set(str(p.resolve()) for p in paths)


def _fmt_seconds(sec: float) -> str:
    sec = max(0.0, float(sec))
    if sec < 60:
        return f"{sec:.0f}s"
    m = int(sec // 60)
    s = int(sec % 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h = int(m // 60)
    m2 = int(m % 60)
    return f"{h}h {m2:02d}m"


def _bar(done: int, total: int, width: int = 28) -> str:
    total = max(1, total)
    done = max(0, min(done, total))
    filled = int(width * done / total)
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def main():
    root = Path(__file__).resolve().parent
    kb_dir = root / "Knowledge Base"

    print("\n=== Developer Knowledge Base Manager ===", flush=True)
    print(f"Knowledge Base folder: {kb_dir}", flush=True)

    if not kb_dir.exists():
        print("\n❌ Folder not found.", flush=True)
        print("Create a folder named 'Knowledge Base' in the same directory as dev_ingest_docs.py.", flush=True)
        return

    files = _gather_kb_files(kb_dir)
    print(f"Found {len(files)} supported documents (.pdf/.txt/.md/.docx).", flush=True)

    rag = RagStore()
    namespace = "knowledge"

    print("\nChoose an action:", flush=True)
    print("  1) Ingest/Update only (no cleanup)", flush=True)
    print("  2) Cleanup deleted files only (no ingest)", flush=True)
    print("  3) Ingest/Update + Cleanup deleted files", flush=True)
    print("  4) Dry-run cleanup (show what would be deleted)", flush=True)
    print("  5) Exit", flush=True)

    choice = input("\nEnter choice (1-5): ").strip()

    if choice == "5" or choice.lower() in ("exit", "quit"):
        print("Exiting.", flush=True)
        return

    current_set = _resolved_path_set(files)

    # -----------------------
    # Cleanup-only modes
    # -----------------------
    if choice in ("2", "4"):
        dry_run = (choice == "4")

        def on_delete(p: str):
            print(f"\n🧹 Removing from DB: {Path(p).name}", flush=True)

        def on_deleted(p: str):
            print(f"✅ Deleted from DB: {Path(p).name}", flush=True)

        report = rag.cleanup_deleted_files(
            namespace=namespace,
            current_paths=current_set,
            dry_run=dry_run,
            on_delete=None if dry_run else on_delete,
            on_deleted=None if dry_run else on_deleted,
        )

        print("\n=== Cleanup Report ===", flush=True)
        print(f"Namespace: {report.get('namespace')}", flush=True)
        print(f"DB tracked files: {report.get('db_paths')}", flush=True)
        print(f"Current local files: {report.get('current_paths')}", flush=True)
        if report.get("dry_run"):
            print(f"Files to delete: {report.get('files_to_delete')}", flush=True)
            to_delete = report.get("to_delete") or []
            if to_delete:
                print("\nFirst items that would be deleted (capped):", flush=True)
                for p in to_delete:
                    print(" -", Path(p).name, flush=True)
        else:
            print(f"Files deleted: {report.get('files_deleted')}", flush=True)

        return

    # -----------------------
    # Ingest modes
    # -----------------------
    if choice not in ("1", "3"):
        print("\n❌ Invalid choice. Exiting.", flush=True)
        return

    cleanup_after = (choice == "3")

    if not files and not cleanup_after:
        print("\n⚠️ No documents to ingest.", flush=True)
        return

    total_files = len(files)
    processed = 0
    start_all = time.perf_counter()
    per_file_times: List[float] = []

    print("\n=== Ingest Run Started ===", flush=True)
    print(f"Cleanup deleted enabled: {cleanup_after}", flush=True)

    # Optional cleanup first (for choice 3): we run it BEFORE ingest so DB matches folder
    if cleanup_after:
        def on_delete(p: str):
            print(f"\n🧹 Removing from DB: {Path(p).name}", flush=True)

        def on_deleted(p: str):
            print(f"✅ Deleted from DB: {Path(p).name}", flush=True)

        print("\n--- Cleanup phase (deleted files) ---", flush=True)
        _ = rag.cleanup_deleted_files(
            namespace=namespace,
            current_paths=current_set,
            dry_run=False,
            on_delete=on_delete,
            on_deleted=on_deleted,
        )
        print("--- Cleanup phase complete ---\n", flush=True)

    # Ingest/update each file with progress output
    for idx, f in enumerate(files, start=1):
        file_name = f.name
        processed += 1

        # Rolling ETA based on avg time per processed doc (after first)
        avg = (sum(per_file_times) / len(per_file_times)) if per_file_times else None
        overall_remaining = (avg * (total_files - (processed - 1))) if avg else None

        print(f"\n[{idx}/{total_files}] Processing: {file_name}", flush=True)
        if overall_remaining is not None:
            print(f"Estimated time remaining (overall): {_fmt_seconds(overall_remaining)}", flush=True)

        t0 = time.perf_counter()

        # We show a 3-step progress bar per document (chunk/embed/upload).
        # The actual work happens inside rag.ingest_file, but we can still present
        # a useful "phase bar" and timing per phase. We approximate phase completion
        # by splitting total doc time into the 3 phases once done.
        phases = ["Chunking", "Embedding", "Uploading"]
        for i, ph in enumerate(phases, start=1):
            print(f"{_bar(i-1, len(phases))} {ph} ...", flush=True)

        # Do the work
        result = rag.ingest_file(f, namespace=namespace)

        t1 = time.perf_counter()
        dt = t1 - t0
        per_file_times.append(dt)

        # Re-print final bar with completion
        print(f"{_bar(len(phases), len(phases))} Completed in {_fmt_seconds(dt)}", flush=True)

        status = result.get("status")
        chunks = int(result.get("chunks_added") or 0)
        note = result.get("note")

        if status == "missing":
            print(f"❌ Missing file (skipped): {file_name}", flush=True)
        elif status == "skipped_unchanged":
            print(f"⏭️  Unchanged (skipped): {file_name}", flush=True)
        elif status == "updated":
            print(f"✅ Updated + re-ingested: {file_name} | Chunks: {chunks}", flush=True)
        elif status == "new":
            print(f"✅ Ingested new: {file_name} | Chunks: {chunks}", flush=True)
        else:
            print(f"✅ Processed: {file_name} | Status: {status} | Chunks: {chunks}", flush=True)

        if note:
            print(f"   Note: {note}", flush=True)

        # ETA for next document
        avg_now = sum(per_file_times) / len(per_file_times)
        remaining_docs = total_files - processed
        if remaining_docs > 0:
            print(f"Next file ETA (avg): {_fmt_seconds(avg_now)}", flush=True)
            print(f"Remaining docs: {remaining_docs} | Est remaining: {_fmt_seconds(avg_now * remaining_docs)}", flush=True)

    elapsed_all = time.perf_counter() - start_all
    print("\n=== Ingest Run Finished ===", flush=True)
    print(f"Processed files: {total_files}", flush=True)
    if per_file_times:
        print(f"Average per file: {_fmt_seconds(sum(per_file_times)/len(per_file_times))}", flush=True)
    print(f"Total elapsed: {_fmt_seconds(elapsed_all)}", flush=True)


if __name__ == "__main__":
    main()
