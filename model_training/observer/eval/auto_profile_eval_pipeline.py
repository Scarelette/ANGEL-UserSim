"""Automatic evaluation of generated long profiles.

short profiles --(stage-1 generator(s) x N)--> long profiles
  -> merge per short profile
  -> Observer symptom network per long profile (model_training.observer.predict_network)
  -> edge "reasonability": mean Yes-rate of the Azure fine-tuned edge classifier
  -> diversity: self-BLEU and embedding distance vs. number of generations k
  -> <output-root>/<run-name>/report.{json,md}

Run from the repository root:
    python -m model_training.observer.eval.auto_profile_eval_pipeline \
        --input data/profiles/short_profiles.jsonl --stage1-models models/Qwen3-Observer-800 \
        --generations-per-model 12 --run-name ours
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional


from angel_common.env import get_env
from angel_common.paths import OUTPUTS_DIR, REPO_ROOT, resolve_model

EVAL_PKG = "model_training.observer.eval"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse JSON in {path}:{line_no}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def normalize_repo_arg(value: str) -> str:
    value_path = Path(value)
    if value_path.is_absolute():
        try:
            return str(value_path.resolve().relative_to(REPO_ROOT))
        except ValueError:
            return value

    repo_prefix = REPO_ROOT.name + "/"
    if value.startswith(repo_prefix):
        return value[len(repo_prefix) :]
    return value


def slugify(value: str) -> str:
    value = value.strip().rstrip("/")
    if not value:
        return "model"
    value = value.split("/")[-1]
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return value.strip("._-") or "model"


def sort_generation_paths(paths: List[Path]) -> List[Path]:
    def key(path: Path) -> Any:
        match = re.search(r"(\d+)(?=\.jsonl$)", path.name)
        number = int(match.group(1)) if match else -1
        return (path.parent.name, number, path.name)

    return sorted(paths, key=key)


def expand_existing_long_profile_globs(patterns: List[str]) -> List[Path]:
    paths: List[Path] = []
    for pattern in patterns:
        normalized = normalize_repo_arg(pattern)
        matches = [Path(match) for match in glob.glob(str(REPO_ROOT / normalized))]
        if not matches:
            matches = [Path(match) for match in glob.glob(normalized)]
        if not matches and normalized.endswith("/.jsonl"):
            corrected = normalized[: -len(".jsonl")] + "*.jsonl"
            matches = [Path(match) for match in glob.glob(str(REPO_ROOT / corrected))]
            if not matches:
                matches = [Path(match) for match in glob.glob(corrected)]
            if matches:
                print(
                    f"Warning: corrected existing long-profile glob {pattern!r} to {corrected!r}",
                    file=sys.stderr,
                )
        if not matches:
            raise ValueError(f"No files matched existing long-profile glob: {pattern}")
        paths.extend(matches)

    unique_paths = {path.resolve(): path.resolve() for path in paths}
    return sort_generation_paths(list(unique_paths.values()))


def expand_existing_network_globs(patterns: List[str]) -> List[Path]:
    paths: List[Path] = []
    for pattern in patterns:
        normalized = normalize_repo_arg(pattern)
        matches = [Path(match) for match in glob.glob(str(REPO_ROOT / normalized))]
        if not matches:
            matches = [Path(match) for match in glob.glob(normalized)]
        matches = [
            path
            for path in matches
            if path.name.endswith(".jsonl")
            and "_err" not in path.stem
            and "_scored" not in path.stem
        ]
        if not matches:
            raise ValueError(f"No network JSONL files matched existing network glob: {pattern}")
        paths.extend(matches)

    unique_paths = {path.resolve(): path.resolve() for path in paths}
    return sort_generation_paths(list(unique_paths.values()))


def gpu_env(gpu: Optional[str]) -> Optional[Dict[str, str]]:
    if gpu is None:
        return None
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    return env


def command_prefix(gpu: Optional[str]) -> str:
    if gpu is None:
        return ""
    return f"CUDA_VISIBLE_DEVICES={gpu} "


def run_command(command: List[str], dry_run: bool = False, gpu: Optional[str] = None) -> None:
    printable = " ".join(command)
    print(f"\n$ {command_prefix(gpu)}{printable}", flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=REPO_ROOT, check=True, env=gpu_env(gpu))


def parse_gpu_list(raw_gpus: Optional[str], arg_name: str = "--gpus") -> List[str]:
    if raw_gpus is None:
        return []
    gpus = [gpu.strip() for gpu in raw_gpus.split(",") if gpu.strip()]
    if not gpus:
        raise ValueError(f"{arg_name} was provided but no GPU ids were found.")
    return gpus


def default_parallel_jobs(gpus: List[str]) -> int:
    return len(gpus) if gpus else 1


def run_parallel_jobs(
    jobs: List[Dict[str, Any]],
    gpus: List[str],
    max_parallel: int,
    dry_run: bool,
    job_kind: str,
    max_parallel_arg: str,
) -> None:
    if not jobs:
        return
    if max_parallel < 1:
        raise ValueError(f"{max_parallel_arg} must be at least 1.")

    if dry_run:
        for job_idx, job in enumerate(jobs):
            gpu = gpus[job_idx % len(gpus)] if gpus else None
            run_command(job["command"], dry_run=True, gpu=gpu)
        return

    pending = list(enumerate(jobs))
    running: List[Dict[str, Any]] = []
    next_gpu_idx = 0

    while pending or running:
        while pending and len(running) < max_parallel:
            job_idx, job = pending.pop(0)
            gpu = None
            if gpus:
                gpu = gpus[next_gpu_idx % len(gpus)]
                next_gpu_idx += 1

            command = job["command"]
            printable = " ".join(command)
            print(
                f"\nStarting {job_kind} job {job_idx + 1}/{len(jobs)} "
                f"on GPU {gpu if gpu is not None else 'default'}:\n$ {command_prefix(gpu)}{printable}",
                flush=True,
            )
            process = subprocess.Popen(command, cwd=REPO_ROOT, env=gpu_env(gpu))
            running.append({"process": process, "job": job, "gpu": gpu, "job_idx": job_idx})

        still_running: List[Dict[str, Any]] = []
        for active in running:
            process = active["process"]
            return_code = process.poll()
            if return_code is None:
                still_running.append(active)
                continue
            if return_code != 0:
                for other in running:
                    other_process = other["process"]
                    if other_process.poll() is None:
                        other_process.terminate()
                raise subprocess.CalledProcessError(return_code, active["job"]["command"])
            print(
                f"Finished {job_kind} job {active['job_idx'] + 1}/{len(jobs)}: "
                f"{active['job']['name']}",
                flush=True,
            )
        running = still_running

        if pending or running:
            time.sleep(5)


def count_successful_generation_rows(path: Path) -> int:
    return sum(
        1
        for row in read_jsonl(path)
        if isinstance(row.get("long_profile_description"), str)
        and row["long_profile_description"].strip()
        and not row.get("error")
    )


def expected_input_count(input_path: Path, limit: Optional[int]) -> int:
    total = len(read_jsonl(input_path))
    if limit is None:
        return total
    return min(limit, total)


def generate_profiles(args: argparse.Namespace, run_dir: Path) -> List[Path]:
    generation_paths: List[Path] = []
    generation_jobs: List[Dict[str, Any]] = []
    expected_rows = expected_input_count(Path(args.input), args.limit)
    gpus = parse_gpu_list(args.generation_gpus, "--generation-gpus")
    max_parallel = args.max_parallel_generations or default_parallel_jobs(gpus)

    for model in args.stage1_models:
        model_label = slugify(model)
        model_dir = run_dir / "long_profiles" / model_label
        model_dir.mkdir(parents=True, exist_ok=True)

        for generation_idx in range(1, args.generations_per_model + 1):
            output_path = model_dir / f"long_profiles_{generation_idx:03d}.jsonl"
            generation_paths.append(output_path)

            if args.resume and output_path.exists():
                actual_rows = count_successful_generation_rows(output_path)
                if actual_rows >= expected_rows:
                    print(f"Skipping existing generation: {rel(output_path)}")
                    continue

            command = [
                sys.executable,
                "-m", f"{EVAL_PKG}.short2long_profile_generation",
                "--input",
                args.input,
                "--output-text",
                rel(output_path),
                "--input-field",
                args.input_field,
                "--stage1-model",
                model,
                "--stage2-deployment",
                args.stage2_deployment,
            ]
            if args.limit is not None:
                command.extend(["--limit", str(args.limit)])
            generation_jobs.append(
                {
                    "command": command,
                    "output_path": output_path,
                    "name": rel(output_path),
                }
            )

    run_parallel_jobs(
        jobs=generation_jobs,
        gpus=gpus,
        max_parallel=max_parallel,
        dry_run=args.dry_run,
        job_kind="generation",
        max_parallel_arg="--max-parallel-generations",
    )

    return generation_paths


def combine_generation_outputs(generation_paths: List[Path], output_path: Path) -> int:
    rows: List[Dict[str, Any]] = []
    for generation_path in generation_paths:
        model_label = generation_path.parent.name
        generation_match = re.search(r"(\d+)", generation_path.stem)
        generation_index = int(generation_match.group(1)) if generation_match else None

        for row in read_jsonl(generation_path):
            if row.get("error") or not isinstance(row.get("long_profile_description"), str):
                continue
            row = dict(row)
            row["model_label"] = model_label
            row["generation_index"] = generation_index
            row["generation_file"] = rel(generation_path)
            rows.append(row)

    write_jsonl(output_path, rows)
    return len(rows)


def run_merge(combined_path: Path, output_path: Path, dry_run: bool) -> None:
    command = [
        sys.executable,
        "-m", f"{EVAL_PKG}.merge_long_profiles",
        "--input-glob",
        rel(combined_path),
        "--output",
        rel(output_path),
    ]
    run_command(command, dry_run=dry_run)


def effective_network_gpus(args: argparse.Namespace) -> List[str]:
    if args.network_gpus:
        return parse_gpu_list(args.network_gpus, "--network-gpus")
    if args.network_gpu:
        return parse_gpu_list(args.network_gpu, "--network-gpu")
    return []


def split_rows_for_network(
    input_path: Path,
    shard_dir: Path,
    num_shards: int,
    limit: Optional[int],
    dry_run: bool,
) -> List[Path]:
    if dry_run:
        return [shard_dir / f"network_input_shard_{idx + 1:03d}.jsonl" for idx in range(num_shards)]

    rows = read_jsonl(input_path)
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError(f"No rows available for network conversion from {input_path}.")

    num_shards = min(num_shards, len(rows))
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_paths: List[Path] = []
    for shard_idx in range(num_shards):
        shard_rows = rows[shard_idx::num_shards]
        shard_path = shard_dir / f"network_input_shard_{shard_idx + 1:03d}.jsonl"
        write_jsonl(shard_path, shard_rows)
        shard_paths.append(shard_path)
    return shard_paths


def merge_jsonl_files(input_paths: List[Path], output_path: Path) -> None:
    rows: List[Dict[str, Any]] = []
    for input_path in input_paths:
        rows.extend(read_jsonl(input_path))
    write_jsonl(output_path, rows)


def merge_existing_network_files(
    input_paths: List[Path],
    output_path: Path,
    model_label: str,
) -> int:
    rows: List[Dict[str, Any]] = []
    for input_path in input_paths:
        generation_match = re.search(r"(\d+)(?=\.jsonl$)", input_path.name)
        generation_index = int(generation_match.group(1)) if generation_match else None
        for row in read_jsonl(input_path):
            if not isinstance(row.get("graph"), list):
                continue
            if not isinstance(row.get("long_profile_description"), str):
                continue
            row = dict(row)
            row.setdefault("model_label", model_label)
            row.setdefault("network_generation_index", generation_index)
            row.setdefault("network_file", rel(input_path))
            rows.append(row)
    write_jsonl(output_path, rows)
    return len(rows)


def run_network_eval(args: argparse.Namespace, combined_path: Path, run_dir: Path) -> Dict[str, Path]:
    network_path = run_dir / "reasonability" / "network_model.jsonl"
    network_err_path = run_dir / "reasonability" / "network_model_err.jsonl"
    scored_path = run_dir / "reasonability" / "network_model_scored.jsonl"
    scored_err_path = run_dir / "reasonability" / "network_model_scored_err.jsonl"

    if args.existing_network_globs:
        network_paths = expand_existing_network_globs(args.existing_network_globs)
        print("Using existing network-model files:")
        for path in network_paths:
            print(f"  - {rel(path)}")
        if not args.dry_run:
            num_rows = merge_existing_network_files(
                input_paths=network_paths,
                output_path=network_path,
                model_label=args.existing_network_label,
            )
            write_jsonl(network_err_path, [])
            print(f"Combined {num_rows} existing network rows into {rel(network_path)}")
    else:
        network_gpus = effective_network_gpus(args)
        max_parallel = args.max_parallel_network_jobs or default_parallel_jobs(network_gpus)
        num_shards = len(network_gpus) if network_gpus else 1

        tmp_dir = (
            run_dir / "reasonability" / "network_shards"
            if args.dry_run
            else Path(tempfile.mkdtemp(prefix="profile_network_"))
        )

        try:
            shard_input_paths = split_rows_for_network(
                input_path=combined_path,
                shard_dir=tmp_dir,
                num_shards=num_shards,
                limit=args.limit_network,
                dry_run=args.dry_run,
            )

            network_jobs: List[Dict[str, Any]] = []
            shard_output_paths: List[Path] = []
            shard_error_paths: List[Path] = []
            for shard_idx, shard_input_path in enumerate(shard_input_paths, start=1):
                shard_output_path = tmp_dir / f"network_output_shard_{shard_idx:03d}.jsonl"
                shard_error_path = tmp_dir / f"network_error_shard_{shard_idx:03d}.jsonl"
                shard_output_paths.append(shard_output_path)
                shard_error_paths.append(shard_error_path)
                command = [
                    sys.executable,
                    "-m", "model_training.observer.predict_network",
                    "--input",
                    str(shard_input_path),
                    "--output",
                    str(shard_output_path),
                    "--error-output",
                    str(shard_error_path),
                    "--input-field",
                    "long_profile_description",
                    "--model-path",
                    args.network_model_path,
                    "--stage1-max-new-tokens",
                    str(args.network_stage1_max_new_tokens),
                    "--stage2-max-new-tokens",
                    str(args.network_stage2_max_new_tokens),
                    "--stage2-retries",
                    str(args.network_stage2_retries),
                ]
                network_jobs.append(
                    {
                        "command": command,
                        "output_path": shard_output_path,
                        "name": f"network shard {shard_idx}",
                    }
                )

            run_parallel_jobs(
                jobs=network_jobs,
                gpus=network_gpus,
                max_parallel=max_parallel,
                dry_run=args.dry_run,
                job_kind="network",
                max_parallel_arg="--max-parallel-network-jobs",
            )

            if not args.dry_run:
                merge_jsonl_files(shard_output_paths, network_path)
                merge_jsonl_files(shard_error_paths, network_err_path)
        finally:
            if not args.dry_run:
                shutil.rmtree(tmp_dir, ignore_errors=True)

    command = [
        sys.executable,
        "-m", f"{EVAL_PKG}.score_network_edges",
        "--input",
        rel(network_path),
        "--output",
        rel(scored_path),
        "--error-output",
        rel(scored_err_path),
        "--input-field",
        "long_profile_description",
        "--graph-field",
        "graph",
        "--max-retries",
        str(args.edge_score_max_retries),
        "--sleep-seconds",
        str(args.edge_score_sleep_seconds),
    ]
    if args.limit_network is not None:
        command.extend(["--limit", str(args.limit_network)])
    run_command(command, dry_run=args.dry_run, gpu=args.score_gpu)

    return {
        "network": network_path,
        "network_errors": network_err_path,
        "scored": scored_path,
        "scored_errors": scored_err_path,
    }


def run_diversity(args: argparse.Namespace, merged_path: Path, run_dir: Path) -> Path:
    if args.dry_run:
        max_profiles = getattr(
            args,
            "expected_max_profiles_per_short_profile",
            len(args.stage1_models) * args.generations_per_model,
        )
        if args.max_diversity_profiles is not None:
            max_profiles = min(max_profiles, args.max_diversity_profiles)
        output_path = run_dir / "diversity" / "diversity_summary_by_k.jsonl"
        for num_profiles in range(2, max_profiles + 1):
            command = [
                sys.executable,
                "-m", f"{EVAL_PKG}.diversity_metrics",
                "--input",
                rel(merged_path),
                "--output",
                f"<temporary>/diversity_{num_profiles}.jsonl",
                "--summary-output",
                f"<temporary>/diversity_{num_profiles}_summary.json",
                "--bleu-order",
                str(args.bleu_order),
                "--embedding-backend",
                args.embedding_backend,
                "--embedding-model",
                args.embedding_model,
                "--num-profiles",
                str(num_profiles),
            ]
            run_command(command, dry_run=True)
        return output_path

    merged_rows = read_jsonl(merged_path)
    max_profiles = 0
    if merged_rows:
        max_profiles = max(len(row.get("generated_profiles", [])) for row in merged_rows)
    if max_profiles < 2:
        raise ValueError("Need at least two generated profiles per short profile for diversity metrics.")
    if args.max_diversity_profiles is not None:
        max_profiles = min(max_profiles, args.max_diversity_profiles)

    summary_rows: List[Dict[str, Any]] = []
    tmp_dir = Path(tempfile.mkdtemp(prefix="profile_diversity_"))
    try:
        for num_profiles in range(2, max_profiles + 1):
            detail_path = tmp_dir / f"diversity_{num_profiles}.jsonl"
            summary_path = tmp_dir / f"diversity_{num_profiles}_summary.json"
            command = [
                sys.executable,
                "-m", f"{EVAL_PKG}.diversity_metrics",
                "--input",
                rel(merged_path),
                "--output",
                str(detail_path),
                "--summary-output",
                str(summary_path),
                "--bleu-order",
                str(args.bleu_order),
                "--embedding-backend",
                args.embedding_backend,
                "--embedding-model",
                args.embedding_model,
                "--num-profiles",
                str(num_profiles),
            ]
            run_command(command, dry_run=args.dry_run)
            if not args.dry_run:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                summary["num_profiles_per_short_profile"] = num_profiles
                summary_rows.append(summary)
    finally:
        if args.keep_diversity_details:
            detail_dir = run_dir / "diversity" / "per_k_details"
            if tmp_dir.exists():
                detail_dir.parent.mkdir(parents=True, exist_ok=True)
                if detail_dir.exists():
                    shutil.rmtree(detail_dir)
                shutil.move(str(tmp_dir), str(detail_dir))
        else:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    output_path = run_dir / "diversity" / "diversity_summary_by_k.jsonl"
    if not args.dry_run:
        write_jsonl(output_path, summary_rows)
    return output_path


def summarize_reasonability(scored_path: Path, error_path: Path) -> Dict[str, Any]:
    rows = read_jsonl(scored_path)
    errors = read_jsonl(error_path)
    scored_values = [
        float(row["average_edge_score"])
        for row in rows
        if isinstance(row.get("average_edge_score"), (int, float))
    ]
    edge_counts = [len(row.get("edge_scores", [])) for row in rows]

    by_model: Dict[str, List[float]] = {}
    for row in rows:
        score = row.get("average_edge_score")
        if not isinstance(score, (int, float)):
            continue
        original = row.get("original_item") or {}
        model_label = original.get("model_label", "unknown")
        by_model.setdefault(str(model_label), []).append(float(score))

    return {
        "num_scored_profiles": len(rows),
        "num_score_errors": len(errors),
        "avg_reasonability_score": mean(scored_values) if scored_values else None,
        "avg_edges_per_profile": mean(edge_counts) if edge_counts else None,
        "by_model": {
            model: {
                "num_profiles": len(values),
                "avg_reasonability_score": mean(values),
            }
            for model, values in sorted(by_model.items())
        },
    }


def load_diversity_summary(path: Path) -> List[Dict[str, Any]]:
    return read_jsonl(path)


def make_report(
    args: argparse.Namespace,
    run_dir: Path,
    generation_paths: List[Path],
    combined_path: Path,
    merged_path: Path,
    reasonability_paths: Dict[str, Path],
    diversity_summary_path: Path,
) -> Dict[str, Any]:
    generation_counts = {
        rel(path): count_successful_generation_rows(path)
        for path in generation_paths
    }
    combined_rows = read_jsonl(combined_path)
    merged_rows = read_jsonl(merged_path)
    reasonability = summarize_reasonability(
        reasonability_paths["scored"],
        reasonability_paths["scored_errors"],
    )
    diversity_by_k = load_diversity_summary(diversity_summary_path)

    report = {
        "run_dir": rel(run_dir),
        "input": args.input,
        "existing_long_profile_globs": args.existing_long_profile_globs or [],
        "existing_network_globs": args.existing_network_globs or [],
        "existing_network_label": args.existing_network_label,
        "stage1_models": args.stage1_models,
        "generations_per_model": args.generations_per_model,
        "expected_max_profiles_per_short_profile": getattr(
            args,
            "expected_max_profiles_per_short_profile",
            len(args.stage1_models) * args.generations_per_model,
        ),
        "generation_gpus": parse_gpu_list(args.generation_gpus, "--generation-gpus"),
        "max_parallel_generations": args.max_parallel_generations
        or default_parallel_jobs(parse_gpu_list(args.generation_gpus, "--generation-gpus")),
        "network_gpus": effective_network_gpus(args),
        "max_parallel_network_jobs": args.max_parallel_network_jobs
        or default_parallel_jobs(effective_network_gpus(args)),
        "score_gpu": args.score_gpu,
        "num_flat_generated_profiles": len(combined_rows),
        "num_short_profiles": len(merged_rows),
        "generation_counts": generation_counts,
        "outputs": {
            "combined_long_profiles": rel(combined_path),
            "merged_generated_profiles": rel(merged_path),
            "network_model": rel(reasonability_paths["network"]),
            "network_errors": rel(reasonability_paths["network_errors"]),
            "scored_network": rel(reasonability_paths["scored"]),
            "score_errors": rel(reasonability_paths["scored_errors"]),
            "diversity_summary_by_k": rel(diversity_summary_path),
        },
        "reasonability": reasonability,
        "diversity_by_k": diversity_by_k,
    }
    write_json(run_dir / "report.json", report)
    write_markdown_report(run_dir / "report.md", report)
    return report


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown_report(path: Path, report: Dict[str, Any]) -> None:
    reasonability = report["reasonability"]
    diversity_by_k = report["diversity_by_k"]
    lines = [
        "# Profile Generation Evaluation Report",
        "",
        f"- Run directory: `{report['run_dir']}`",
        f"- Input: `{report['input']}`",
        f"- Existing long-profile globs: `{', '.join(report['existing_long_profile_globs']) or 'none'}`",
        f"- Existing network globs: `{', '.join(report['existing_network_globs']) or 'none'}`",
        f"- Existing network label: `{report['existing_network_label']}`",
        f"- Stage-1 models: `{', '.join(report['stage1_models']) or 'none'}`",
        f"- Generations per model: `{report['generations_per_model']}`",
        f"- Expected max profiles per short profile: `{report['expected_max_profiles_per_short_profile']}`",
        f"- Generation GPUs: `{', '.join(report['generation_gpus']) or 'default'}`",
        f"- Max parallel generation jobs: `{report['max_parallel_generations']}`",
        f"- Network GPUs: `{', '.join(report['network_gpus']) or 'default'}`",
        f"- Max parallel network jobs: `{report['max_parallel_network_jobs']}`",
        f"- Edge-score GPU: `{report['score_gpu'] or 'default'}`",
        f"- Short profiles: `{report['num_short_profiles']}`",
        f"- Generated long profiles: `{report['num_flat_generated_profiles']}`",
        "",
        "## Reasonability",
        "",
        f"- Scored profiles: `{reasonability['num_scored_profiles']}`",
        f"- Score errors: `{reasonability['num_score_errors']}`",
        f"- Average edge score: `{fmt(reasonability['avg_reasonability_score'])}`",
        f"- Average edges per profile: `{fmt(reasonability['avg_edges_per_profile'])}`",
        "",
        "### Reasonability by model",
        "",
        "| Model | Profiles | Avg edge score |",
        "| --- | ---: | ---: |",
    ]
    for model, stats in reasonability["by_model"].items():
        lines.append(
            f"| {model} | {stats['num_profiles']} | {fmt(stats['avg_reasonability_score'])} |"
        )

    lines.extend(
        [
            "",
            "## Diversity by generation count",
            "",
            "| Profiles per short profile | Avg self-BLEU | Avg mean cosine distance | Avg median cosine distance |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for row in diversity_by_k:
        lines.append(
            "| "
            f"{row.get('num_profiles_per_short_profile')} | "
            f"{fmt(row.get('avg_self_bleu'))} | "
            f"{fmt(row.get('avg_mean_pairwise_cosine_distance'))} | "
            f"{fmt(row.get('avg_median_pairwise_cosine_distance'))} |"
        )

    lines.extend(
        [
            "",
            "## Output files",
            "",
        ]
    )
    for name, output_path in report["outputs"].items():
        lines.append(f"- {name}: `{output_path}`")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run short-to-long generation, reasonability scoring, diversity scoring, and report creation."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Short patient profile JSONL (one row per patient, text in --input-field).",
    )
    parser.add_argument("--input-field", default="short_patient_profile")
    parser.add_argument(
        "--stage1-models",
        nargs="+",
        default=[],
        help="Stage-1 models for short-to-long generation. Use Azure deployment names, Claude names, Gemini names, or local model paths.",
    )
    parser.add_argument(
        "--existing-long-profile-globs",
        nargs="+",
        default=None,
        help="Skip generation and use existing long-profile JSONL files. "
        "Example: outputs/long_profile/claude/*.jsonl outputs/long_profile/gpt5/*.jsonl",
    )
    parser.add_argument("--stage2-deployment", default=get_env("ANGEL_GPT5_DEPLOYMENT", "gpt-5", "AZURE_OPENAI_DEPLOYMENT"))
    parser.add_argument("--generations-per-model", type=int, default=1)
    parser.add_argument(
        "--generation-gpus",
        default=None,
        help="Comma-separated GPU ids for parallel generation jobs, e.g. '0,1,2,3'. "
        "Each launched job receives CUDA_VISIBLE_DEVICES=<gpu>.",
    )
    parser.add_argument(
        "--max-parallel-generations",
        type=int,
        default=None,
        help="Maximum generation jobs to run at once. Defaults to number of --generation-gpus, or 1.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--output-root",
        default=str(OUTPUTS_DIR / "observer" / "auto_eval_runs"),
        help="Directory for useful run outputs.",
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume", action="store_true", help="Reuse complete generation files if present.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")

    parser.add_argument("--network-model-path", default=None, help="Default: resolve_model('observer').")
    parser.add_argument(
        "--network-gpu",
        default=None,
        help="Backward-compatible single GPU for network conversion. Prefer --network-gpus for multi-GPU runs.",
    )
    parser.add_argument(
        "--network-gpus",
        default=None,
        help="Comma-separated GPU ids for parallel predict_network shards, e.g. '0,1,2,3'.",
    )
    parser.add_argument(
        "--existing-network-globs",
        nargs="+",
        default=None,
        help="Skip predict_network and use existing network-model JSONL files. "
        "Files with '_err' or '_scored' in the name are ignored.",
    )
    parser.add_argument(
        "--existing-network-label",
        default="existing_network",
        help="Model label to attach to rows loaded through --existing-network-globs for the final report.",
    )
    parser.add_argument(
        "--max-parallel-network-jobs",
        type=int,
        default=None,
        help="Maximum network-conversion jobs to run at once. Defaults to number of --network-gpus, or 1.",
    )
    parser.add_argument(
        "--score-gpu",
        default=None,
        help="Single GPU id/group for score_network_edges.py, e.g. '0'. Sets CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument("--network-stage1-max-new-tokens", type=int, default=2048)
    parser.add_argument("--network-stage2-max-new-tokens", type=int, default=2048)
    parser.add_argument("--network-stage2-retries", type=int, default=10)
    parser.add_argument("--limit-network", type=int, default=None)
    parser.add_argument("--edge-score-max-retries", type=int, default=5)
    parser.add_argument("--edge-score-sleep-seconds", type=float, default=2.0)

    parser.add_argument("--bleu-order", type=int, default=4, choices=[1, 2, 3, 4])
    parser.add_argument(
        "--embedding-backend",
        default="sentence-transformers",
        choices=["auto", "sentence-transformers", "tfidf"],
        help="Default tfidf avoids downloading/loading extra embedding models.",
    )
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument(
        "--max-diversity-profiles",
        type=int,
        default=None,
        help="Highest number of generated profiles per short profile to include in diversity-by-k.",
    )
    parser.add_argument(
        "--keep-diversity-details",
        action="store_true",
        help="Keep per-short-profile diversity JSONL files for each generation count.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.generations_per_model < 1:
        raise ValueError("--generations-per-model must be at least 1.")
    if args.max_parallel_generations is not None and args.max_parallel_generations < 1:
        raise ValueError("--max-parallel-generations must be at least 1.")
    if args.max_parallel_network_jobs is not None and args.max_parallel_network_jobs < 1:
        raise ValueError("--max-parallel-network-jobs must be at least 1.")
    if args.max_diversity_profiles is not None and args.max_diversity_profiles < 2:
        raise ValueError("--max-diversity-profiles must be at least 2.")
    if not args.stage1_models and not args.existing_long_profile_globs:
        raise ValueError("Provide --stage1-models to generate profiles or --existing-long-profile-globs to reuse existing profiles.")
    if args.stage1_models and args.existing_long_profile_globs:
        raise ValueError("Use either --stage1-models or --existing-long-profile-globs in one run, not both.")
    if args.existing_network_globs and not args.existing_long_profile_globs:
        raise ValueError("--existing-network-globs still needs --existing-long-profile-globs for diversity metrics.")
    args.input = normalize_repo_arg(args.input)
    args.output_root = normalize_repo_arg(args.output_root)
    args.network_model_path = normalize_repo_arg(resolve_model("observer", args.network_model_path))
    args.stage1_models = [normalize_repo_arg(model) for model in args.stage1_models]
    if args.existing_long_profile_globs:
        args.existing_long_profile_globs = [
            normalize_repo_arg(pattern) for pattern in args.existing_long_profile_globs
        ]
    if args.existing_network_globs:
        args.existing_network_globs = [
            normalize_repo_arg(pattern) for pattern in args.existing_network_globs
        ]
    if args.score_gpu is None:
        args.score_gpu = args.network_gpu

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = REPO_ROOT / args.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.existing_long_profile_globs:
        generation_paths = expand_existing_long_profile_globs(args.existing_long_profile_globs)
        args.expected_max_profiles_per_short_profile = len(generation_paths)
        print("Using existing long-profile files:")
        for path in generation_paths:
            print(f"  - {rel(path)}")
    else:
        generation_paths = generate_profiles(args, run_dir)
        args.expected_max_profiles_per_short_profile = len(args.stage1_models) * args.generations_per_model
    combined_path = run_dir / "combined_long_profiles.jsonl"
    merged_path = run_dir / "merged_generated_profiles.jsonl"

    if not args.dry_run:
        combined_count = combine_generation_outputs(generation_paths, combined_path)
        print(f"Combined {combined_count} generated profiles into {rel(combined_path)}")

    run_merge(combined_path, merged_path, dry_run=args.dry_run)
    reasonability_paths = run_network_eval(args, combined_path, run_dir)
    diversity_summary_path = run_diversity(args, merged_path, run_dir)

    if not args.dry_run:
        report = make_report(
            args=args,
            run_dir=run_dir,
            generation_paths=generation_paths,
            combined_path=combined_path,
            merged_path=merged_path,
            reasonability_paths=reasonability_paths,
            diversity_summary_path=diversity_summary_path,
        )
        print(f"\nReport written to {report['run_dir']}/report.md")


if __name__ == "__main__":
    main()
