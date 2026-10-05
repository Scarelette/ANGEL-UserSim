"""Fine-tune the rollout therapist on Azure OpenAI.

Checks ``therapist_training.jsonl``, uploads it and starts a fine-tuning job.
The paper fine-tuned GPT-4o with the service's default hyperparameters.

    python -m model_training.therapist.fine_tune --dry-run          # check the file only (free)
    python -m model_training.therapist.fine_tune --model gpt-4o-2024-08-06

When the job succeeds, deploy the fine-tuned model in the Azure AI Foundry
portal (Fine-tuning -> your job -> Deploy) and set ``ANGEL_THERAPIST_DEPLOYMENT``
to that deployment name. Fine-tuning and hosting the deployment are billed by
Azure. Check which base models your region can fine-tune before running.

Uses ``AZURE_OPENAI_ENDPOINT`` / ``AZURE_OPENAI_API_KEY``, or the
``ANGEL_THERAPIST_AZURE_*`` overrides if the therapist lives on another resource.
"""

from __future__ import annotations

import argparse
import json
import time

from angel_common.env import get_env
from angel_common.llm import azure_openai_client
from angel_common.paths import DATA_DIR, resolve_path

MIN_ROWS = 10  # Azure OpenAI's minimum for a fine-tuning file


def check_file(path) -> int:
    """Number of rows; raises ValueError on the first malformed one."""
    n = 0
    with open(path, encoding="utf-8-sig") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                msgs = json.loads(line)["messages"]
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                raise ValueError(f"line {i}: not a {{'messages': [...]}} row ({e})")
            roles = [m.get("role") for m in msgs]
            if roles[-1:] != ["assistant"] or any(r not in ("system", "user", "assistant") for r in roles):
                raise ValueError(f"line {i}: roles {roles}; need system/user/assistant, ending with assistant")
            if any(not (m.get("content") or "").strip() for m in msgs):
                raise ValueError(f"line {i}: empty message content")
            n += 1
    if n < MIN_ROWS:
        raise ValueError(f"{n} rows; Azure OpenAI needs at least {MIN_ROWS}")
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-file", default=str(DATA_DIR / "therapist" / "therapist_training.jsonl"))
    ap.add_argument("--model", default="gpt-4o-2024-08-06", help="base model to fine-tune")
    ap.add_argument("--suffix", default="ai-therapist", help="added to the fine-tuned model's name")
    ap.add_argument("--dry-run", action="store_true", help="check the file and stop")
    ap.add_argument("--wait", action="store_true", help="poll until the job finishes")
    args = ap.parse_args()

    path = resolve_path(args.train_file)
    n = check_file(path)
    print(f"{path}: {n} rows OK")
    if args.dry_run:
        return

    client = azure_openai_client(endpoint=get_env("ANGEL_THERAPIST_AZURE_ENDPOINT"),
                                 api_key=get_env("ANGEL_THERAPIST_AZURE_API_KEY"))
    with open(path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="fine-tune")
    print("uploaded file:", uploaded.id)
    job = client.fine_tuning.jobs.create(training_file=uploaded.id, model=args.model, suffix=args.suffix)
    print("fine-tuning job:", job.id, job.status)

    while args.wait and job.status not in ("succeeded", "failed", "cancelled"):
        time.sleep(60)
        job = client.fine_tuning.jobs.retrieve(job.id)
        print(time.strftime("%H:%M"), job.status)
    if job.status == "succeeded":
        print("fine-tuned model:", job.fine_tuned_model, "-> deploy it, then set ANGEL_THERAPIST_DEPLOYMENT")
    elif job.status in ("failed", "cancelled"):
        raise SystemExit(f"job {job.status}: {getattr(job, 'error', None)}")


if __name__ == "__main__":
    main()
