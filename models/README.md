Place model weights here (this directory is gitignored except for this file):

    models/Angel-Observer/   stage-1 Observer  (Hugging Face: ChengLi0228/Angel-Observer)
    models/Angel-Actor/      stage-2 Actor     (Hugging Face: ChengLi0228/Angel-Actor)

You don't have to: if a folder is missing, the model is downloaded from its
Hugging Face id on first use. To download into this folder instead:

    hf download ChengLi0228/Angel-Observer --local-dir models/Angel-Observer
    hf download ChengLi0228/Angel-Actor    --local-dir models/Angel-Actor

Or point to them with ANGEL_OBSERVER_MODEL / ANGEL_ACTOR_MODEL (a path or a
Hugging Face id).

Earlier versions called these folders Qwen3-Observer-800 and
qwen3-8b-dpo-merged. Rename an old copy, or point the variables at it.
