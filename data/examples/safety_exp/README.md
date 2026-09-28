# Synthetic safety_exp example

Invented for this repository so the pipeline can run end to end. It is not
from the paper's data and describes no real person.

- `example_source_transcript.txt` — 3-turn source transcript in the
  `You said:` / `ChatGPT said:` format that `generate_redteam_transcript`
  and `query_models` parse. The assistant side escalates mildly, as the real
  source does over 58 turns.
- `example_partial_context.txt` — first turn only; a stand-in for PARTIAL context.
- `example_profile_ids.txt` — profile-id list for `experiments/safety_exp/scripts/run_redteam.sh`.
