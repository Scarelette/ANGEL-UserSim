"""Fixed-agenda profile-expansion experiment (main paper experiment)."""

from angel_common.env import load_env

# Make credentials in <repo>/.env visible to SDKs that read os.environ directly
# (anthropic.Anthropic(), transformers' HF_TOKEN, ...).
load_env()
