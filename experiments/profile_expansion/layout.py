"""Default input/output locations for the profile-expansion experiment.

Inputs live under ``$ANGEL_DATA_DIR/profile_expansion`` (default ``data/``) and
everything generated goes under ``$ANGEL_OUTPUT_DIR/profile_expansion``
(default ``outputs/``, gitignored). Every script also accepts explicit paths.
"""

from angel_common.paths import DATA_DIR, OUTPUTS_DIR

INPUT_DIR = DATA_DIR / "profile_expansion"
SHORT_PROFILES = INPUT_DIR / "selected_50_short_patient_profiles_v2.jsonl"

OUT_DIR = OUTPUTS_DIR / "profile_expansion"
RESULTS_DIR = OUT_DIR / "results"            # agenda transcripts + metric JSONs
CLEAN_DIR = RESULTS_DIR / "clean"            # combined metric JSONs (main results)
