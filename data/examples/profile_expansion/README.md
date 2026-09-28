# Example inputs — profile expansion

`short_profiles.example.jsonl` contains **two synthetic profiles** (written for
this repository, not taken from any case report) with exactly the schema of the
paper's input file `selected_50_short_patient_profiles_v2.jsonl`:

| key | type | used by |
|---|---|---|
| `id` | int | record id (`profile_id` in outputs) |
| `source_title` | str | carried through to outputs |
| `name`, `age`, `gender`, `diagnosis_hint` | str / int / str / list | Angel stage-1 fallback profile |
| `short_patient_profile` | str | **the input every patient model receives** |
| `patient_psi_profile` | dict | CCD-style profile (see Known issues in the experiment README) |

The paper's 50 profiles were derived from published clinical case reports and
are not redistributed here. To run the full experiment, place your own file at
`data/profile_expansion/selected_50_short_patient_profiles_v2.jsonl` or pass `--input`.
