"""The five patient simulators compared in the experiment (``--model``).

``build_evaluation_patient_model`` builds (or reuses) the simulator for one
profile; the implementations live in ``patients/`` and ``angel_initializer.py``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from angel_common.paths import resolve_model


EVALUATION_MODEL_CHOICES = ("patient_psi", "roleplay_doh", "eeyore", "one_stage", "angel")
# None = resolve through angel_common.paths.resolve_model:
#   eeyore    -> EEYORE_MODEL        (default liusiyang/eeyore_sft_epoch2_dpo_round2_epoch1_llama3.1_8B)
#   one_stage -> ANGEL_ACTOR_MODEL   (the stage-2 Actor, prompted with the short profile directly)
#   angel     -> ANGEL_ACTOR_MODEL + ANGEL_OBSERVER_MODEL
DEFAULT_EEYORE_MODEL_NAME: Optional[str] = None
DEFAULT_ONE_STAGE_MODEL_NAME: Optional[str] = None
DEFAULT_ANGEL_MODEL_NAME: Optional[str] = None
DEFAULT_ANGEL_STAGE1_MODEL_NAME: Optional[str] = None


def _extract_model_payload(
    profile_item: Dict[str, Any],
    model_name: str,
    *,
    required: bool = True,
) -> Optional[Dict[str, Any]]:
    if not isinstance(profile_item, dict):
        raise TypeError(f"profile_item must be a dict, got {type(profile_item)}")

    if model_name == "angel":
        payload = profile_item.get("angel_processed_result")
    else:
        payload = profile_item.get("patient_processed_result")

    if not isinstance(payload, dict):
        if not required:
            return None
        available = sorted(profile_item.keys())
        raise ValueError(
            f"Missing payload for model={model_name}. "
            f"Expected {'angel_processed_result' if model_name == 'angel' else 'patient_processed_result'} dict. "
            f"Available keys: {available}"
        )
    return payload


def _extract_complaint_text(profile_item: Dict[str, Any], payload: Optional[Dict[str, Any]]) -> str:
    if isinstance(payload, dict):
        complaint = payload.get("complaints")
        if isinstance(complaint, str) and complaint.strip():
            return complaint.strip()

    short_profile = profile_item.get("short_patient_profile")
    if isinstance(short_profile, str) and short_profile.strip():
        return short_profile.strip()

    complaints_alt = profile_item.get("Complaints")
    if isinstance(complaints_alt, str) and complaints_alt.strip():
        return complaints_alt.strip()

    raise ValueError("Missing complaint text (no complaints, short_patient_profile, or Complaints).")


def _build_minimal_patient_psi_profile(profile_item: Dict[str, Any]) -> Dict[str, Any]:
    short_profile = _extract_complaint_text(profile_item, payload={})
    raw_id = profile_item.get("id")
    name = profile_item.get("name") or ""
    diagnosis_hint = profile_item.get("diagnosis_hint")
    if isinstance(diagnosis_hint, list):
        type_values = [str(item) for item in diagnosis_hint if item is not None]
    else:
        type_values = []

    return {
        "name": str(name),
        "id": f"short_profile_{raw_id}" if raw_id is not None else "short_profile",
        "type": type_values,
        "history": short_profile,
        "helpless_belief": [],
        "unlovable_belief": [],
        "worthless_belief": [],
        "intermediate_belief": "",
        "coping_strategies": "",
        "situation": "",
        "auto_thought": "",
        "emotion": [],
        "behavior": "",
    }


def _build_eval_profile_prompt(profile_text: str) -> str:
    return (
        "Imagine you are a patient who has been experiencing mental health challenges.\n\n"
        "Below is your clinical background and case profile:\n"
        f"{profile_text}"
    )


def build_evaluation_patient_model(
    *,
    model_name: str,
    profile_item: Dict[str, Any],
    current_model: Optional[Any] = None,
    eeyore_model_name: Optional[str] = DEFAULT_EEYORE_MODEL_NAME,
    one_stage_model_name: Optional[str] = DEFAULT_ONE_STAGE_MODEL_NAME,
    angel_model_name: Optional[str] = DEFAULT_ANGEL_MODEL_NAME,
    angel_stage1_model_name: Optional[str] = DEFAULT_ANGEL_STAGE1_MODEL_NAME,
    device_map: str = "auto",
    verbose: bool = False,
) -> Any:
    if model_name not in EVALUATION_MODEL_CHOICES:
        raise ValueError(f"Unsupported model_name: {model_name}. Choices: {EVALUATION_MODEL_CHOICES}")

    if model_name == "patient_psi":
        payload = _extract_model_payload(profile_item, model_name, required=False)
        from experiments.profile_expansion.patients.patient_psi import Patient_Psi

        psi_profile = payload.get("patient_psi_profile") if isinstance(payload, dict) else None
        if not isinstance(psi_profile, dict):
            psi_profile = _build_minimal_patient_psi_profile(profile_item)
            if verbose:
                print("[ModelInit] patient_psi_profile missing; using short-profile fallback.", flush=True)

        if isinstance(current_model, Patient_Psi):
            current_model.system_prompt = current_model._build_prompt(psi_profile)
            return current_model
        return Patient_Psi(psi_profile)

    if model_name == "roleplay_doh":
        payload = _extract_model_payload(profile_item, model_name, required=False)
        from experiments.profile_expansion.patients.roleplay_doh import Patient_RoleplayDOH

        complaint_text = _extract_complaint_text(profile_item, payload)
        if isinstance(current_model, Patient_RoleplayDOH):
            current_model.profile = complaint_text
            current_model.system_prompt = _build_eval_profile_prompt(complaint_text)
            return current_model
        return Patient_RoleplayDOH(complaint_text)

    if model_name == "eeyore":
        payload = _extract_model_payload(profile_item, model_name, required=False)
        from experiments.profile_expansion.patients.eeyore import Eeyore

        complaint_text = _extract_complaint_text(profile_item, payload)
        if isinstance(current_model, Eeyore):
            current_model.system_prompt = _build_eval_profile_prompt(complaint_text)
            return current_model
        return Eeyore(
            complaint_text,
            model_name=resolve_model("eeyore", eeyore_model_name),
            device_map=device_map,
        )

    if model_name == "one_stage":
        payload = _extract_model_payload(profile_item, model_name, required=False)
        from experiments.profile_expansion.patients.angel import Angel as OneStagePatientModel

        complaint_text = _extract_complaint_text(profile_item, payload)
        if isinstance(current_model, OneStagePatientModel):
            current_model.set_profile(profile=complaint_text)
            return current_model
        return OneStagePatientModel(
            profile=complaint_text,
            model_name=resolve_model("actor", one_stage_model_name),
            device_map=device_map,
        )

    from experiments.profile_expansion.angel_initializer import TwoStageAngelPatient

    if isinstance(current_model, TwoStageAngelPatient):
        current_model.initialize_from_profile_item(profile_item)
        return current_model

    model = TwoStageAngelPatient(
        stage1_model_name=angel_stage1_model_name,
        stage2_model_name=angel_model_name,
        device_map=device_map,
        verbose=verbose,
    )
    model.initialize_from_profile_item(profile_item)
    return model
