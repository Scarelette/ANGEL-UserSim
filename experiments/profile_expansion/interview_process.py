"""Fixed-topic interview controller for profile-expansion experiments."""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from angel_common.llm import getOutput
from experiments.profile_expansion.azure_clients import deployment, role_azure_client


THERAPIST_SYSTEM_PROMPT = """You are a licensed mental health therapist 
(e.g., clinical psychologist or licensed counselor) trained in evidence-based practice.
You must always speak as the therapist only.

Guidelines:
- Maintain professional, ethical, empathetic tone
- Use CBT-style Socratic questioning
- Do NOT diagnose
- Do NOT role-play the patient
- Keep responses under 200 words
- End with a reflective or exploratory question
"""


# Interview agenda derived from simulation_annotator/Interview Guidance.md.
# Nine main sessions and their sub-sessions are flattened into ordered topics.
# Each topic carries the session it belongs to, the guidance "Goal", an opening
# question, and the remaining "Core interview questions" as follow-ups.
AGENDA: List[Dict[str, Any]] = [
    {
        "key": "presenting_problem",
        "name": "Presenting problem / chief complaint",
        "session": "1. Presenting Problem / Chief Complaint",
        "goal": (
            "Understand why the patient is seeking help now, how they describe the "
            "main concern in their own words, and what they hope to receive from the "
            "interview or treatment."
        ),
        "opening": "What brings you in today?",
        "followups": [
            "What are you hoping we can help you with?",
            "How would you describe the problem?",
            "What feels most important for me to understand about what you are going through?",
        ],
    },
    {
        "key": "symptoms_emotions",
        "name": "Major symptoms: emotions",
        "session": "2. Major Symptoms",
        "goal": "Understand the patient's emotional experience.",
        "opening": "What emotions have you been feeling most often?",
        "followups": [
            "What feels most distressing emotionally?",
            "How intense do these feelings get?",
            "How often do these emotions come up?",
        ],
    },
    {
        "key": "symptoms_behaviors",
        "name": "Major symptoms: behaviors",
        "session": "2. Major Symptoms",
        "goal": "Understand how the problem shows up in the patient's behavior.",
        "opening": "What do you do when the problem gets worse?",
        "followups": [
            "Have you changed how you act or what you do day to day?",
            "Are there things you avoid because of the problem?",
            "Have other people noticed changes in your behavior?",
        ],
    },
    {
        "key": "symptoms_cognitions",
        "name": "Major symptoms: cognitions",
        "session": "2. Major Symptoms",
        "goal": (
            "Understand the patient's thoughts, beliefs, worries, interpretations, "
            "and cognitive patterns."
        ),
        "opening": "What thoughts usually come up when you feel distressed?",
        "followups": [
            "What do you worry about most?",
            "What do you tell yourself about the situation?",
            "How do you interpret what is happening to you?",
        ],
    },
    {
        "key": "onset_timeline",
        "name": "History of problem: onset / timeline",
        "session": "3. History of Problem",
        "goal": "Establish the chronology and course of the presenting problem.",
        "opening": "How long has this been a concern?",
        "followups": [
            "When did this first begin?",
            "Is this the first time you have had these problems?",
            "If not, how many times has this happened before, and when was the first time?",
            "Has the problem changed over time?",
        ],
    },
    {
        "key": "triggers",
        "name": "History of problem: triggers / precipitating events",
        "session": "3. History of Problem",
        "goal": (
            "Identify events, stressors, or situations that started, reactivated, or "
            "worsened the problem."
        ),
        "opening": "According to you, what caused the problem?",
        "followups": [
            "Were there any new events happening in your life, or anything stressful?",
            "What events or situations make the problem worse?",
            "Are there situations where the problem feels better?",
            "Can you tell me more about how these events are linked to your problem?",
        ],
    },
    {
        "key": "impact_functioning",
        "name": "Impact on functioning",
        "session": "4. Impact on Functioning",
        "goal": (
            "Assess how the problem affects the patient's daily life, including work "
            "or school, relationships, health, self-care, and everyday tasks."
        ),
        "opening": "How has this problem affected your life?",
        "followups": [
            "Can you do your usual work or everyday tasks?",
            "Have you noticed any changes in your relationships?",
            "How has it affected your work, school, health, or daily responsibilities?",
            "What does a typical day look like for you now?",
        ],
    },
    {
        "key": "current_coping",
        "name": "Current coping",
        "session": "5. Current Coping",
        "goal": (
            "Understand how the patient currently deals with the problem, what helps, "
            "what makes it worse, and whether avoidance or other maladaptive coping "
            "patterns are present."
        ),
        "opening": "How have you been coping or dealing with this?",
        "followups": [
            "What helps, and what makes it worse?",
            "Are there times when the problem feels better or worse?",
            "What do you usually do when the distress becomes strong?",
        ],
    },
    {
        "key": "social_support",
        "name": "Social support",
        "session": "6. Social Support",
        "goal": (
            "Assess who is important in the patient's life, the quality of available "
            "support, whether the patient feels isolated, and what others have suggested."
        ),
        "opening": "Who are the important people in your life that you see or talk to regularly?",
        "followups": [
            "Do you have close friends or family?",
            "Would you like to have more friends or support?",
            "What do your family, friends, or other people suggest you do?",
            "Who do you usually turn to when things are difficult?",
        ],
    },
    {
        "key": "past_treatment",
        "name": "Past treatment / psychiatric history",
        "session": "7. Past Treatment / Psychiatric History",
        "goal": (
            "Collect prior treatment experiences, psychiatric diagnoses, "
            "hospitalizations, medication history, previous episodes, family mental "
            "health history, and barriers to seeking help."
        ),
        "opening": "Have you ever received help for this or similar problems?",
        "followups": [
            "What kind of help did you receive, when, for how long, and what was useful or not?",
            "Have you ever received a formal mental health diagnosis?",
            "Have you ever been hospitalized for psychological problems or stressors?",
            "Have you had prior psychological or psychiatric treatment?",
            "Has anything gotten in the way of getting help?",
            "Are you aware of similar concerns among family members?",
        ],
    },
    {
        "key": "risk_suicide_self_harm",
        "name": "Risk assessment: suicide / self-harm",
        "session": "8. Risk Assessment",
        "goal": (
            "Assess safety concerns related to suicide or self-harm, including "
            "frequency, plans, intent, access to means, risk factors, and protective "
            "factors."
        ),
        "opening": "Have you had thoughts that life was not worth living?",
        "followups": [
            "Have you thought you would be better off dead?",
            "Have you attempted to hurt yourself or end your life?",
            "How often do these thoughts occur, and when was the last time?",
            "Have you made plans to end your life, or thought about how you might go about it?",
            "Have you come close to acting on these thoughts?",
            "Do you have access to medications, guns, or other means?",
            "What keeps you going, and who do you go to when you feel this way?",
        ],
    },
    {
        "key": "risk_harm_others",
        "name": "Risk assessment: harm to others / victimization",
        "session": "8. Risk Assessment",
        "goal": "Assess risk of harm to others and any recent abuse or victimization.",
        "opening": "What does it look like when you get really angry or upset?",
        "followups": [
            "Have you ever had thoughts, made statements, or attempted to hurt others?",
            "Have you recently been physically hurt or threatened by someone else?",
        ],
    },
    {
        "key": "risk_substance_use",
        "name": "Risk assessment: substance use",
        "session": "8. Risk Assessment",
        "goal": "Assess substance use patterns and their impact on the patient's life.",
        "opening": "Do you use alcohol, drugs, tobacco, or prescription medications?",
        "followups": [
            "How much do you use, and how often?",
            "How long has this been going on?",
            "Has substance use affected your work, school, relationships, or health?",
            "Has anyone expressed concern about your use, or have there been any legal problems?",
        ],
    },
    {
        "key": "treatment_goal",
        "name": "Treatment goal / hope",
        "session": "9. Treatment Goal / Hope",
        "goal": (
            "Understand the patient's goals for treatment, expectations for therapy, "
            "desired changes, strengths, sources of resilience, and any remaining concerns."
        ),
        "opening": "What are your goals for therapy?",
        "followups": [
            "What do you picture therapy to look like?",
            "What are you hoping will change?",
            "What do you think are your greatest strengths?",
            "Is there anything else we may have missed or have not covered today?",
        ],
    },
]


@dataclass(frozen=True)
class TopicDecision:
    enough_information: bool
    reason: str
    followup_needed: bool
    forced_stop: bool = False


@dataclass(frozen=True)
class TherapistTransitionPlan:
    decision: str
    reason: str
    followup_question: str = ""


class TopicTransitionJudge:
    def evaluate(
        self,
        *,
        patient_answer: str,
        topic: Dict[str, Any],
        transcript: List[Dict[str, Any]],
    ) -> TopicDecision:
        raise NotImplementedError

    @staticmethod
    def force_stop(decision: TopicDecision) -> TopicDecision:
        if decision.enough_information:
            return decision
        return TopicDecision(
            enough_information=False,
            reason=f"{decision.reason}:max_topic_turns_reached",
            followup_needed=False,
            forced_stop=True,
        )


class GPT5TopicTransitionJudge(TopicTransitionJudge):
    """Prompted GPT-5 judge for topic transition state.

    Uses ``simulate_patient/output_generator.py`` so the transition decision can
    follow the same model utility already used elsewhere in this repo.
    """

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries

    def evaluate(
        self,
        *,
        patient_answer: str,
        topic: Dict[str, Any],
        transcript: List[Dict[str, Any]],
    ) -> TopicDecision:
        prompt = self._build_prompt(patient_answer=patient_answer, topic=topic, transcript=transcript)

        last_error = None
        for _ in range(self.max_retries):
            try:
                raw = getOutput(prompt, context=None, tag=0, max_completion_tokens=220, max_retries=2)
                parsed = _extract_json(raw)
                decision = str(parsed.get("decision", "")).strip().lower()
                reason = str(parsed.get("reason", "")).strip() or "gpt5_transition_judge"
                if decision == "move_next":
                    return TopicDecision(True, reason, False)
                if decision == "follow_up":
                    return TopicDecision(False, reason, True)
                last_error = f"unknown decision={decision!r}; raw={raw}"
            except Exception as exc:
                last_error = repr(exc)

        return TopicDecision(False, f"gpt5_judge_failed:{last_error}", True)

    def _build_prompt(
        self,
        *,
        patient_answer: str,
        topic: Dict[str, Any],
        transcript: List[Dict[str, Any]],
    ) -> str:
        current_topic_turns = [
            {
                "role": turn.get("role"),
                "content": turn.get("content"),
                "turn_in_topic": turn.get("turn_in_topic"),
            }
            for turn in transcript
            if turn.get("topic_key") == topic["key"]
        ]
        return f"""
You are controlling topic transitions in a psychotherapy intake simulation.

Current session:
{topic.get("session", topic["name"])}

Current topic:
{topic["name"]}

Goal of this topic:
{topic.get("goal", "")}

Task:
Decide whether the patient's latest answer contains enough information to meet the goal of this topic.

Move to the next topic ONLY if the answer is specific enough and not mainly vague.
Ask a follow-up within the same topic if the answer is vague, incomplete, contradictory/ambivalent, or mentions clinically important material that needs more clarification.

Current-topic dialogue:
{json.dumps(current_topic_turns, ensure_ascii=False, indent=2)}

Latest patient answer:
{patient_answer}

Return valid JSON only:
{{
  "decision": "move_next" or "follow_up",
  "reason": "brief reason"
}}
""".strip()


class PromptedAgendaTherapist:
    """Agenda-aware therapist using the same Azure settings as AITherapist."""

    def __init__(self, question_max_tokens: int = 180, transition_plan_max_tokens: int = 220):
        self.client = role_azure_client("therapist", async_client=False)
        self.question_max_tokens = question_max_tokens
        self.transition_plan_max_tokens = transition_plan_max_tokens

    def generate(
        self,
        *,
        transcript: List[Dict[str, Any]],
        topic: Dict[str, Any],
        turn_in_topic: int,
        previous_decision: Optional[TopicDecision] = None,
    ) -> str:
        messages = self._build_agenda_messages(
            transcript=transcript,
            topic=topic,
            turn_in_topic=turn_in_topic,
            previous_decision=previous_decision,
        )
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=deployment("therapist"),
                    messages=messages,
                    max_completion_tokens=self.question_max_tokens,
                    temperature=1.0,
                    top_p=1.0,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                )
                return _clean_therapist_question(response.choices[0].message.content or "")
            except Exception as exc:
                if attempt == 2:
                    raise RuntimeError(f"Agenda therapist failed: {exc}") from exc
        raise RuntimeError("Agenda therapist failed")

    def plan_next(
        self,
        *,
        transcript: List[Dict[str, Any]],
        topic: Dict[str, Any],
        topic_index: int,
        agenda: List[Dict[str, Any]],
        turn_in_topic: int,
    ) -> TherapistTransitionPlan:
        messages = self._build_transition_plan_messages(
            transcript=transcript,
            topic=topic,
            topic_index=topic_index,
            agenda=agenda,
            turn_in_topic=turn_in_topic,
        )
        last_error: Optional[str] = None
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=deployment("therapist"),
                    messages=messages,
                    max_completion_tokens=self.transition_plan_max_tokens,
                    temperature=0.7,
                    top_p=1.0,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                )
                raw = response.choices[0].message.content or ""
                parsed = _extract_json(raw)
                decision = str(parsed.get("decision", "")).strip().lower()
                reason = str(parsed.get("reason", "")).strip() or "therapist_plan"

                if decision == "move_next":
                    return TherapistTransitionPlan(decision="move_next", reason=reason)

                if decision == "follow_up":
                    followup_question = _clean_therapist_question(str(parsed.get("followup_question", "")).strip())
                    if not followup_question:
                        followup_question = self._fallback_followup_question(topic=topic, turn_in_topic=turn_in_topic)
                    return TherapistTransitionPlan(
                        decision="follow_up",
                        reason=reason,
                        followup_question=followup_question,
                    )

                last_error = f"unknown_decision:{decision!r}"
            except Exception as exc:
                last_error = repr(exc)
            if attempt < 2:
                continue

        return TherapistTransitionPlan(
            decision="follow_up",
            reason=f"therapist_plan_failed:{last_error}",
            followup_question=self._fallback_followup_question(topic=topic, turn_in_topic=turn_in_topic),
        )

    def _build_transition_plan_messages(
        self,
        *,
        transcript: List[Dict[str, Any]],
        topic: Dict[str, Any],
        topic_index: int,
        agenda: List[Dict[str, Any]],
        turn_in_topic: int,
    ) -> List[Dict[str, str]]:
        current_topic_turns = [
            {
                "role": turn.get("role"),
                "content": turn.get("content"),
                "turn_in_topic": turn.get("turn_in_topic"),
            }
            for turn in transcript
            if turn.get("topic_key") == topic["key"]
        ]
        agenda_lines = [f"{idx + 1}. {item['name']} ({item['key']})" for idx, item in enumerate(agenda)]
        instruction = f"""
You are running a fixed-agenda psychotherapy intake.

Agenda order (must all be covered): 
{chr(10).join(agenda_lines)}

Current agenda topic index: {topic_index + 1}/{len(agenda)}
Current session: {topic.get('session', topic['name'])}
Current topic: {topic['name']} ({topic['key']})
Goal of this topic: {topic.get('goal', '')}
Current therapist turn inside this topic just completed: {turn_in_topic}

Decide whether to move to the next topic now or ask one same-topic follow-up.

Rules:
- Return "move_next" only when the goal of this topic has been met with enough concrete information.
- Return "follow_up" when the latest answer is vague, incomplete, contradictory, or needs clarification.
- If "follow_up", provide exactly one brief natural therapist question that stays in the current topic.
- Do not skip required agenda topics.
- Output valid JSON only.

Current-topic dialogue:
{json.dumps(current_topic_turns, ensure_ascii=False, indent=2)}

Return:
{{
  "decision": "move_next" or "follow_up",
  "reason": "brief reason",
  "followup_question": "required if decision is follow_up; otherwise empty string"
}}
""".strip()
        return [
            {"role": "system", "content": THERAPIST_SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
        ]

    @staticmethod
    def _fallback_followup_question(*, topic: Dict[str, Any], turn_in_topic: int) -> str:
        followups = topic.get("followups") or []
        if not followups:
            return topic["opening"]
        idx = min(max(turn_in_topic - 1, 0), len(followups) - 1)
        return followups[idx]

    def _build_agenda_messages(
        self,
        *,
        transcript: List[Dict[str, Any]],
        topic: Dict[str, Any],
        turn_in_topic: int,
        previous_decision: Optional[TopicDecision],
    ) -> List[Dict[str, str]]:
        reference_questions = [topic.get("opening", "")] + list(topic.get("followups") or [])
        reference_block = "\n".join(f"  - {q}" for q in reference_questions if q)
        system_prompt = f"""{THERAPIST_SYSTEM_PROMPT}

Additional agenda-control rules:
- You are conducting a fixed-agenda intake interview.
- Current session: {topic.get("session", topic["name"])}.
- Current topic: {topic["name"]}.
- Goal of this topic: {topic.get("goal", "")}
- Reference core questions for this topic (for guidance only):
{reference_block}
- These reference questions are NOT a script. Use them to guide the goal, but
  adapt the wording, order, and focus to what the patient has already said.
  Do not ask them verbatim or re-ask things already answered.
- Ask exactly one therapist question that serves the goal above.
- Stay strictly within the current topic.
- Do not move to another topic.
- Do not explain the agenda or mention topic-control rules.
- Keep the question natural, brief, and clinically appropriate.
"""

        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        for turn in transcript:
            content = (turn.get("content") or "").strip()
            if not content:
                continue
            if turn.get("role") == "therapist":
                messages.append({"role": "assistant", "content": content})
            elif turn.get("role") == "patient":
                messages.append({"role": "user", "content": content})

        if turn_in_topic == 1:
            instruction = (
                f"Open the topic '{topic['name']}'. "
                f"Here is a reference question you may adapt to the patient's context "
                f"(do not have to ask it verbatim): {topic['opening']}"
            )
        else:
            reason = previous_decision.reason if previous_decision else "needs follow-up"
            followups = topic.get("followups") or []
            suggested = followups[min(turn_in_topic - 2, len(followups) - 1)] if followups else topic["opening"]
            instruction = (
                f"The latest answer needs a same-topic follow-up because: {reason}. "
                f"Ask one follow-up for '{topic['name']}'. "
                f"Here is a reference question you may adapt to what the patient has "
                f"already said (do not have to ask it verbatim): {suggested}"
            )

        messages.append({"role": "user", "content": instruction})
        return messages


class TopicInterviewController:
    def __init__(
        self,
        agenda: List[Dict[str, Any]] = AGENDA,
        transition_judge: Optional[TopicTransitionJudge] = None,
        therapist: Optional[Any] = None,
        transition_policy: str = "therapist",
        max_therapist_turns_per_topic: int = 8,
        verbose: bool = False,
    ):
        if max_therapist_turns_per_topic < 1:
            raise ValueError("max_therapist_turns_per_topic must be >= 1")
        if transition_policy not in {"therapist", "judge"}:
            raise ValueError("transition_policy must be one of {'therapist', 'judge'}")
        self.agenda = agenda
        self.transition_policy = transition_policy
        self.transition_judge = transition_judge or (
            GPT5TopicTransitionJudge() if transition_policy == "judge" else None
        )
        self.therapist = therapist or PromptedAgendaTherapist()
        self.max_therapist_turns_per_topic = max_therapist_turns_per_topic
        self.verbose = verbose

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message, flush=True)

    async def run(
        self,
        *,
        patient_model: Any,
        profile_item: Dict[str, Any],
        model_label: str,
        run_index: int,
        temperature: float = 0.8,
        max_tokens: int = 350,
    ) -> Dict[str, Any]:
        short_profile = _resolve_short_patient_profile(profile_item)
        transcript: List[Dict[str, Any]] = []
        profile_id = profile_item.get("id")
        self._log(f"[Interview] start profile_id={profile_id} run_index={run_index} model={model_label}")

        for topic_index, topic in enumerate(self.agenda):
            self._log(
                f"[Interview][Topic {topic_index + 1}/{len(self.agenda)}] "
                f"start key={topic['key']} name={topic['name']}"
            )
            decision = await self._run_topic(
                patient_model=patient_model,
                transcript=transcript,
                topic=topic,
                topic_index=topic_index,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            final_decision = decision["decision"]
            self._log(
                f"[Interview][Topic {topic_index + 1}/{len(self.agenda)}] "
                f"done therapist_turns={decision['therapist_turns']} "
                f"enough={final_decision.enough_information} forced_stop={final_decision.forced_stop} "
                f"reason={final_decision.reason}"
            )

        self._log(f"[Interview] complete profile_id={profile_id} run_index={run_index} topics={len(self.agenda)}")
        long_profile = _resolve_long_profile(profile_item=profile_item, patient_model=patient_model)
        return {
            "profile_id": profile_item.get("id"),
            "source_title": profile_item.get("source_title"),
            "short_patient_profile": short_profile,
            "model": model_label,
            "long_profile": long_profile,
            "transcript": _align_transcript_with_topics(transcript=transcript, agenda=self.agenda),
        }

    async def _run_topic(
        self,
        *,
        patient_model: Any,
        transcript: List[Dict[str, Any]],
        topic: Dict[str, Any],
        topic_index: int,
        temperature: float,
        max_tokens: int,
    ) -> Dict[str, Any]:
        final_decision = TopicDecision(False, "not_started", True)
        therapist_turns = 0
        pending_followup_question: Optional[str] = None

        for turn_in_topic in range(1, self.max_therapist_turns_per_topic + 1):
            therapist_turns += 1
            if pending_followup_question:
                question = pending_followup_question
                pending_followup_question = None
                self._log(
                    f"  [TopicTurn {turn_in_topic}/{self.max_therapist_turns_per_topic}] "
                    "using therapist transition-plan follow-up"
                )
            else:
                self._log(
                    f"  [TopicTurn {turn_in_topic}/{self.max_therapist_turns_per_topic}] "
                    "generating therapist question"
                )
                question = self.therapist.generate(
                    transcript=transcript,
                    topic=topic,
                    turn_in_topic=turn_in_topic,
                    previous_decision=final_decision if turn_in_topic > 1 else None,
                )
            self._log(f"  [TopicTurn {turn_in_topic}] therapist question ready (chars={len(question or '')})")
            transcript.append(
                {
                    "role": "therapist",
                    "content": question,
                    "topic_key": topic["key"],
                    "topic_name": topic["name"],
                    "topic_index": topic_index,
                    "turn_in_topic": turn_in_topic,
                }
            )

            self._log(f"  [TopicTurn {turn_in_topic}] generating patient answer")
            patient_answer = await patient_model.generate(
                transcript,
                **_build_patient_generate_kwargs(
                    patient_model=patient_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ),
            )
            self._log(f"  [TopicTurn {turn_in_topic}] patient answer ready (chars={len(patient_answer or '')})")
            transcript.append(
                {
                    "role": "patient",
                    "content": patient_answer,
                    "topic_key": topic["key"],
                    "topic_name": topic["name"],
                    "topic_index": topic_index,
                    "turn_in_topic": turn_in_topic,
                }
            )

            if self.transition_policy == "judge":
                if self.transition_judge is None:
                    raise RuntimeError("transition_policy='judge' requires a transition_judge")
                self._log(f"  [TopicTurn {turn_in_topic}] evaluating transition with judge")
                final_decision = self.transition_judge.evaluate(
                    patient_answer=patient_answer,
                    topic=topic,
                    transcript=transcript,
                )
                self._log(
                    f"  [TopicTurn {turn_in_topic}] decision "
                    f"enough={final_decision.enough_information} "
                    f"followup={final_decision.followup_needed} "
                    f"forced_stop={final_decision.forced_stop} "
                    f"reason={final_decision.reason}"
                )
                if final_decision.enough_information:
                    break
                continue

            # Therapist-driven transition policy:
            # one call decides whether to move topic or asks the next same-topic question.
            if turn_in_topic >= self.max_therapist_turns_per_topic:
                final_decision = TopicDecision(False, "therapist_policy:needs_followup", True)
                self._log(f"  [TopicTurn {turn_in_topic}] max turns reached before transition")
                break

            self._log(f"  [TopicTurn {turn_in_topic}] planning therapist-driven transition")
            plan = self.therapist.plan_next(
                transcript=transcript,
                topic=topic,
                topic_index=topic_index,
                agenda=self.agenda,
                turn_in_topic=turn_in_topic,
            )
            self._log(
                f"  [TopicTurn {turn_in_topic}] therapist plan "
                f"decision={plan.decision} reason={plan.reason}"
            )
            if plan.decision == "move_next":
                final_decision = TopicDecision(True, f"therapist_transition:{plan.reason}", False)
                break

            pending_followup_question = plan.followup_question
            final_decision = TopicDecision(False, f"therapist_followup:{plan.reason}", True)

        final_decision = TopicTransitionJudge.force_stop(final_decision)
        return {"therapist_turns": therapist_turns, "decision": final_decision}


def _resolve_short_patient_profile(profile_item: Dict[str, Any]) -> str:
    short_profile = profile_item.get("short_patient_profile")
    if isinstance(short_profile, str) and short_profile.strip():
        return short_profile.strip()

    patient_payload = profile_item.get("patient_processed_result")
    if isinstance(patient_payload, dict):
        complaint = patient_payload.get("complaints")
        if isinstance(complaint, str) and complaint.strip():
            return complaint.strip()

    return ""


def _resolve_long_profile(profile_item: Dict[str, Any], patient_model: Any) -> Optional[Dict[str, Any]]:
    getter = getattr(patient_model, "get_stage1_long_profile", None)
    if callable(getter):
        try:
            long_profile = getter()
            if isinstance(long_profile, dict):
                return long_profile
        except Exception:
            pass

    payload = profile_item.get("angel_processed_result")
    if isinstance(payload, dict):
        return payload

    return None


def _build_patient_generate_kwargs(
    *,
    patient_model: Any,
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
    try:
        params = inspect.signature(patient_model.generate).parameters
    except Exception:
        params = {}

    kwargs: Dict[str, Any] = {}
    if "temperature" in params:
        kwargs["temperature"] = temperature
    if "max_tokens" in params:
        kwargs["max_tokens"] = max_tokens
    elif "max_new_tokens" in params:
        kwargs["max_new_tokens"] = max_tokens
    return kwargs


def _align_transcript_with_topics(
    *,
    transcript: List[Dict[str, Any]],
    agenda: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    ordered_keys: List[str] = []
    topic_blocks: Dict[str, Dict[str, Any]] = {}

    for topic in agenda:
        key = topic["key"]
        ordered_keys.append(key)
        topic_blocks[key] = {
            "topic_key": key,
            "topic_name": topic.get("name", key),
            "turns": [],
        }

    for turn in transcript:
        key = str(turn.get("topic_key") or "__unknown_topic__")
        if key not in topic_blocks:
            ordered_keys.append(key)
            topic_blocks[key] = {
                "topic_key": key,
                "topic_name": turn.get("topic_name") or key,
                "turns": [],
            }

        topic_blocks[key]["turns"].append(
            {
                "role": turn.get("role"),
                "content": turn.get("content"),
                "turn_in_topic": turn.get("turn_in_topic"),
            }
        )

    return [topic_blocks[key] for key in ordered_keys if topic_blocks[key]["turns"]]


def build_agenda_therapist(mode: str) -> Any:
    if mode == "azure":
        return PromptedAgendaTherapist()
    raise ValueError(f"Unsupported therapist mode: {mode}")


def _extract_json(text: str) -> Dict[str, Any]:
    if not text:
        raise ValueError("empty JSON text")
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError(f"Could not parse JSON from: {text}")


def _clean_therapist_question(text: str) -> str:
    text = (text or "").strip()
    for prefix in ("therapist:", "assistant:"):
        if text.lower().startswith(prefix):
            text = text[len(prefix) :].strip()
    return text
