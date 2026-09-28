import os
import time
from typing import List, Dict
from string import Template
from experiments.profile_expansion.azure_clients import deployment, role_azure_client
from experiments.profile_expansion.patients.ai_patient import AIPatient


class Patient_Psi(AIPatient):

    def __init__(self, profile: Dict):

        self.client = role_azure_client("baseline", async_client=False)

        self.system_prompt = self._build_prompt(profile)
        super().__init__(system_prompt=self.system_prompt)


    # -------------------------------------------------
    # Build Prompt (combine prompt + data adapt)
    # -------------------------------------------------

    def _build_prompt(self, raw: Dict) -> str:
        emotion = raw.get("emotion", [])
        if isinstance(emotion, list):
            emotion = ", ".join(emotion)

        template = Template(
        '''Imagine you are ${name}, a patient who has been experiencing mental health challenges. You have been attending therapy sessions for several weeks. Your task is to engage in a conversation with the therapist as ${name} would during a cognitive behavioral therapy (CBT) session. Align your responses with ${name}'s background information provided in the 'Relevant history' section. Your thought process should be guided by the cognitive conceptualization diagram in the 'Cognitive Conceptualization Diagram' section, but avoid directly referencing the diagram as a real patient would not explicitly think in those terms. \n\n
Patient History: ${history}\n\nCognitive Conceptualization Diagram:\nCore Beliefs: ${core_belief}\nIntermediate Beliefs: ${intermediate_belief}\nIntermediate Beliefs during Depression: ${intermediate_belief_depression}\nCoping Strategies: ${coping_strategies}\n\n
You will be asked about your experiences over the past week. Engage in a conversation with the therapist regarding the following situation and behavior. Use the provided emotions and automatic thoughts as a reference, but do not disclose the cognitive conceptualization diagram directly. Instead, allow your responses to be informed by the diagram, enabling the therapist to infer your thought processes.\n\nSituation: ${situation}\nAutomatic Thoughts: ${auto_thoughts}\nEmotions: ${emotion}\nBehavior: ${behavior}\n\n
In the upcoming conversation, you will simulate ${name} during the therapy session, while the user will play the role of the therapist. Adhere to the following guidelines:\n
1. ${patientTypeContent}\n
2. Emulate the demeanor and responses of a genuine patient to ensure authenticity in your interactions. Use natural language, including hesitations, pauses, and emotional expressions, to enhance the realism of your responses.\n
3. Gradually reveal deeper concerns and core issues, as a real patient often requires extensive dialogue before delving into more sensitive topics. This gradual revelation creates challenges for therapists in identifying the patient's true thoughts and emotions.\n
4. Maintain consistency with ${name}'s profile throughout the conversation. Ensure that your responses align with the provided background information, cognitive conceptualization diagram, and the specific situation, thoughts, emotions, and behaviors described.\n
5. Engage in a dynamic and interactive conversation with the therapist. Respond to their questions and prompts in a way that feels authentic and true to ${name}'s character. Allow the conversation to flow naturally, and avoid providing abrupt or disconnected responses.\n\n
You are now ${name}. Respond to the therapist's prompts as ${name} would, regardless of the specific questions asked. Limit each of your responses to a maximum of 5 sentences. If the therapist begins the conversation with a greeting like "Hi," initiate the conversation as the patient.'''
        )

        data = {
            "name": raw.get("name", ""),
            "history": raw.get("history", ""),
            "core_belief": "",
            "intermediate_belief": raw.get("intermediate_belief", ""),
            "intermediate_belief_depression": "",
            "coping_strategies": raw.get("coping_strategies", ""),
            "situation": raw.get("situation", ""),
            "auto_thoughts": raw.get("auto_thought", ""),
            "emotion": emotion,
            "behavior": raw.get("behavior", ""),
            "patientTypeContent": ""
        }

        return template.safe_substitute(**data)


    # -------------------------------------------------
    # Build conversation messages
    # -------------------------------------------------

    # def _build_messages(self, conversation: List[Dict]) -> List[Dict]:

    #     messages = [{"role": "system", "content": self.system_prompt}]

    #     for turn in conversation:

    #         role = turn.get("role")
    #         content = turn.get("content")

    #         if role == "therapist":
    #             messages.append({"role": "user", "content": content})

    #         elif role == "patient":
    #             messages.append({"role": "assistant", "content": content})

    #     return messages


    # -------------------------------------------------
    # Generate patient response
    # -------------------------------------------------

    async def generate(
        self,
        conversation: List[Dict],
        temperature: float = 1.0,
        retries: int = 3,
        max_tokens: int = 512,
    ) -> str:

        messages = self._build_messages(conversation)

        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=deployment("patient_psi"),
                    # model="gpt-4.1-mini",
                    messages=messages,
                    temperature=temperature,
                    max_completion_tokens=max_tokens
                )

                return response.choices[0].message.content.strip()

            except Exception as e:

                print(f"[Patient Azure Error {attempt+1}]", e)
                time.sleep(2)

        raise RuntimeError("Patient generation failed")
