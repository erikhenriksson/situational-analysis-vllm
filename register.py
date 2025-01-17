from vllm import LLM, SamplingParams
import torch
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


@dataclass
class ClassificationResult:
    path: List[str]
    main_label: str
    sub_label: Optional[str]
    raw_responses: Dict[str, str]


class RegisterClassifier:
    def __init__(self, model_name: str, cache_dir: str):
        self.llm = self._setup_llm(model_name, cache_dir)
        self.sampling_params = SamplingParams(
            temperature=0.0,
            top_p=0.5,
            repetition_penalty=1,
            max_tokens=10,
        )

    def _map_to_labels(
        self, path: List[str], responses: Dict[str, str]
    ) -> Tuple[str, Optional[str]]:
        # Mode 3 is machine translated
        if "mode_3" in path:
            return "MT", None

        # Lyrical content
        if "purpose_6" in path:
            return "LY", None

        # Spoken content
        if "mode_1" in path:
            if responses.get("is_interview") == "1":
                return "SP", "it"
            return "SP", None

        # Interactive discussion
        if "interaction_1" in path:
            return "ID", None

        # Get the purpose and subcategory
        purpose = responses.get("communicative_purpose")
        subcategory = responses.get("final_category")

        if purpose == "1":  # Narrative
            if subcategory == "1":
                return "NA", "ne"  # News
            elif subcategory == "2":
                return "NA", "sr"  # Sports report
            elif subcategory == "3":
                return "NA", "nb"  # Narrative blog
            return "NA", None  # Other narrative

        elif purpose == "5":  # How-to/Instructions
            if subcategory == "1":
                return "HI", "re"  # Recipe
            return "HI", None  # Other how-to

        elif purpose == "2":  # Informative
            if subcategory == "1":
                return "IN", "en"  # Encyclopedia
            elif subcategory == "2":
                return "IN", "ra"  # Research article
            elif subcategory == "3":
                return "IN", "dtp"  # Description of thing/person
            elif subcategory == "4":
                return "IN", "fi"  # FAQ
            elif subcategory == "5":
                return "IN", "lt"  # Legal terms
            return "IN", None  # Other informative

        elif purpose == "3":  # Opinion
            if subcategory == "1":
                return "OP", "rv"  # Review
            elif subcategory == "2":
                return "OP", "ob"  # Opinion blog
            elif subcategory == "3":
                return "OP", "rs"  # Religious
            elif subcategory == "4":
                return "OP", "av"  # Advice
            return "OP", None  # Other opinion

        elif purpose == "4":  # Info-persuasion
            if subcategory == "1":
                return "IP", "ds"  # Description to sell
            elif subcategory == "2":
                return "IP", "ed"  # Editorial
            return "IP", None  # Other persuasive

        return "NA", None  # Default case

    def _setup_llm(self, model: str, cache_dir: str) -> LLM:
        return LLM(
            model=model,
            download_dir=cache_dir,
            dtype="bfloat16",
            tensor_parallel_size=torch.cuda.device_count(),
            gpu_memory_utilization=0.9,
        )

    def _format_prompt(self, system: str, user: str) -> str:
        return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    def _generate(self, prompt: str) -> str:
        outputs = self.llm.generate([prompt], sampling_params=self.sampling_params)
        return outputs[0].outputs[0].text.strip(" `\n")

    def classify_document(self, document: str) -> ClassificationResult:
        path = []
        raw_responses = {}

        # Level 1: Check if document has enough text
        system_prompt = """You are a helpful assistant that classifies text registers. Answer exactly as instructed with no additional text."""

        user_prompt = f"""Does the following document have enough text to be rated for register? If not, answer exactly "0". If yes, answer exactly "1" (without quotes).

Document:
{document}"""

        response = self._generate(self._format_prompt(system_prompt, user_prompt))
        raw_responses["has_enough_text"] = response

        if response == "0":
            return ClassificationResult(
                path=["insufficient_text"],
                main_label="NA",
                sub_label=None,
                raw_responses=raw_responses,
            )

        # Level 2: Mode of production
        user_prompt = f"""What is the mode of production of the following text? Is it:
(1) Internet texts that originated in the spoken mode (e.g., transcripts of speeches or interviews)
(2) Internet texts that originated in the written mode
(3) Internet texts that are clearly machine translated or generated from a template
Output exactly "1", "2", or "3" (without quotes).

Document:
{document}"""

        response = self._generate(self._format_prompt(system_prompt, user_prompt))
        raw_responses["mode_of_production"] = response
        path.append(f"mode_{response}")

        if response == "1":
            # Check if interview
            user_prompt = f"""Is the following text an interview? If yes, answer "1". If not, output "0" (without quotes).

Document:
{document}"""

            response = self._generate(self._format_prompt(system_prompt, user_prompt))
            raw_responses["is_interview"] = response
            path.append(f"interview_{response}")

            main_label, sub_label = self._map_to_labels(path, raw_responses)
            return ClassificationResult(
                path=path,
                main_label=main_label,
                sub_label=sub_label,
                raw_responses=raw_responses,
            )

        elif response == "2":
            # Check if interactive
            user_prompt = f"""Is the document:
(1) an interactive discussion (e.g., discussion forums)
(2) a non-interactive Internet text
Note that non-interactive texts followed by reader comments should be considered (2) non-interactive.
Output exactly "1" or "2" (without quotes).

Document:
{document}"""

            response = self._generate(self._format_prompt(system_prompt, user_prompt))
            raw_responses["interaction_type"] = response
            path.append(f"interaction_{response}")

            if response == "1":
                main_label, sub_label = self._map_to_labels(path, raw_responses)
                return ClassificationResult(
                    path=path,
                    main_label=main_label,
                    sub_label=sub_label,
                    raw_responses=raw_responses,
                )

            # Get communicative purpose
            user_prompt = f"""What is the communicative purpose of the following text?
(1) to narrate events
(2) to describe information
(3) to express opinion
(4) to use facts to persuade
(5) to explain instructions
(6) to express oneself lyrically
Output exactly "1", "2", "3", "4", "5", or "6" (without quotes).

Document:
{document}"""

            response = self._generate(self._format_prompt(system_prompt, user_prompt))
            raw_responses["communicative_purpose"] = response
            path.append(f"purpose_{response}")

            # Handle each communicative purpose
            purpose_prompts = {
                "1": """Is the text a:
(1) news report/blog
(2) sports report
(3) personal/diary blog
(4) text with some other narrative communicative purpose
Output exactly "1", "2", "3", or "4" (without quotes).""",
                "2": """Is the text an:
(1) encyclopedia article
(2) research article
(3) description of a thing or a person
(4) frequently asked questions page about information
(5) legal terms and conditions
(6) text with some other communicative purpose to describe information
Output exactly "1", "2", "3", "4", "5", or "6" (without quotes).""",
                "3": """Is the text a:
(1) review
(2) opinion blog
(3) religious blog or sermon
(4) advice based on personal opinion
(5) text with some other communicative purpose to express opinion
Output exactly "1", "2", "3", "4", or "5" (without quotes).""",
                "4": """Is the text a:
(1) description with intent to sell
(2) news & opinion blog or editorial with the purpose to persuade the reader
(3) text with some other communicative purpose to persuade the reader with facts
Output exactly "1", "2", or "3" (without quotes).""",
                "5": """Is the text a:
(1) recipe
(2) some other communicative purpose to explain instructions
Output exactly "1" or "2" (without quotes).""",
            }

            if response in purpose_prompts:
                user_prompt = f"""{purpose_prompts[response]}

Document:
{document}"""

                final_response = self._generate(
                    self._format_prompt(system_prompt, user_prompt)
                )
                raw_responses["final_category"] = final_response
                path.append(f"subcategory_{final_response}")

        # Map to final labels and return result
        main_label, sub_label = self._map_to_labels(path, raw_responses)
        return ClassificationResult(
            path=path,
            main_label=main_label,
            sub_label=sub_label,
            raw_responses=raw_responses,
        )


def main():
    cache_dir = "/scratch/project_462000642/ehenriks/situational-analysis-vllm/cache"
    model = "meta-llama/Llama-3.3-70B-Instruct"

    classifier = RegisterClassifier(model, cache_dir)

    # Example document - a news article
    document = """In a groundbreaking development, scientists at the University of Technology 
    announced today their successful creation of a new renewable energy source. 
    The team, led by Dr. Sarah Johnson, has developed a novel method to harness 
    solar energy with unprecedented efficiency. "This breakthrough could 
    revolutionize how we power our homes," said Dr. Johnson at the press conference. 
    The research, which took five years to complete, demonstrates a 40% increase 
    in energy capture compared to traditional solar panels."""

    result = classifier.classify_document(document)

    print("\nDocument Classification Results")
    print("-" * 30)
    print(f"Input Document Preview:\n{document[:150]}...\n")
    print(f"Main Label: {result.main_label}")
    if result.sub_label:
        print(f"Sub Label: {result.sub_label}")
    print("\nClassification Path:")
    for i, step in enumerate(result.path, 1):
        print(f"{i}. {step}")
    print("\nRaw LLM Responses:")
    for question, response in result.raw_responses.items():
        print(f"{question}: {response}")


if __name__ == "__main__":
    main()
