from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Dict, Any, Optional, Tuple, Union
from collections import Counter
import os
import csv
from prompts_explain import *

# Will be imported conditionally based on backend choice
openai = None
torch = None
LLM = None  # Define these at module level
SamplingParams = None


@dataclass
class InputDocument:
    true_label: str
    text: str


class BackendType(Enum):
    VLLM = auto()
    OPENAI = auto()


@dataclass
class Message:
    role: str
    content: str


@dataclass
class ClassificationResult:
    true_label: str  # Added to store the true label
    path: List[str]
    main_label: str
    sub_label: Optional[str]
    raw_responses: Dict[str, str]


class LLMBackend(ABC):
    @abstractmethod
    def generate(self, messages: List[Message]) -> str:
        pass


@dataclass
class VotingResult:
    primary_label: Optional[Tuple[str, Optional[str]]]  # (main_label, sub_label)
    secondary_label: Optional[Tuple[str, Optional[str]]]  # (main_label, sub_label)
    raw_results: List[ClassificationResult]


class VLLMBackend(LLMBackend):
    def __init__(self, model_name: str, cache_dir: str):
        global torch, LLM, SamplingParams
        if torch is None:
            import torch
        if LLM is None:
            from vllm import LLM as _LLM, SamplingParams as _SP

            LLM = _LLM
            SamplingParams = _SP

        self.llm = self._setup_llm(model_name, cache_dir)
        self.sampling_params = SamplingParams(
            temperature=0.0,
            top_p=0.5,
            repetition_penalty=1,
            max_tokens=10,
        )

    def _setup_llm(self, model: str, cache_dir: str) -> Any:
        return LLM(
            model=model,
            download_dir=cache_dir,
            dtype="bfloat16",
            tensor_parallel_size=torch.cuda.device_count(),
            gpu_memory_utilization=0.9,
        )

    def _setup_llm(self, model: str, cache_dir: str) -> LLM:
        return LLM(
            model=model,
            download_dir=cache_dir,
            dtype="bfloat16",
            tensor_parallel_size=torch.cuda.device_count(),
            gpu_memory_utilization=0.9,
        )

    def generate(self, messages: List[Message]) -> str:
        # Convert messages to vLLM format
        prompt = ""
        for msg in messages:
            if msg.role == "system":
                prompt += f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{msg.content}<|eot_id|>"
            elif msg.role == "user":
                prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{msg.content}<|eot_id|>"
            elif msg.role == "assistant":
                prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n{msg.content}<|eot_id|>"

        prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"

        outputs = self.llm.generate([prompt], sampling_params=self.sampling_params)
        return outputs[0].outputs[0].text.strip(" `\n")


class OpenAIBackend(LLMBackend):
    def __init__(self, model_name: str):
        global openai
        if openai is None:
            import openai
            from openai import OpenAI

        self.model = model_name
        if not os.getenv("OPENROUTER_API_KEY"):
            raise ValueError("OPENROUTER_API_KEY environment variable must be set")

        openai = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )

    def generate(self, messages: List[Message]) -> str:
        # Convert messages to OpenAI format
        openai_messages = [
            {"role": msg.role, "content": msg.content} for msg in messages
        ]

        response = openai.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            temperature=1,
            max_tokens=10,
            # top_p=0.5,
        )
        full_response = response.choices[0].message.content.strip()

        digit = full_response.split(":")[0].strip()

        return digit


class RegisterClassifier:
    def __init__(
        self,
        backend_type: BackendType,
        model_name: str,
        cache_dir: Optional[str] = None,
    ):
        if backend_type == BackendType.VLLM:
            if not cache_dir:
                raise ValueError("cache_dir must be provided for VLLM backend")
            self.backend = VLLMBackend(model_name, cache_dir)
        elif backend_type == BackendType.OPENAI:
            self.backend = OpenAIBackend(model_name)
        else:
            raise ValueError(f"Unsupported backend type: {backend_type}")

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

    def _generate(self, messages: List[Message]) -> str:
        return self.backend.generate(messages)

    def classify_document(self, document: InputDocument) -> ClassificationResult:
        path = []
        raw_responses = {}

        # Initialize chat with system message and document
        chat_history = [
            Message(role="system", content=SYSTEM),
            Message(role="user", content=INITIAL_PROMPT.format(document=document.text)),
        ]

        # Level 1: Check if document has enough text
        response = self._generate(chat_history)
        chat_history.append(Message(role="assistant", content=response))
        raw_responses["has_enough_text"] = response

        if response == "0":
            return ClassificationResult(
                true_label=document.true_label,
                path=["insufficient_text"],
                main_label="INSUFFICIENT",
                sub_label=None,
                raw_responses=raw_responses,
            )

        # Level 2: Mode of production
        chat_history.append(Message(role="user", content=MODE_OF_PRODUCTION))
        response = self._generate(chat_history)
        chat_history.append(Message(role="assistant", content=response))
        raw_responses["mode_of_production"] = response
        path.append(f"mode_{response}")

        if response == "1":
            # Check if interview
            chat_history.append(Message(role="user", content=IS_INTERVIEW))
            response = self._generate(chat_history)
            chat_history.append(Message(role="assistant", content=response))
            raw_responses["is_interview"] = response
            path.append(f"interview_{response}")

            main_label, sub_label = self._map_to_labels(path, raw_responses)
            return ClassificationResult(
                true_label=document.true_label,
                path=path,
                main_label=main_label,
                sub_label=sub_label,
                raw_responses=raw_responses,
            )

        elif response == "2":
            # Check if interactive
            chat_history.append(Message(role="user", content=IS_INTERACTIVE))
            response = self._generate(chat_history)
            chat_history.append(Message(role="assistant", content=response))
            raw_responses["interaction_type"] = response
            path.append(f"interaction_{response}")

            if response == "1":
                main_label, sub_label = self._map_to_labels(path, raw_responses)
                return ClassificationResult(
                    true_label=document.true_label,
                    path=path,
                    main_label=main_label,
                    sub_label=sub_label,
                    raw_responses=raw_responses,
                )

            # Get communicative purpose
            chat_history.append(Message(role="user", content=COMMUNICATIVE_PURPOSE))
            response = self._generate(chat_history)
            chat_history.append(Message(role="assistant", content=response))
            raw_responses["communicative_purpose"] = response
            path.append(f"purpose_{response}")

            if response in PURPOSE_SPECIFIC_PROMPTS:
                chat_history.append(
                    Message(role="user", content=PURPOSE_SPECIFIC_PROMPTS[response])
                )
                final_response = self._generate(chat_history)
                chat_history.append(Message(role="assistant", content=final_response))
                raw_responses["final_category"] = final_response
                path.append(f"subcategory_{final_response}")

        # Update the return statement to include true_label
        main_label, sub_label = self._map_to_labels(path, raw_responses)
        return ClassificationResult(
            true_label=document.true_label,
            path=path,
            main_label=main_label,
            sub_label=sub_label,
            raw_responses=raw_responses,
        )


def read_tsv_file(file_path: str, max_chars: int = 10000) -> List[InputDocument]:
    documents = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) >= 2:  # Ensure we have both label and text
                true_label = row[0]
                text = row[1][:max_chars]  # Truncate text to max_chars
                documents.append(InputDocument(true_label=true_label, text=text))
    return documents


def get_label_tuple(result: ClassificationResult) -> Tuple[str, Optional[str]]:
    return (result.main_label, result.sub_label)


def combine_classifications(results: List[ClassificationResult]) -> VotingResult:
    """
    Combine multiple classification results using voting logic.
    Returns primary and secondary labels if they exist based on voting rules.
    """
    if not results or len(results) != 4:
        raise ValueError("Exactly 4 classification results are required")

    # Count occurrences of each unique label combination
    label_counts = Counter(get_label_tuple(result) for result in results)

    # Sort by frequency, then by label for consistency
    sorted_labels = sorted(label_counts.items(), key=lambda x: (-x[1], x[0]))

    primary_label = None
    secondary_label = None

    # If highest count is 3 or 4, it's the primary label
    if sorted_labels[0][1] >= 3:
        primary_label = sorted_labels[0][0]
    # If we have two pairs of 2 votes each
    elif (
        len(sorted_labels) >= 2
        and sorted_labels[0][1] == 2
        and sorted_labels[1][1] == 2
    ):
        primary_label = sorted_labels[0][0]
        secondary_label = sorted_labels[1][0]
    # If we have one pair of 2 votes (and two singles)
    elif sorted_labels[0][1] == 2:
        primary_label = sorted_labels[0][0]

    return VotingResult(
        primary_label=primary_label,
        secondary_label=secondary_label,
        raw_results=results,
    )


def format_label(main_label: str, sub_label: Optional[str] = None) -> str:
    """Format a label pair into a single string."""
    if sub_label:
        return f"{main_label} {sub_label}"
    return main_label


def format_true_label(label: str) -> str:
    """Format true label by sorting components."""
    parts = label.split()
    return " ".join(sorted(parts))


def format_voting_result(voting_result: VotingResult) -> str:
    """Format voting result into a single sorted string."""
    labels = []

    if voting_result.primary_label:
        main_label, sub_label = voting_result.primary_label
        labels.append(format_label(main_label, sub_label))

    if voting_result.secondary_label:
        main_label, sub_label = voting_result.secondary_label
        labels.append(format_label(main_label, sub_label))

    return " ".join(sorted(labels))


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Run register classification with specified backend"
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["vllm", "openai"],
        required=True,
        help="Backend to use for classification",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name to use",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        help="Cache directory for vLLM (required if using vLLM backend)",
    )
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to input TSV file containing true labels and documents",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to output JSONL file for results",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=100000,
        help="Maximum number of characters to use from each document",
    )

    args = parser.parse_args()

    if args.backend == "vllm":
        backend_type = BackendType.VLLM
    elif args.backend == "openai":
        backend_type = BackendType.OPENAI
    else:
        raise ValueError(f"Unsupported backend: {args.backend}")

    if backend_type == BackendType.VLLM and not args.cache_dir:
        parser.error("--cache-dir is required when using vLLM backend")

    classifier = RegisterClassifier(
        backend_type=backend_type, model_name=args.model, cache_dir=args.cache_dir
    )

    documents = read_tsv_file(args.input_path, args.max_chars)

    with open(args.output_path, "w", encoding="utf-8") as f:
        for i, document in enumerate(documents, 1):
            print(f"\nProcessing document {i}/{len(documents)}")

            # Run classification 4 times
            results = []
            for _ in range(4):
                result = classifier.classify_document(document)
                results.append(result)

            # Combine results using voting
            voting_result = combine_classifications(results)

            # Format output
            output = {
                "true_label": format_true_label(document.true_label),
                "predicted_label": format_voting_result(voting_result),
            }

            # Write to file and print
            json_line = json.dumps(output)
            f.write(json_line + "\n")
            f.flush()  # Force write to disk
            print(json_line)


if __name__ == "__main__":
    main()
