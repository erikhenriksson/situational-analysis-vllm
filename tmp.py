from vllm import LLM, SamplingParams  # type: ignore
from vllm.sampling_params import GuidedDecodingParams  # type: ignore
import torch  # type: ignore
from pydantic import BaseModel  # type: ignore


def LLM_setup(model, cache_dir):
    return LLM(
        model=model,
        download_dir=cache_dir,
        dtype="bfloat16",
        # max_model_len=128_000,
        tensor_parallel_size=torch.cuda.device_count(),
        # pipeline_parallel_size=2, # use if multiple nodes are needed
        # enforce_eager=False,
        gpu_memory_utilization=0.9,
    )


def generate(llm, batched_input, response_schema):

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=0.5,
        repetition_penalty=1,
        max_tokens=3000,
        guided_decoding=response_schema,
    )

    batched_outputs = llm.generate(
        batched_input, sampling_params=sampling_params, use_tqdm=False
    )

    return [out.outputs[0].text.strip(" `\n") for out in batched_outputs]


def get_response_format():
    class ResponseFormat(BaseModel):
        answer: list[str]
        explanation: list[str]

    json_schema = ResponseFormat.model_json_schema()

    return GuidedDecodingParams(json=json_schema)


def batched(data, batch_size, start_index):
    batch = []
    for i, doc in enumerate(data):
        if i < start_index:
            continue
        batch.append(doc)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def format_prompt(system, user):
    return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\x0A\x0A{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\x0A\x0A{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\x0A\x0A"


system = """You don't know anything about geography. Give me your answer and then explain why you gave that answer. The output must be formatted as a JSON object. Follow this formatting exactly.
    - Never add any preamble or anything else outside the JSON object.
    - Example: {{"answer": <"answer">, "explanation": <"explanation">}},
"""
documents = ["What is the capital of France?", "What is the capital of Paris?"]


def main():
    cache_dir = ".cache"
    model = "microsoft/phi-4"

    llm = LLM_setup(model, cache_dir)

    data = [format_prompt(system, doc) for doc in documents]
    json_schema = get_response_format()

    for _, batch in enumerate(batched(data, 1, 0)):

        batched_outputs = generate(llm, batch, json_schema)
        print(batched_outputs)


if __name__ == "__main__":
    main()
