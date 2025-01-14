from vllm import LLM, SamplingParams
from multiprocessing import freeze_support
from huggingface_hub import login
import os

os.environ["HF_HOME"] = ".hf"

from dotenv import load_dotenv

load_dotenv()

token = os.getenv("HF_TOKEN")

login(token=token)


def format_prompt(system, user):
    return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\x0A\x0A{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\x0A\x0A{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\x0A\x0A"


def run():
    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    # Create an LLM.
    llm = LLM(
        model="meta-llama/Llama-3.3-70B-Instruct",
        tensor_parallel_size=4,
        download_dir=".cache",
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
        max_model_len=128_000,
    )
    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == "__main__":
    freeze_support()
    run()
