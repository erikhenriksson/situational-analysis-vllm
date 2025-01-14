from vllm import LLM, SamplingParams
from huggingface_hub import login
from vllm.engine.arg_utils import EngineArgs
import os

os.environ["HF_HOME"] = ".hf"

from dotenv import load_dotenv

load_dotenv()
import os

token = os.getenv("HF_TOKEN")

login(token=token)
# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create engine args with disabled frontend multiprocessing
engine_args = EngineArgs(
    model="meta-llama/Llama-3.3-70B-Instruct",
    tensor_parallel_size=4,
    disable_frontend_multiprocessing=True,
)

# Create an LLM with the engine args
llm = LLM(engine_args=engine_args)

# Generate texts from the prompts.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
