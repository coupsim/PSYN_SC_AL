import os
import utils

from huggingface_hub import snapshot_download, login


def main():
    model_ids = ["bigscience/T0_3B", 
                 "meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-1B-Instruct", "meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-3B", "meta-llama/Llama-3.2-3B-Instruct",
                 "nvidia/Nemotron-Mini-4B-Instruct", 
                 "ibm-granite/granite-3.0-1b-a400m-base", "ibm-granite/granite-3.0-1b-a400m-instruct", "ibm-granite/granite-3.0-3b-a800m-base", "ibm-granite/granite-3.0-3b-a800m-instruct", "ibm-granite/granite-3.0-2b-base", "ibm-granite/granite-3.0-2b-instruct",
                 "microsoft/Phi-3.5-mini-instruct",
                 "google/gemma-2-2b", "google/gemma-2-2b-it",
                 "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-3B-Instruct"]
    
    # Login to Hugging Face
    login(token=utils.HC_TOKEN)

    for model_id in model_ids:
        print(f"Downloading model {model_id}...")

        # Specify the directory to download the model
        model_dir = "./models/" + model_id.split("/")[1]
        if not os.path.exists(model_dir):
            snapshot_download(repo_id=model_id, local_dir=model_dir)

if __name__ == "__main__":
    main()