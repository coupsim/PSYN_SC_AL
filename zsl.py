import os
import utils
import pandas as pd

from huggingface_hub import snapshot_download, login
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM


def main():
    # Login to Hugging Face
    login(token=utils.HC_TOKEN)

    for model in utils.MODELS:
        print(f"Model: {model}")
        model_dir = f"models/{model}"

        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        # Add a padding token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # Load the model
        if model == "T0_3B":
            llm = AutoModelForSeq2SeqLM.from_pretrained(model_dir, torch_dtype="auto")
        else:
            llm = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype="auto")
        # Resize the model embeddings to account for the new pad token if added
        llm.resize_token_embeddings(len(tokenizer))
        
        for dataset in utils.DATASETS:
            print(f"Dataset: {dataset}")
            # Prepare the data
            data = pd.read_csv(f"data/{dataset}_data_serialized.csv")
            data = data.sample(frac=1, random_state=utils.SEED).reset_index(drop=True)
            # Sample 250 rows from the Available label and 250 rows from the Discontinued label
            data = pd.concat([data[data["label"] == "Available"].sample(250, random_state=utils.SEED), 
                              data[data["label"] == "Discontinued"].sample(250, random_state=utils.SEED)]).reset_index(drop=True)

            answers = []
            # Loop through the data
            for i, row in data.iterrows():
                note = row["note"]
                label = row["label"]

                # Prepare the prompt
                if dataset == "arrow":
                    prompt = "Diod features: " + note + " \
                        Question: Is this diod available? Yes or no? \
                        Answer:"
                elif dataset == "phone":
                    prompt = "Phone features: " + note + " \
                        Question: Is this phone available? Yes or no? \
                        Answer:"
                else:
                    raise ValueError("Invalid dataset")
                
                # Generate text
                inputs = tokenizer(prompt, return_tensors='pt', padding=True)
                input_ids = inputs.input_ids
                attention_mask = inputs.attention_mask
                output = llm.generate(input_ids, attention_mask=attention_mask, max_length=len(input_ids[0]) + 5,
                                        eos_token_id=tokenizer.eos_token_id)
                
                # Decode the output
                if model == "T0_3B":
                    answers.append(tokenizer.decode(output[0], skip_special_tokens=True))
                else:
                    answers.append(tokenizer.decode(output[0], skip_special_tokens=True).split("Answer:")[1].split()[0])
                    answers[-1] = "Yes" if "yes" in answers[-1].lower() else answers[-1]
                    answers[-1] = "No" if "no" in answers[-1].lower() else answers[-1]
                    answers[-1] = "Yes" if "oui" in answers[-1].lower() else answers[-1]
                    answers[-1] = "No" if "non" in answers[-1].lower() else answers[-1]
                
                print(f"Model: {model}, Dataset: {dataset}, Row: {i}, Real Answer: {'Yes' if row['label']=='Available' else 'No'}, Model Answer: {answers[-1]}")

            # Save the answers to a CSV file
            data["answer"] = answers
            data.to_csv(f"results/{model}_{dataset}.csv", index=False)

if __name__ == "__main__":
    main()
