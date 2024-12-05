import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, roc_curve, auc

arrow_data = pd.read_csv("data/arrow_data_serialized.csv")
phone_data = pd.read_csv("data/phone_data_serialized.csv")

# print dataset sizes, number of elements in each label, and percentage of each label
print("Arrow dataset:")
print(arrow_data.shape)
print(arrow_data["label"].value_counts())
print(arrow_data["label"].value_counts(normalize=True))

print("\nPhone dataset:")
print(phone_data.shape)
print(phone_data["label"].value_counts())
print(phone_data["label"].value_counts(normalize=True))


datasets = ["arrow", "phone"]
models = ["T0_3B",
          "Llama-3.2-3B-Instruct",
          "gemma-2-2b-it",
          "Phi-3.5-mini-instruct"]


for dataset in datasets:
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for model in models:
        if model != "Phi-3.5-mini-instruct":
            data = pd.read_csv(f"results/{model}_{dataset}.csv")
            labels = data["label"].apply(lambda x: 1 if x == "Available" else 0)
            answers = data["answer"].apply(lambda x: 1 if x == "Yes" else 0)
        else:
            if dataset == "arrow":
                data = pd.read_csv(f"results/{model}_{dataset}.csv")

                # Get the valid and invalid indexes
                invalid = data[(data['answer'] != 'Yes') & (data['answer'] != 'No')].index
                valid = data[(data['answer'] == 'Yes') | (data['answer'] == 'No')].index

                # Create a new dataset with only the valid indexes
                evaluate = data.loc[valid]
                
                # Change all the 'Yes' and 'No' to 1 and 0
                evaluate['answer'] = evaluate['answer'].apply(lambda x: 1 if x == 'Yes' else 0)
                evaluate['label'] = evaluate['label'].apply(lambda x: 1 if x == 'Available' else 0)

                labels = evaluate["label"]
                answers = evaluate["answer"]
            else:
                continue

        fpr[model], tpr[model], _ = roc_curve(labels, answers, pos_label=1)

        roc_auc[model] = auc(fpr[model], tpr[model])

    # make font bigger
    plt.rcParams.update({'font.size': 14})
    plt.rcParams["font.family"] = "Times New Roman"
    for model in fpr:
        plt.plot(fpr[model], tpr[model], lw = 2, label = f"{model}: {roc_auc[model]:.2f}")

    plt.plot([0, 1], [0, 1],
            linestyle = '--',
            color = (0.6, 0.6, 0.6),
            label = 'random guessing')
    plt.plot([0, 0, 1], [0, 1, 1],
            linestyle = ':',
            color = 'black', 
            label = 'perfect performance')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    # plt.title('Receiver Operator Characteristic')
    plt.legend(loc = "lower right")
    plt.tight_layout()
    plt.savefig(f"media/{dataset}_roc.png")
    plt.show()
            