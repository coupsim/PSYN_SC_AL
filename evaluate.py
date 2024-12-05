import utils
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score


def main():
    for model in utils.MODELS:
        for dataset in utils.DATASETS:
            # Load the CSV data
            data = pd.read_csv("results/" + model + "_" + dataset + ".csv")
            
            # Get the valid and invalid indexes
            invalid = data[(data['answer'] != 'Yes') & (data['answer'] != 'No')].index
            valid = data[(data['answer'] == 'Yes') | (data['answer'] == 'No')].index

            # Create a new dataset with only the valid indexes
            evaluate = data.loc[valid]
            
            # Change all the 'Yes' and 'No' to 1 and 0
            evaluate['answer'] = evaluate['answer'].apply(lambda x: 1 if x == 'Yes' else 0)
            evaluate['label'] = evaluate['label'].apply(lambda x: 1 if x == 'Available' else 0)

            # Calculate the accuracy, precision, recall, F1 score, and AUC score
            accuracy = accuracy_score(evaluate['label'], evaluate['answer'])
            precision = precision_score(evaluate['label'], evaluate['answer'])
            recall = recall_score(evaluate['label'], evaluate['answer'])
            f1 = f1_score(evaluate['label'], evaluate['answer'])
            auc = roc_auc_score(evaluate['label'], evaluate['answer'])
            confusion = confusion_matrix(evaluate['label'], evaluate['answer'])

            # Print the results
            print(f"Model: {model}, Dataset: {dataset}")
            print(f"Accuracy: {accuracy}")
            print(f"Precision: {precision}")
            print(f"Recall: {recall}")
            print(f"F1 Score: {f1}")
            print(f"AUC Score: {auc}")
            print(f"Confusion Matrix: {confusion}")


            if len(invalid) > 0:
                print(f"Invalid answers found in {model} - {dataset}: {invalid}")
                print(f"Number of Invalid and Valid answers: Invalid - {len(invalid)} + Valid {len(valid)} = Total {len(invalid) + len(valid)}")
                print(f"Unique Invalid answers: {data.loc[invalid, 'answer'].unique()}")

                # Save the results to a CSV file
                results = pd.DataFrame({'model': [model], 'dataset': [dataset], 'accuracy': [accuracy], 'precision': [precision], 'recall': [recall], 'f1': [f1], 'auc': [auc], 'invalid': [len(invalid)], 'valid': [len(valid)], 'ds_size': [len(invalid) + len(valid)], 'unique_invalid': [data.loc[invalid, 'answer'].unique()]})
                results.to_csv("evaluations/" + model + "_" + dataset + ".csv", index=False)
            else:
                # Save the results to a CSV file
                results = pd.DataFrame({'model': [model], 'dataset': [dataset], 'accuracy': [accuracy], 'precision': [precision], 'recall': [recall], 'f1': [f1], 'auc': [auc]})
                results.to_csv("evaluations/" + model + "_" + dataset + ".csv", index=False)

if __name__ == "__main__":
    main()