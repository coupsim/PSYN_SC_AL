import pandas as pd
import os


def main():
    # For every dataset: arrow_data.xlsx, phone_data.csv, sncf_data.xlsx
    datasets = ['arrow', 'phone', 'sncf']
    for dataset in datasets:
        if dataset == 'arrow':
            # Load the data from the Excel file
            df = pd.read_excel("data/" + dataset + "_data.xlsx", sheet_name=[0, 1])

            # Add label column to the data
            df[0]["label"] = ["Discontinued"] * df[0].shape[0]
            df[1]["label"] = ["Available"] * df[1].shape[0]
            df = pd.concat([df[0], df[1]]).reset_index(drop=True)

            # Fill missing data with NA
            df = df.fillna("NA")
        elif dataset == 'phone':
            # Load the data from the CSV file
            df = pd.read_csv("data/" + dataset + "_data.csv", on_bad_lines='skip')

            # Rename the status column to label
            df = df.rename(columns={'status': 'label'})

            # Split the label column to get the status
            split_data = df["label"].str.split(".")
            split_data = split_data.tolist()
            split_data = [item[0] for item in split_data] if len(split_data[0]) > 1 else split_data

            # Replace the status with the correct label
            split_data = ["Available" if item == "Coming soon" else item for item in split_data]
            split_data = ["Discontinued" if item == "Cancelled" else item for item in split_data]
            
            # Update the label column
            cols = df.columns.tolist()
            cols.remove('label')
            cols.append('label')
            df = df[cols]
            df['label'] = split_data
        elif dataset == 'sncf':
            # Load the data from the Excel file
            df = pd.read_excel("data/" + dataset + "_data.xlsx")
        else:
            # Raise an exception if the dataset name is invalid
            raise ValueError("Invalid dataset name!")


        # Serialize the dataset
        serialized_data = []
        for index, row in df.iterrows():
            obso_case = ""
            for column in df.columns:
                if column != 'label':
                    obso_case += f"The {column} is {row[column]}. "
                else:
                    obso_solution = row[column]
            serialized_data.append({
                'note': obso_case,
                'label': obso_solution
            })
        
        # Create a new DataFrame with the serialized data
        serialized_df = pd.DataFrame(serialized_data)
        
        # Save the serialized dataset to a CSV file
        output_file_path = os.path.join('data', dataset + '_data_serialized.csv')
        serialized_df.to_csv(output_file_path, index=False)

if __name__ == "__main__":
    main()