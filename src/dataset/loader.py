import pandas as pd

def load_dataset(dataset_path: str, translation_path: str = 'data/translation_df.csv') -> pd.DataFrame:
    """
    Loads a dataset from a parquet file and translates the column names from Swedish to English using a provided translation file.
    The data is from 2019-01-01 to 
    
    Args:
        dataset_path (str): The file path to the dataset in parquet format.
        translation_path (str, optional): The file path to the CSV file containing Swedish to English translations of column names. Defaults to 'data/translation_df.csv'.
    
    Returns:
        pd.DataFrame: The loaded dataset with translated and cleaned column names.
    """
    dataset_df = pd.read_parquet(dataset_path)
    translations_df = pd.read_csv(translation_path)
    
    translation_reference = translations_df.set_index('Swedish')['English'].to_dict()

    translated_columns = {col: translation_reference.get(col, col) for col in dataset_df.columns}
    dataset_df.rename(columns=translated_columns, inplace=True)
    dataset_df.columns = dataset_df.columns.str.replace(' ', '_')
    
    return dataset_df
