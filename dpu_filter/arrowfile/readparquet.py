import sys
import pandas as pd

def read_and_print_parquet(file_path):
    # Read the Parquet file into a DataFrame
    df = pd.read_parquet(file_path)
    
    # Print the DataFrame
    print(df)

if __name__ == "__main__":
    # Check if the correct number of command-line arguments is provided
    if len(sys.argv) != 2:
        print("Usage: python script.py <parquet_file>")
        sys.exit(1)

    # Get the file path from command-line arguments
    parquet_file = sys.argv[1]

    # Call the function with the provided file path
    read_and_print_parquet(parquet_file)

