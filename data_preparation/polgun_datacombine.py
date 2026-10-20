import pandas as pd
import os
def add_time_difference_column(df):
    """
    Calculates the time difference for each row relative to the first row and
    adds it as a new column to the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame with a datetime column named 'Time'.

    Returns:
        pd.DataFrame: The DataFrame with a new column 'time_difference' containing
                      the time differences in seconds.
    """
    first_timestamp = df['Time'].iloc[0]
    df['time_difference'] = (df['Time'] - first_timestamp).dt.total_seconds()
    return df

def timestamp_csv_file(original_name, original_path):
    """
    Reads a CSV file, adds a time difference column, reorders the columns,
    and saves the modified DataFrame to a new CSV file.
    """
    # Define the original filename of the uploaded CSV file
    #original_name = 'anode_current_10222020.csv'
    # In your local script, you would use:
    #original_path = '/Users/wange/Coding/Python/ai_conditioning/gun_comm_data/gun_conditioning/'
    
    # 1. Read the CSV file, parsing the first column as a datetime object
    original_filename = os.path.join(original_path, original_name)
    df = pd.read_csv(original_filename, parse_dates=[0])
    first_timestamp = df['Time'].iloc[0]
    df['time_difference'] = (df['Time'] - first_timestamp).dt.total_seconds()

    # 2. Call the function to add the new column
    df_with_diff = df.copy()
    

    # 3. Reorder columns: move 'time_difference' to the second position
    cols = df_with_diff.columns.tolist()
    cols.remove('time_difference')
    cols.insert(1, 'time_difference')
    df_reordered = df_with_diff[cols]

    # 4. Define the new file name based on the original name
    base_name, ext = os.path.splitext()
    new_filename = f"{base_name}_timestamp{ext}"


    # 5. Save the reordered DataFrame to a new CSV file
    df_reordered.to_csv(new_filename, index=False)



    # Display the first 5 rows of the reordered dataframe to verify
    print(df_reordered.head().to_markdown(index=False, numalign="left", stralign="left"))
    return new_filename


def process_time_data(original_name,original_path):
    """
    Reads a CSV file, keeps only the first row for each unique timestamp
    in the first column, and saves the processed data to a new file.

    Args:
        input_filename (str): The name of the input CSV file.

    Returns:
        str: The name of the output CSV file created.
    """
    input_filename = os.path.join(original_path,original_name)
    # 1. Read the CSV file, parsing the first column as datetime objects
    df = pd.read_csv(input_filename, parse_dates=[0])

    # Get the name of the first column (time column)
    time_column_name = df.columns[0]

    # 2. Keep only the first line of each same time row
    df_processed = df.drop_duplicates(subset=[time_column_name], keep='first')

    # 3. Construct the output filename
    # Split the filename into base and extension
    base_name, ext = os.path.splitext(input_filename)
    output_filename = f"{base_name}_hz{ext}"

    # 4. Save the processed DataFrame to the new file
    df_processed.to_csv(output_filename, index=False)

    return output_filename

def combine(original_file1,original_file2,col_indices_str,file_path):
    """
    Combines two CSV files by copying specified columns from file2 to file1,
    matching rows based on the 'Time' column (first column).

    Args:
        file1 (str): The filename of the base CSV file (e.g., 'gun_voltage_10222020_hz.csv').
                     Assumed to have a 'Time' column at index 0.
        file2 (str): The filename of the source CSV file (e.g., 'geiger_count_10222020.csv').
                     Assumed to have its primary time column at index 0.
        col_indices_str (str): A comma-separated string of 0-based column indices
                                from file2 to copy (e.g., "1,3,5,7" for gc_diff columns).

    Returns:
        str: The name of the new combined CSV file.
    """
    # 1. Load file1 (base file) - parsing its first column as datetime
    file1 = os.path.join(file_path, original_file1)
    df1 = pd.read_csv(file1, parse_dates=[0])
    time_col_df1 = df1.columns[0] # Get the actual name of the time column in df1

    # 2. Load file2 (source file) - load without initial date parsing, then convert explicitly
    
    file2 = os.path.join(file_path,original_file2)
    df2 = pd.read_csv(file2)
    time_col_df2 = df2.columns[0] # Get the actual name of the time column in df2

    # Explicitly convert the primary time column of df2 to datetime, coercing errors to NaT
    df2[time_col_df2] = pd.to_datetime(df2[time_col_df2], errors='coerce')

    # Drop rows where the time column became NaT (Not a Time) after coercion
    df2.dropna(subset=[time_col_df2], inplace=True)

    # 3. Parse the column indices string into a list of integers
    cols_to_copy_indices = [int(c) for c in col_indices_str.split(',')]

    # 4. Select the specified columns from file2, plus its time column for merging
    # Ensure all selected indices are valid
    for idx in cols_to_copy_indices:
        if idx >= len(df2.columns):
            raise ValueError(f"Column index {idx} is out of bounds for file2: {file2} which has {len(df2.columns)} columns.")

    # Get the names of the columns to be copied, including the time column from df2 for merging
    selected_cols_names = [time_col_df2] + [df2.columns[i] for i in cols_to_copy_indices]
    
    # Create a temporary DataFrame with only the relevant columns from df2
    temp_df2_for_merge = df2[selected_cols_names].copy()
    
    # Drop duplicates from temp_df2_for_merge based on its time column to ensure a clean merge
    temp_df2_for_merge = temp_df2_for_merge.drop_duplicates(subset=[time_col_df2], keep='first')

    # 5. Combine DataFrames using a left merge on the time column
    combined_df = pd.merge(df1, temp_df2_for_merge, left_on=time_col_df1, right_on=time_col_df2, how='left')

    # Handle potential duplicate time column in the merged_df
    if time_col_df1 != time_col_df2 and time_col_df2 in combined_df.columns:
        combined_df = combined_df.drop(columns=[time_col_df2])
    elif f"{time_col_df1}_x" in combined_df.columns and f"{time_col_df1}_y" in combined_df.columns:
        combined_df = combined_df.drop(columns=[f"{time_col_df1}_y"])
        combined_df = combined_df.rename(columns={f"{time_col_df1}_x": time_col_df1})
    
    # 6. Apply row_ini logic: Fill with 0s for initial rows in the new columns
    # Identify the columns that were newly added from file2. These are the ones in `selected_cols_names`
    # excluding the time column (which is time_col_df2).
    new_data_cols = [col for col in selected_cols_names if col != time_col_df2]
    '''
    if row_ini > 0:
        # Fill NaN values (resulting from the left merge where no match was found) with 0s
        # specifically for rows before row_ini and for the newly added columns.
        for col in new_data_cols:
            if col in combined_df.columns: # Ensure the column exists after merge
                # Use .loc for setting values to avoid SettingWithCopyWarning
                combined_df.loc[combined_df.index < row_ini, col] = 0
    '''
    # 7. Apply conditional filling for missing values in the newly added columns:
    #    - If continuous empty values < 3, then interpolate (limit=2).
    #    - If continuous empty values >= 3, then leave as NaN.
    for col in new_data_cols:
        if col in combined_df.columns: # Ensure the column exists
            combined_df[col].interpolate(method='linear', inplace=True)

    # 6. Define the output filename (user didn't specify, so using a default)
    #output_filename = "combined_output.csv"
    base_name, ext = os.path.splitext(file1)
    output_filename = f"{base_name}_combined{ext}"
    # 7. Save the combined DataFrame
    combined_df.to_csv(output_filename, index=False)

    return output_filename


if __name__ == "__main__":#"C:\Users\skantamne\Downloads\PhD EGun\Data\lerec_voltage_03252025_0930.csv"
    # Example usage
    original_name = "gun_voltage_12222020_combined.csv"
    original_path = r"C:\Users\skantamne\Downloads\PhD EGun\Data\Archive 2\polgun_conditioning"
    remove_row = False  # Set to True to remove rows with the same timestamp
    add_timestamp = False  # Set to True to add a timestamp column
    combine_files = True  # Set to True to combine files
    
    # remove the row with the same timestamp, only keep the first row
    if remove_row: 
        output_file=process_time_data(original_name,original_path)
    
    #  add a timestamp column to the file, the difference is calculated from the first row
    elif add_timestamp:
        output_file =timestamp_csv_file(original_name, original_path)
    
    # combine two files, the first file is the base file, the second file is the source file. 
    # Leave the empty values as NaN    
    elif combine_files:
        file1 = 'gun_voltage_12222020_combined.csv'
        file2 = 'gauge_pressure_12222020.csv'
        col_indices_str = "1"  # Example column indices to copy from file2
        output_file = combine(file1, file2, col_indices_str,file_path=original_path)
    
    print(f"Original file: {original_name}")
    print(f"Processed data saved to: {output_file}")

    # Optionally, you can load and display the head of the new file to verify
    df_output = pd.read_csv(output_file, parse_dates=[0])
    print("\nFirst 5 rows of the new _hz file:")
    print(df_output.head().to_markdown(index=False, numalign="left", stralign="left"))