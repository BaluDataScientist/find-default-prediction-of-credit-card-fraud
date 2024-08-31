import pandas as pd
import joblib
import numpy as np
from scipy import stats
from sklearn.preprocessing import OrdinalEncoder

def preprocess_data(data, is_dataframe=False):
    if not is_dataframe:
        # Load the data from CSV
        data_frame = pd.read_csv(data)
    else:
        # Assume data is already a DataFrame
        data_frame = data.copy()
    
    # Cast 'Time' to integer
    data_frame['Time'] = data_frame['Time'].astype(int)
    
    # Reshape and encode the 'Time' feature
    time_encoded = OrdinalEncoder().fit_transform(data_frame[['Time']])
    data_frame['Time_Encoded'] = time_encoded
    
    # Drop the original 'Time' column
    data_frame = data_frame.drop(columns=['Time'])
    
    # Load the scaler
    scaler = joblib.load('models/amount_scaler.joblib')
    
    # Scale the 'Amount' column and apply Box-Cox transformation
    data_frame['Amount_Scaled'] = scaler.transform(data_frame[['Amount']])
    
    # Apply Box-Cox transformation to the 'Amount' column with the given lambda value
    boxcox_lambda = -0.04497254555023551
    data_frame['Amount_BoxCox'] = stats.boxcox(data_frame['Amount'] + 1, lmbda=boxcox_lambda)
    
    # Drop the 'Amount' and 'Amount_Scaled' columns
    data_frame = data_frame.drop(columns=['Amount', 'Amount_Scaled'])
    
    return data_frame

# If you want to load a csv file mark has "is_dataframe=False" if the data is a dataframe mark has "is_dataframe=True"