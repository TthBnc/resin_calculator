import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor

# Load the models, scalers, and results_df
with open('class_reg_models.pkl', 'rb') as f:
    models = pickle.load(f)
with open('class_reg_scalers.pkl', 'rb') as f:
    scalers = pickle.load(f)
results_df = pd.read_pickle('class_reg_results_df.pkl')

# Function to predict with all models
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

def predict_with_all_models(models, scalers, input_data, results_df):
    predictions_list = []
    intervals_list = []

    for column_name in models.keys():
        model = models[column_name]
        scaler = scalers[column_name]

        input_data_scaled = scaler.transform(input_data)

        predictions = model.predict(input_data_scaled)

        predictions_list.append({'Name': column_name, 'Value': predictions[0]})

        # Check if the model is a regression model
        if isinstance(model, RandomForestRegressor):
            # Get the MAE for this model
            mae = results_df.loc[results_df['column'] == column_name, 'test_score'].values[0]

            # Compute the lower and upper bound of the predictions
            lower_bound = predictions - mae
            upper_bound = predictions + mae

            # Add the intervals to the DataFrame
            intervals_list.append({'Name': column_name, 'Lower': lower_bound[0], 'Upper': upper_bound[0]})

    predictions_df = pd.DataFrame(predictions_list)
    intervals_df = pd.DataFrame(intervals_list)

    return predictions_df, intervals_df

# Create Streamlit inputs
st.title('Resin calculator')

fazekido = st.number_input('Fazékidő (min):')
szakitoszilardsag = st.number_input('Szakítószilárdság [MPa]:')
szakadasinyulas = st.number_input('Szakadási nyúlás [%]:')

# Create DataFrame from inputs
input_data = pd.DataFrame([[fazekido, szakitoszilardsag, szakadasinyulas]])

if st.button('Calculate'):
    # Make predictions
    predictions_df, intervals_df = predict_with_all_models(models, scalers, input_data, results_df)
    
    # Display predictions
    st.write('Predictions:')
    st.dataframe(predictions_df)
    st.dataframe(intervals_df)
