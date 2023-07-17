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
    predictions_df = pd.DataFrame()
    intervals_df = pd.DataFrame()

    for column_name in models.keys():
        model = models[column_name]
        scaler = scalers[column_name]

        input_data_scaled = scaler.transform(input_data)

        predictions = model.predict(input_data_scaled)

        predictions_df[column_name] = predictions

        if isinstance(model, RandomForestRegressor):
            # Check if 'test_score' is available for this model
            if column_name in results_df['column'].values:
                mae = results_df.loc[results_df['column'] == column_name, 'test_score'].values[0]
                lower_bound = predictions - mae
                upper_bound = predictions + mae
            else:
                lower_bound = upper_bound = [np.nan] * len(predictions)

            intervals_df[column_name] = list(zip(lower_bound, upper_bound))
        else:
            intervals_df[column_name] = [(np.nan, np.nan)] * len(predictions)

    return predictions_df, intervals_df

# Create Streamlit inputs
st.title('My Prediction App')

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
