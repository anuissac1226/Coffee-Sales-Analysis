import streamlit as st
import pandas as pd
import pickle

def load_assets():
    with open('LinearRegression_model_for_rating_prediction.pkl', 'rb') as file:
        rating_model = pickle.load(file) #loading the best model
    with open('scaler_rating.pkl', 'rb') as file:
        rating_scaler = pickle.load(file) #loading the std scalar
    with open('encoder.pkl', 'rb') as file:
        encoder = pickle.load(file) #loading the encoder
    with open('decision_tree_classifier_for_popularity.pkl', 'rb') as file:
        popularity_model = pickle.load(file)  #loading the popularity prediction model
    with open('scaler_popularity.pkl', 'rb') as file:
        popularity_scaler = pickle.load(file)
    return rating_model, rating_scaler, encoder, popularity_model, popularity_scaler

rating_model, rating_scaler, encoder, popularity_model, popularity_scaler = load_assets()

# Loading dataset
def load_dataset():
    return pd.read_csv('data/merged_df.csv')

df = load_dataset()

# Streamlit app
st.title('Rating and Popularity Prediction')
# User Input
#Prediction Type
prediction_type = st.radio("Select the type of prediction:", ('Rating Prediction', 'Popularity Prediction'))

#Dropdown for roast type
roast_types = [
    'roast_dark',
    'roast_light',
    'roast_medium',
    'roast_medium_dark',
    'roast_medium_light',
    'roast_very_dark'
]
selected_roast = st.selectbox("Select Roast Type:", roast_types)

#Create a dictionary for roast type
roast_encoding = {roast: 0 for roast in roast_types}
roast_encoding[selected_roast] = 1  #setting selected roast type as 1

#Dropdown for region
regions = [
    'region_africa_arabia',
    'region_caribbean',
    'region_central_america',
    'region_hawaii',
    'region_asia_pacific',
    'region_south_america'
]
selected_region = st.selectbox("Select Region:", regions)

#Create a dictionary for region
region_encoding = {region: 0 for region in regions}
region_encoding[selected_region] = 1  #setitng selected region as 1

#Input for other numeric fields
aroma = st.sidebar.slider("Aroma (1-10):", min_value=1, max_value=10, value=5)
flavor = st.sidebar.slider("Flavor (1-10):", min_value=1, max_value=10, value=5)
aftertaste = st.sidebar.slider("Aftertaste (1-10):", min_value=1, max_value=10, value=5)
acid_or_milk = st.sidebar.slider("Acidity/Milkiness (1-10):", min_value=1, max_value=10, value=5)
body = st.sidebar.slider("Body (1-10):", min_value=1, max_value=10, value=5)

#Dropdown for roaster
roaster_options = df['roaster'].unique()
selected_roaster = st.selectbox("Select Roaster:", roaster_options)

#Combine input values into a dataframe
input_data = pd.DataFrame({
    **roast_encoding,#dictionary unpacking
    **region_encoding,
    'aroma': [aroma],
    'flavor': [flavor],
    'aftertaste': [aftertaste],
    'acid_or_milk': [acid_or_milk],
    'body': [body],
    'roaster': [selected_roaster]
})

#one hot encoding for 'roaster' column
encoded_roaster = encoder.transform(input_data[['roaster']])
encoded_roaster_df = pd.DataFrame(
    encoded_roaster.toarray(),
    columns=encoder.get_feature_names_out(['roaster'])
)

#Drop the original 'roaster' column and concatenate encoded roaster values
input_data = input_data.drop(columns=['roaster']).reset_index(drop=True)
input_data = pd.concat([input_data, encoded_roaster_df], axis=1)

#standard scaling
input_data_scaled_for_rating = rating_scaler.transform(input_data)
input_data_scaled_for_popularity = popularity_scaler.transform(input_data)

# Predict button
if prediction_type == 'Rating Prediction':
    st.subheader("Rating Prediction")
    if st.button("Predict Rating"):
        prediction = rating_model.predict(input_data_scaled_for_rating)[0]
        st.success(f"The predicted rating is {prediction:.2f}")
elif prediction_type == 'Popularity Prediction':
    st.subheader("Popularity Prediction")
    if st.button("Predict Popularity"):
        popularity_prediction = popularity_model.predict(input_data_scaled_for_popularity)[0]
        st.success(f"This product is {popularity_prediction}")