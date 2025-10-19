from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix, roc_auc_score
import pickle
import pandas as pd
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt

# Load Dataset
merged_df = pd.read_csv('data/merged_df.csv')

# Feature and target columns
features = ['roast_dark', 'roast_light', 'roast_medium', 'roast_medium_dark',
            'roast_medium_light', 'roast_very_dark',
            'region_africa_arabia', 'region_caribbean', 'region_central_america',
            'region_hawaii', 'region_asia_pacific', 'region_south_america',
            'aroma', 'flavor', 'aftertaste', 'acid_or_milk', 'body', 'roaster']
target_rating = 'normalized_rating'
target_popularity = 'popularity_tier'

df_rating = merged_df[features + [target_rating]]
df_popularity = merged_df[features + [target_popularity]]

#Split data into features and target for predicting rating and popularity
X = df_rating[features]
Y_rating = df_rating[target_rating]
Y_popularity = df_popularity[target_popularity]

#One Hot Encoding for 'roaster'
encoder = OneHotEncoder(drop='first')
categorical_columns = ['roaster']
encoded_columns = encoder.fit_transform(X[categorical_columns])
#Create a DataFrame with encoded features
encoded_df = pd.DataFrame(encoded_columns.toarray(), columns=encoder.get_feature_names_out(categorical_columns))
#Drop the original roaster column and concatenate the encoded features
X_encoded = X.drop(columns=categorical_columns).reset_index(drop=True)
X_encoded = pd.concat([X_encoded, encoded_df], axis=1)

#Test Train split for rating
X_train_rating, X_test_rating, Y_train_rating, Y_test_rating = train_test_split(X_encoded, Y_rating, test_size=0.2, random_state=42)
# Stratified split for popularity
X_train_popularity, X_test_popularity, Y_train_popularity, Y_test_popularity = train_test_split(
    X_encoded, Y_popularity, test_size=0.2, stratify=Y_popularity, random_state=42
)

# Standardizing the features
scaler = StandardScaler()
X_train_rating_scaled = scaler.fit_transform(X_train_rating)
X_test_rating_scaled = scaler.transform(X_test_rating)

scaler1 = StandardScaler()
X_train_popularity_scaled = scaler1.fit_transform(X_train_popularity)
X_test_popularity_scaled = scaler1.transform(X_test_popularity)

#Applying resampling(SMOTE) to the training data
smote = SMOTE(random_state=42)
X_resampled, Y_resampled = smote.fit_resample(X_train_popularity_scaled, Y_train_popularity)

#Save the encoder and scaler
with open('encoder.pkl', 'wb') as file:
    pickle.dump(encoder, file)
print("encoder saved successfully!!!")
with open('scaler_rating.pkl', 'wb') as file:
    pickle.dump(scaler, file)
print("Scaler saved successfully!!!")
with open('scaler_popularity.pkl', 'wb') as file:
    pickle.dump(scaler1, file)
print("Scaler1 saved successfully!!!")

models_rating = {
    'LinearRegression' :  LinearRegression(),
    'knn' : KNeighborsRegressor(n_neighbors=5),
    'decision_tree': DecisionTreeRegressor(max_depth=5,min_samples_split=10,min_samples_leaf=2,random_state=42),
    'random_forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

for name,model in models_rating.items():
    model.fit(X_train_rating_scaled,Y_train_rating)
    Y_pred_rating=model.predict(X_test_rating_scaled)
    mse=mean_squared_error(Y_test_rating,Y_pred_rating)
    r2=r2_score(Y_test_rating,Y_pred_rating)
    print(f'{name}-mean squared error: {mse}')
    print(f'{name}-R2: {r2}')

#Save trained model for rating prediction
for name, model in models_rating.items():
    filename = f'{name}_model_for_rating_prediction.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"{name} saved successfully as {filename}")


#Train Decision Tree model for popularity classification
model_popularity = DecisionTreeClassifier(max_depth=10,min_samples_split=10,min_samples_leaf=2,random_state=42, class_weight='balanced')
model_popularity.fit(X_resampled, Y_resampled)
Y_pred_popularity = model_popularity.predict(X_test_popularity_scaled)

#Model evaluation
print("Classification Report:")
print(classification_report(Y_test_popularity, Y_pred_popularity))

# ROC-AUC Score
roc_auc = roc_auc_score(Y_test_popularity, model_popularity.predict_proba(X_test_popularity_scaled), multi_class='ovr')
print(f"ROC-AUC Score: {roc_auc}")

#Confusion matrix
cm = confusion_matrix(Y_test_popularity, Y_pred_popularity, labels=['Highly Popular', 'Moderately Popular', 'Less Popular'])
print(f"confusion matrix: {cm}")

#Plot confusion matrix using heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrBr', xticklabels=['Highly Popular', 'Moderately Popular', 'Less Popular'], yticklabels=['Highly Popular', 'Moderately Popular', 'Less Popular'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Popularity Prediction')
plt.show()

#Save the trained model for popularity prediction
with open('decision_tree_classifier_for_popularity.pkl', 'wb') as file:
    pickle.dump(model_popularity, file)
print("Model for popularity saved successfully!")