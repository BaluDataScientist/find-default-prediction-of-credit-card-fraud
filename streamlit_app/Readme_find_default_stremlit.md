Credit Card Fraud Detection App
This Streamlit app allows users to detect fraudulent credit card transactions using various machine learning models. The app provides an interface to upload transaction data, select a model, and predict the likelihood of fraud.

Features
Model Selection: Choose from multiple machine learning models to predict fraud.
Upload CSV Files: Users can upload their own CSV files containing transaction data.
Prediction Results: The app generates a file with predictions, dynamically named based on the selected model.

How to Use
Clone the Repository:
git clone https://github.com/your-username/your-repository.git

Navigate to the Project Directory:
cd your-repository
Install Dependencies: Ensure you have all required dependencies installed. You can do this by running:
pip install -r requirements.txt

Run the App: Launch the Streamlit app by running:
streamlit run app.py

Interact with the App:
Upload a CSV file containing credit card transaction data.
Select the desired machine learning model from the sidebar.
Click the "Predict" button to generate predictions.
Download the output file, named predictions_{model_name}.csv, containing the predictions.

Project Structure

app.py: The main Streamlit app script.

models/: Contains trained models and metrics.
data/: Example data files (if any).
notebooks/: Jupyter notebooks used for model training and evaluation.
requirements.txt: List of Python dependencies required to run the app.

Model Overview
The app supports the following machine learning models:
Logistic Regression
Random Forest
Decision Tree
XGBoost
LightGBM
Gaussian Naive Bayes
K Nearest Neighbour (KNN)
Support Vector Machine Classifier (SVM)

Each model has been trained and fine-tuned to detect fraudulent transactions with high accuracy.

License
This project is licensed under the MIT License. See the LICENSE file for more details.

Acknowledgments
Streamlit for providing an easy-to-use platform for deploying machine learning models.
upgrad for the dataset.