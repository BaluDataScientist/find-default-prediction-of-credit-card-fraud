
## find-default (prediction-of-credit-card-fraud)

A credit card is one of the most used financial products to make online purchases and payments. Though the Credit cards can be a convenient way to manage your finances, they can also be risky. Credit card fraud is the unauthorized use of someone else's credit card or credit card information to make purchases or withdraw cash. It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase.

The dataset contains transactions made by credit cards in September 2013 by European cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. 

The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.

## Business understanding: 

In the current world, lot of new technologies are coming daily. Banking Transaction have evolved a lot from paper works to digital transactions and to card transactions. As a result, the number of transactions has increased multifold, while cash and cheque transactions have drastically decreased, putting a significant load on banks' servers for daily transactions. Customers leave behind a lot of digital footprints, and in some cases, these are compromised, leading to confidential information falling into the wrong hands. But the onus of safety and security of customer's money is on the Banks. Due to high volume transactions, its impractical to be manually monitored. Hence the Banks need several models to monitor the transactions and raise alarm when something is wrong. 

One such model requirement is to identify credit card transaction fraud and raise alarm as quickly as possible. So that neither the customer, POS vendors, nor the bank loses money. Hence the idea here is creating a model, using the features available of the customer and his transactions, so that model can predict the fraud transactions quickly in real time and shortlist the transaction for a manual intervention to cross check with card holder immediately.

Since there is monetary value associated with every transaction and missing one fraud transaction shall end up has a loss to the Bank. It is understood that the False Negative is more dangerous than False Positive. Has false Negative will lead to financial losses but false Positive shall lead customer inconvenience. I'm sure later is much better to handle. 

So, we need to build a strong model which as high accuracy in identifying a fraud transaction more accurately in the shortest time.
## Project Folder Structure:

1. data/raw: Contains the original dataset (creditcard.csv).

2. data/processed-data: Contains processed datasets used in the notebooks.

3. models: Contains trained models and the final hyperparameter-tuned model for deployment.

4. visuals: Contains all figures and images generated during the project.

5. requirements.txt: Lists all dependencies required to run the project.

6. README.md: This file, explaining the project structure and usage.

### Key Files and Directories

#### data: 
The data/processed-data folder contains the processed files such as X_train, y_train, etc., which are loaded at the beginning of the second and third notebooks.

#### models: 
The models folder contains trained models (pre & post hyperparameter tuning) and the Json file containing metrics of every model. The scaler file amount_scaler.joblib is used to fit the training data is saved here.

#### notebooks:
The notebooks folder contains 3 Notebooks.

Notebook 1: data-exploration-and-preprocessing.ipynb
Notebook 2: model-training-evaluation.ipynb
Notebook 3: Hyperparameter-tuning-final-model-evaluation.ipynb

#### Visuals: 
The visuals folder contains .png files of plots and figures generated during EDA and model evaluation.


## Installation Instruction

### Prerequisites

1. Python 3.6 or higher

2. Jupyter Notebook or Jupyter Lab

3. Libraries listed in the requirements.txt file

### Installation

1.	Clone this repository.
2.	Navigate to the project directory.
3.	Install the required dependencies using the following command:

    ‘’’pip install -r requirements.txt’’’

### How to Run the Project & use

1. Ensure that the dataset is located in the data/raw folder.

2. Open the Jupyter notebooks in the following sequence:

	Notebook 1: data-exploration-and-preprocessing.ipynb

	Notebook 2: model-training-evaluation. ipynb

	Notebook 3: Hyperparameter-tuning-final-model-evaluation.ipynb

3. Run the cells in each notebook to replicate the results.

## Project Steps and Reasoning

### Data Exploration and Understanding:

Exploratory Data Analysis (EDA): Initial exploration of data to understand distributions, correlations, and outliers.

### Data Preprocessing: 

The Time column's data type is marked as float, but it should be an integer. The same is corrected.

Since the features V1 - V28 are PCA ((Principal Component Analysis)-transformed, we don't need to treat any outliers separately for these features.

Check the class in Target variable and check relation with the time column to see if there is a pattern. we see most of FRAUD transactions are around low transaction value. And there is no pattern with time. So, we can drop the time column.  

'Fraud_CumSum' and 'NonFraud_Cumsum' columns are used for only in EDA to understand time and amount. Hence it is dropped to avoid complex issues in deployment or test data prep


#### Split the Data 

We are splitting the data into 80:20 ratio randomly.  80% is used in training and the rest 20% is used in testing. 

Scale the data - Since all columns are PCA transformed, we can scale the left out Amount column using the Standard Scaler. Saved the scaler file using joblib for future use in test and deployment. 

After checking the transformation of the scaled Amount column, we found that it was skewed. Therefore, we applied a Box-Cox transformation only to the Amount column, as the other columns are already PCA-transformed.




#### ENCODING & SCALING & TRANFORMATION

We have done a detailed EDA & pre-processed the split data. Encoding done after split to avoid data leak.

It's clear that all features except for Time, Amount and Class are already PCA Transformed. We have to work on these 3 columns to make them uniform.

1. Let’s encode the Time column with Ordinal Encoder since the data in the column has some order.

2. Amount column is a continuous data lets transform it after splitting to avoid data leakage to model.

3. Class column has binary values, hence no need to encode or transform

#### Final Features selection for training

We have done following steps on the X_train data frame to prepare it for training the models. Above encoding, scaling, transformation executes as planned. 

After Ordinal encoding of Time column new column Time_Encoded is created and hence duplicate Time column is dropped from X_train. 

We have scaled the amount column after splitting and a new column Amount_Scaled is created. We have found the data is skewed by using few plots and hence transformed the Amount column using BoxCox Transformation as this is best for imbalanced data and it will help in getting a symmetrical structure to the data. After Transforming Amount_scaled, a new column Amount_BoxCox was created with transformed values. 
To avoid duplication, dropped Amount, Amount_Scaled.

Below Features are kept ready for training the model.

Index(['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11',
       'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21',
'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Time_Encoded',
'Amount_BoxCox'],
dtype='object')


#### Created Pipeline for preparing test data

Use the enclosed function data_preparation.py file to clean the test data or in deployment to prepare the test data. 

Once this function is saved. You can call the function 
X_test = preprocess_data(X_test, is_dataframe=True)

Where X_test is the test dataframe of test value created in splitting stage. If you want to load a csv file mark has "is_dataframe=False" and if the data is a dataframe mark has "is_dataframe=True". Usually in jupyter notebook a dataframe is feeded and in external platform it may be otherwise. Hence both options included in the function.

#### Create Undersampling and Oversampling Object

Now that the data is more consistent after all cleaning and transformation lets create objects of Undersampling and oversampling(SMOTE) techniques.

#### Save the processed data and objects

Since the data processing is complete. A final processed data file is saved in the data folder for future reference as required in the guidelines. Also objects are saved as a joblib file for quick reference in future. The list is as below.

joblib.dump(X_train, 'data/processed-data/X_train_v1.joblib')
joblib.dump(y_train, 'data/processed-data/y_train_v1.joblib')
joblib.dump(X_test, 'data/processed-data/X_test_v1.joblib')
joblib.dump(y_test, 'data/processed-data/y_test_v1.joblib')
joblib.dump(y_test, 'models/amount_scaler.joblib')
joblib.dump(X_train_resampled, 'data/processed-data/X_train_resampled_v1.joblib')
joblib.dump(y_train_resampled, 'data/processed-data/y_train_resampled_v1.joblib')
joblib.dump(X_train_smote, 'data/processed-data/X_train_smote_v1.joblib')
joblib.dump(y_train_smote, 'data/processed-data/y_train_smote_v1.joblib')

## 2.Model Training and Evaluation:

Baseline Models: Logistic Regression, Gaussian Naive Bayes, Decision Trees, Random Forest, Suppot Vector Machine (classifier), KNN - K Nearest Neighbour, Light GBM and XG Boost are initially used to establish a baseline. Trained models are saved as a Joblib file in the models folder with clear nomenclature to identify. KNN algorithm has some problem with pandas frame, hence only for this algorithm test data is changed into a smoothened numpy array to train the model. 

Resampling Techniques: Undersampling, oversampling using SMOTE are applied to address class imbalance. Trained models under each algorith and sampling technique is saved with clear nomenclature as joblib file for future reference.

Model Evaluation: Models are evaluated using accuracy, F1 score, and ROC AUC. Every models evaluation matrix is saved as a Json file for consolidation in the later stage.







### Evaluation

1. As discussed earlier, its clearly evident that Accuracy metric is showing high performance in most of the models. F1 Score shows a slight difference in the performance metrics. But ROC_AUC clearly distinguishes the model performance more clearly as expected. It displays the strength of ROC_AUC in imbalanced data performance evaluation.

Top 5 Models as per metrics :
1. Accuracy : Xgb_rus, Xgb_base, rf_base, Xgb_smote, dt_smote.
2. F1_score : Xgb_rus, Xgb_base, rf_base, Xgb_smote, dt_base.
3. Roc_Auc : lgbm_rus, Xgb_smote, xgb_rux, xgb_base, Lgbm_smote


Its clear that best results across all Algorithms are from XGB and LGBM, Random Forest. In Accuracy & F1_score we can see rf_base, dt_base are placed in Top 5 but dont feature in the top 5 of  ROC_AUC metric. 

I wanted give more weightage to ROC_AUC metric since it is very robust in problems with imbalanced data. Hence I'm proceeding with Hyper parameter Tuning or Cross-validation for 3 algorithms(XGB,LGBM,RF). Lets check if this improves the performance or help in understanding important hyperparameter.

### Hyperparameter Tuning and Final Model Evaluation:

As observed, XGBoost, RF & LBGB models have performed well on test data we will choose these 3 models to hyper parameter tuning using GridSearchCV to get the best possible params.

Based on the best params lets build a final best model under XGBoost & LightGBM, so that we can deploy one of them as for deployment.

#### GridSearchCV: 
Used to find the best hyperparameters for models like XGBoost, RF and LightGBM. Using the best hyperparameter new XGBoost model was trained.

#### COMPARE XGBoost Base Model & XGBoost Best Model

The first grid search found these hyperparameters for XGBoost:
Best parameters for XGBoost: {'learning_rate': 0.2, 'max_depth': 3, 'n_estimators': 300, 'subsample': 0.7}

If you observe the best model metrics, we can see a very small improvement in Accuracy and F1Score values of XGB_best Model compared with the XGB_base Model. However, my main goal of improving the ROC AUC was not achieved. In fact, the best model underperformed slightly. But when compared with top Roc_AUC of XGB_smote value of 0.987574. 

It is clear the current result is Low. Though the difference in performance of the model is very negligible I'm still trying if the metrics can improve.

It is not a good idea to implement a XGB_Smote in production since it has worked on artifically created data. So trying to get best parameters to improve Roc_Auc performance of base model. So tuning some of the GridSearchCV values to see if we can get better parameters. 

#### Run a XGBoost GridSearchCV for the 2nd time with slight modification to the parameter and found below best parameter. 

Second Grid search on XGBoost model gave these best Hyper parameters has {'learning_rate': 0.15, 'max_depth': 7, 'n_estimators': 400, 'subsample': 0.7}

New XGB_best1 evaluation metrics values are comparatively outperformed across all metrics of previous version of XGB_best and Original Model XGB_base. 

#### 2. Hyperparamete Tuning – RandomizedsearchCV – Random Forest

RandomizedsearchCV was done on Random Forest Model and got below Best Hyper parameters for RF : {'bootstrap': True, 'max_depth': 15, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 75}

#### 3. Hyperparameter Tuning - GridsearchCV - LightGBM

GridseachCV was done on LightGBM Model and got below
Best Hyperparameters for LightGBM:  

{'learning_rate': 0.01, 'max_depth': 20, 'min_child_samples': 50, 'n_estimators': 100, 'num_leaves': 31, 'subsample': 0.8}

Hyper parameter tuned model lgbm_best model has perfomed better than the base model lgbm_base across all 3 metrics (accuracy, f1Score, Roc_auc). So further no tuning required.

Lets compare all the model metrics and decide on the best model for production.

## Final Model Selection: 

The best-performing model is selected based on a comparison of metrics.

Final Model: Xgb_best1 was chosen for production based on superior performance.

### Cross Validtaion Score

Below are the Cross Validation Score of 3 hyper parameter tuned models. This clearly shows the generalization of the model. Standard Deviation is low, it’s a strong indication of generalization. 





### Results

Based on the above summary of metrics of the top model. We can clearly see that XGB Algorithm model Xgb_best1 has performed well across all evaluation matrix and the low Standard Deviation of the model clearly indicates that the model has very well generalized. It also indicates that the model is less likely to overfit and will perform very well on unseen data. 

Though the difference in the metrics is minimal, we can clearly say that this model is the optimal choice for deployment. This model is expected to effectively detect fraudulent transactions and minimize both false positives and false negatives. 

XGBoost model (XGB_BEST1) is an optimal choice for better prediction capacity, but lets remember that this models needs more Resource & time to train and infer.  Considering the perfection trade off with resources is well justified.  But if the time and resources are a concern, the next best model to choose will be LGBM_Best which is quite fast in computation and in handling Large dataset, specially sparse dataset. 

### Feature Importance

Based on the above metrics and analysis we have found that the Model “xgb_best1” has the perfect model in the given situation. Below are the important Features which has influenced the model. Same is plotted in chronology to visualize the important feature in the dataset.

## Future Work

1. Though the model performance is good. We need to fit few of the deep learning models to check if it can improve performance. 
2. We should continuously improve our model by reviewing the prediction and re-training the model wherever possible. 
3. Every day fraud transactions are changing and our approach to identify the same also has to be upgraded based on regular learning. 
4. We need to incorporate new data and improve the model at regular interval.


