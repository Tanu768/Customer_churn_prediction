📌 Customer Churn Prediction
This project focuses on predicting whether a customer is likely to stop using a company’s service (customer churn) using machine learning. The trained model is integrated into a simple application for real-time prediction.
🧠 Project Overview
Customer churn occurs when customers discontinue a service. Predicting churn in advance helps businesses take preventive actions such as offering discounts or improving customer support.
In this project, customer data is analyzed and a machine learning classification model is built to predict churn. The final model is saved and used in an application for making predictions on new customer data.
📁 Project Structure
Copy code

Customer_churn_prediction
│
├── app.py                     # Application file for prediction
├── notebook.ipynb             # Data analysis, preprocessing & model training
├── customer_churn_data.csv    # Dataset
├── model.pkl                  # Trained machine learning model
├── scaler.pkl                 # Scaler used for feature scaling
└── README.md
📊 Dataset
File: customer_churn_data.csv
Type: Tabular customer data
Target Variable: Churn
1 → Customer churned
0 → Customer did not churn
The dataset contains customer-related information such as demographics, service usage, and billing details.
🛠️ Technologies Used
Python
Libraries:
NumPy
Pandas
Matplotlib
Seaborn
Scikit-learn
Model Saving: Pickle / Joblib
Application: Python-based app (app.py)
Notebook: Jupyter Notebook
⚙️ Data Preprocessing
The following preprocessing steps were performed:
Handling missing values
Encoding categorical variables
Feature scaling using StandardScaler
Splitting data into training and testing sets
The trained scaler is saved as scaler.pkl and reused during prediction.
🤖 Machine Learning Model
Problem Type: Binary Classification
Models used:
Logistic Regression
Random Forest Classifier
The final model was selected based on performance on test data and saved as model.pkl.
📈 Model Evaluation
The model was evaluated using:
Accuracy
Precision
Recall
F1-Score
Confusion Matrix
Since churn data is imbalanced, more importance was given to Recall and F1-Score to correctly identify churn customers.
🚀 How to Run the Project
1️⃣ Clone the repository
Copy code
Bash
git clone https://github.com/Tanu768/Customer_churn_prediction.git
cd Customer_churn_prediction
2️⃣ Install required libraries
Copy code
Bash
pip install pandas numpy scikit-learn matplotlib seaborn
3️⃣ Run the notebook
Open and execute:
Copy code

notebook.ipynb
4️⃣ Run the application
Copy code
Bash
python app.py
🖥️ Application Usage
Enter customer details in the application
Click on Predict
The model predicts whether the customer is likely to churn or not
💼 Business Impact
This project helps businesses to:
Identify customers who are at risk of churning
Take early retention actions
Reduce customer loss and revenue decline
Improve customer satisfaction
🔮 Future Enhancements
Hyperparameter tuning for better performance
Use advanced models like XGBoost or LightGBM
Add model explainability
Deploy as a web API
👩‍💻 Author
Tanu Yadav
B.Tech Computer Science Engineering Student
Interested in Data Science & Machine Learning
GitHub: https://github.com/Tanu768
