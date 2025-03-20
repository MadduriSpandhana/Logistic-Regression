#!/usr/bin/env python
# coding: utf-8

# In[57]:


get_ipython().system('pip install pandas numpy seaborn matplotlib scikit-learn streamlit')

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
df1 = pd.read_csv("Titanic_test.csv")
df1


# In[58]:


df2 = pd.read_csv("Titanic_train.csv")
df2


# In[59]:


# Display summary statistics
print(df1.describe())


# In[60]:


print(df2.describe())


# In[61]:


df1.info()


# In[62]:


df2.info()


# In[63]:


# Make a copy to process (so the original stays unchanged)
df1_cleansed = df1.copy
df2_cleansed = df2.copy


# In[64]:


# Drop unnecessary columns
df1_cleansed = df1.drop(columns=['Cabin', 'Name', 'Ticket','PassengerId'])


# In[65]:


df2_cleansed = df2.drop(columns=['Cabin', 'Name', 'Ticket','PassengerId'])


# In[66]:


# Check for missing values
print(df1_cleansed.isnull().sum()) 


# In[67]:


print(df2_cleansed.isnull().sum()) 


# In[68]:


# Missing values: age and fare in df1
# age and embarked in df2


# In[69]:


#Based on skewness we can handle missing values:
df1_cleansed.hist(figsize=(10, 10))
plt.show()


# In[70]:


df2_cleansed.hist(figsize=(10, 10))
plt.show()


# In[71]:


df1_cleansed['Age'].fillna(df1['Age'].median(), inplace=True)


# In[72]:


df1_cleansed['Fare'].fillna(df1['Age'].median(), inplace=True)


# In[73]:


df2_cleansed['Age'].fillna(df1['Age'].median(), inplace=True)


# In[74]:


df2_cleansed['Embarked'].fillna(df1['Embarked'].mode()[0], inplace=True)


# In[75]:


print(df1_cleansed.isnull().sum()) #1 and 6 categorical


# In[76]:


print(df2_cleansed.isnull().sum()) #2 and 7 categorical


# In[77]:


# Create boxplots to identify outliers
df1_cleansed.boxplot(figsize=(12, 10))
plt.show()


# In[78]:


# Create boxplots to identify outliers
df2_cleansed.boxplot(figsize=(12, 10))
plt.show()


# In[79]:


# Define function to detect outliers and store them separately
def treat_and_store_outliers1(df1_cleansed, columns):
    outliers_df_1 = pd.DataFrame()  # DataFrame to store outliers
    treated_df_1 = df1_cleansed.copy()  # Copy of original dataframe for modifications
    
    for column in columns:
        Q1 = df1_cleansed[column].quantile(0.25)
        Q3 = df1_cleansed[column].quantile(0.75)
        IQR1 = Q3 - Q1
        
        # Define bounds
        lower_bound = Q1 - 1.5 * IQR1
        upper_bound = Q3 + 1.5 * IQR1
        
        # Store outliers separately
        outliers_1 = df1_cleansed[(df1_cleansed[column] < lower_bound) | (df1_cleansed[column] > upper_bound)]
        outliers_df_1 = pd.concat([outliers_df_1, outliers_1])
        
        # Replace outliers with median
        median_value = df1_cleansed[column].median()
        treated_df_1[column] = np.where((df1_cleansed[column] < lower_bound) | (df1_cleansed[column] > upper_bound), 
                                      median_value, 
                                      df1_cleansed[column])
    
    return treated_df_1, outliers_df_1

#Identifying the numerical columns
df_num_1=df1_cleansed.select_dtypes(include=np.number)
df_num_1


# In[80]:


# Process outliers
cleaned_df_1, outliers_df_1 = treat_and_store_outliers1(df1_cleansed,df_num_1 )

# Save treated and outliers data for further analysis
cleaned_df_1.to_csv("cleaned_data1.csv", index=False)
outliers_df_1.to_csv("outliers_data.csv", index=False)

# Display results
print("Outliers stored in 'outliers_data.csv'")
print("Cleaned data stored in 'cleaned_data1.csv'")


# In[81]:


df1__=pd.read_csv("cleaned_data1.csv")
df1__


# In[82]:


def treat_and_store_outliers2(df2_cleansed, columns):
    outliers_df_2 = pd.DataFrame()  # DataFrame to store outliers
    treated_df_2= df2_cleansed.copy()  # Copy of original dataframe for modifications
    
    for column in columns:
        Q1 = df2_cleansed[column].quantile(0.25)
        Q3 = df2_cleansed[column].quantile(0.75)
        IQR2 = Q3 - Q1
        
        # Define bounds
        lower_bound = Q1 - 1.5 * IQR2
        upper_bound = Q3 + 1.5 * IQR2
        
        # Store outliers separately
        outliers_2 = df2_cleansed[(df2_cleansed[column] < lower_bound) | (df2_cleansed[column] > upper_bound)]
        outliers_df_2 = pd.concat([outliers_df_2, outliers_2])
        
        # Replace outliers with median
        median_value = df2_cleansed[column].median()
        treated_df_2[column] = np.where((df2_cleansed[column] < lower_bound) | (df2_cleansed[column] > upper_bound), 
                                      median_value, 
                                      df2_cleansed[column])
    
    return treated_df_2, outliers_df_2

#Identifying the numerical columns
df_num_2=df2_cleansed.select_dtypes(include=np.number)
df_num_2


# In[83]:


# Process outliers
cleaned_df_2, outliers_df_2 = treat_and_store_outliers2(df2_cleansed,df_num_2)

# Save treated and outliers data for further analysis
cleaned_df_2.to_csv("cleaned_data2.csv", index=False)
outliers_df_2.to_csv("outliers_data.csv", index=False)

# Display results
print("Outliers stored in 'outliers_data.csv'")
print("Cleaned data stored in 'cleaned_data2.csv'")


# In[84]:


df2__=pd.read_csv("cleaned_data2.csv")
df2__


# In[85]:


#The data frame is mostly skewed so using normalization.
from sklearn.preprocessing import MinMaxScaler
MM = MinMaxScaler()
MM_X1 = MM.fit_transform(df_num_1)
MM_X1 = pd.DataFrame(MM_X1)
MM_X1.columns = ["Pclass","Age","SibSp","Parch","Fare"]
MM_X1#here values lie btw 0 to 1


# In[172]:


MM_X2 = MM.fit_transform(df_num_2)
MM_X2 = pd.DataFrame(MM_X2)
MM_X2.columns = ["Survived","Pclass","Age","SibSp","Parch","Fare"]
MM_X2#here values lie btw 0 to 1


# In[174]:


# OneHotencoding
from sklearn.preprocessing import OneHotEncoder
import pandas as pd  

# Initialize OneHotEncoder without dropping any category
OHE = OneHotEncoder(sparse_output=False)  # Ensures dense output

# Transform 'Sex' column
df_sex1 = pd.DataFrame(OHE.fit_transform(df1__[['Sex']]), 
                        columns=['female', 'male'],  # Correct column names
                        index=df1__.index)  # Keep index aligned

# Drop original "Sex" column & concatenate new columns
df1_ = df1__.drop(columns=["Sex"]).join(df_sex1)
df1_ # Check the result


# In[170]:


from sklearn.preprocessing import OneHotEncoder
import pandas as pd  

# Initialize OneHotEncoder without dropping any category
OHE = OneHotEncoder(sparse_output=False)  # Ensures dense output

# Transform 'Sex' column
df_sex2 = pd.DataFrame(OHE.fit_transform(df2__[['Sex']]), 
                        columns=['female', 'male'],  # Correct column names
                        index=df2__.index)  # Keep index aligned

# Drop original "Sex" column & concatenate new columns
df2_= df2__.drop(columns=["Sex"]).join(df_sex2)

df2_  # Check the result


# In[173]:


from sklearn.preprocessing import OneHotEncoder
import pandas as pd  

# Initialize OneHotEncoder (without dropping any category)
OHE_embarked = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# Transform 'Embarked' column
df_embarked = pd.DataFrame(OHE_embarked.fit_transform(df1_[['Embarked']]), 
                           columns=OHE_embarked.get_feature_names_out(['Embarked']), 
                           index=df1_.index)  # Keep index aligned

# Drop original "Embarked" column & concatenate new columns
df1__ = df1_.drop(columns=["Embarked"]).join(df_embarked)

df1__  # Check the result


# In[175]:


from sklearn.preprocessing import OneHotEncoder
import pandas as pd  

# Initialize OneHotEncoder (without dropping any category)
OHE_embarked = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# Transform 'Embarked' column
df_embarked = pd.DataFrame(OHE_embarked.fit_transform(df2_[['Embarked']]), 
                           columns=OHE_embarked.get_feature_names_out(['Embarked']), 
                           index=df2_.index)  # Keep index aligned

# Drop original "Embarked" column & concatenate new columns
df2__ = df2_.drop(columns=["Embarked"]).join(df_embarked)

df2__ 


# In[144]:


# Print new feature names
print("Model will be trained on these features:", df2.columns.tolist())


# In[146]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
sns.stripplot(x=df2__['Survived'], y=df2__['Fare'], jitter=True)
plt.title('Fare vs Survived')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Fare')
plt.show()


# In[147]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
sns.stripplot(x=df2__['Survived'], y=df2__['Pclass'], jitter=True)
plt.title('Pclass vs Survived')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Pclass')
plt.show()


# In[148]:


plt.figure(figsize=(8,5))
sns.stripplot(x=df2__['Survived'], y=df2__['Age'], jitter=True)
plt.title('Age vs Survived')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Age')
plt.show()


# In[153]:


plt.figure(figsize=(8,5))
sns.stripplot(x=df2__['Survived'], y=df2__['SibSp'], jitter=True)
plt.title('Sibsp vs Survived')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('SibSp')
plt.show()


# In[152]:


plt.figure(figsize=(8,5))
sns.stripplot(x=df2__['Survived'], y=df2__['Parch'], jitter=True)
plt.title('Survived vs Parch')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Parch')
plt.show()


# In[154]:


plt.figure(figsize=(8,5))
sns.stripplot(x=df2__['Survived'], y=df2__['female'], jitter=True)
plt.title('Sibsp vs female')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('female')
plt.show()


# In[155]:


plt.figure(figsize=(8,5))
sns.stripplot(x=df2__['Survived'], y=df2__['male'], jitter=True)
plt.title('Sibsp vs male')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('male')
plt.show()


# In[159]:


plt.figure(figsize=(8,5))
sns.stripplot(x=df2__['Survived'], y=df2__['Embarked_C'], jitter=True)
plt.title('Sibsp vs Embarked_C')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Embarked_C')
plt.show()


# In[160]:


plt.figure(figsize=(8,5))
sns.stripplot(x=df2__['Survived'], y=df2__['Embarked_Q'], jitter=True)
plt.title('Sibsp vs Embarked_Q')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Embarked_Q')
plt.show()


# In[161]:


plt.figure(figsize=(8,5))
sns.stripplot(x=df2__['Survived'], y=df2__['Embarked_S'], jitter=True)
plt.title('Sibsp vs Embarked_S')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Embarked_S')
plt.show()


# In[163]:


import pandas as pd
import numpy as np

# Compute correlation matrix
corr_matrix = df1__.corr()

# Display the correlation matrix
print(corr_matrix)


# In[166]:


print(df1_cleansed.isnull().sum())  # Shows how many NaNs exist in each column


# In[167]:


print(df1_cleansed['Parch'].isnull().sum())  # Should print 0


# In[164]:


import pandas as pd
import numpy as np

# Compute correlation matrix
corr_matrix = df2__.corr()

# Display the correlation matrix
print(corr_matrix)


# In[168]:


import seaborn as sns
import matplotlib.pyplot as plt

heatmap1 = MM_X1.corr()#cleaned df

plt.figure(figsize=(12,6))
sns.heatmap(heatmap1,annot=True,fmt="g")


# In[123]:


import seaborn as sns
import matplotlib.pyplot as plt

heatmap1 = MM_X2.corr()#cleaned df

plt.figure(figsize=(12,6))
sns.heatmap(heatmap1,annot=True,fmt="g")


# In[176]:


# Split Data


X_train =df2__.drop(columns=["Survived"])  # Remove target column
y_train =df2__["Survived"]
# Print feature names to ensure correct format
print("Model was trained on these features:", X_train.columns.tolist())

# X_test = MM_X1


# In[177]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Train model with L2 Regularization
log_reg = LogisticRegression(penalty="l2", C=1.0, solver="liblinear")
log_reg.fit(X_train, y_train)

# Cross-validation
cv_scores = cross_val_score(log_reg, X_train, y_train, cv=5)
print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f}")


# In[179]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

# Predictions on training data
y_pred_train = log_reg.predict(X_train)

# Compute metrics
accuracy = accuracy_score(y_train, y_pred_train)
precision = precision_score(y_train, y_pred_train)
recall = recall_score(y_train, y_pred_train)
f1 = f1_score(y_train, y_pred_train)
roc_auc = roc_auc_score(y_train, log_reg.predict_proba(X_train)[:, 1])

# Confusion matrix
conf_matrix = confusion_matrix(y_train, y_pred_train)
tn, fp, fn, tp = conf_matrix.ravel()
specificity = tn / (tn + fp)

# Print results
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"ROC-AUC Score: {roc_auc:.4f}")

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_train, log_reg.predict_proba(X_train)[:, 1])
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, color="blue", label="ROC Curve")
plt.plot([0, 1], [0, 1], "k--")  # Random classifier line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()


# In[180]:


# Get feature importance (coefficients)
feature_importance = pd.DataFrame({"Feature": X_train.columns, "Coefficient": log_reg.coef_[0]})
feature_importance = feature_importance.sort_values(by="Coefficient", ascending=False)

print(feature_importance)


# In[181]:


import pickle

# Save the trained Logistic Regression model
with open("logistic_model.pkl", "wb") as file:
    pickle.dump(log_reg, file)


print("Model saved successfully!")


# In[ ]:




