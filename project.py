import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score,confusion_matrix, classification_report,mean_absolute_error,root_mean_squared_error,r2_score,f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder,StandardScaler,OneHotEncoder,RobustScaler
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(layout="wide", page_title="Fraud Detection Dashboard")

st.title("Fraud Detection Dashboard")
st.markdown("This dashboard analyzes a financial transaction dataset to detect fraudulent activities.")
st.header("1. Data Exploration")
tab1, tab2, tab3 = st.tabs(["Dataset Info", "Data Distribution", "Correlation Analysis"])

@st.cache_data
def read_csvdata(filename):
    df=pd.read_csv(filename)
    df=pd.DataFrame(df)
    return df
#df=pd.read_csv("AIML Dataset.csv")
df=read_csvdata("small.csv")

with tab1:
    st.subheader("Raw Data Preview")
    st.dataframe(df.head()) 
    
    st.subheader("Data Types")
    st.write(df.dtypes)

    st.subheader("Descriptive Statistics")
    st.write(df.describe())
    
    st.subheader("Missing Values")
    st.write(df.isnull().sum())

Y=df[["isFraud","isFlaggedFraud"]]
Y.head()
Y1=Y[["isFraud"]]
Y2=Y[["isFlaggedFraud"]]
print(Y1.head())
print(Y2.head())


with tab2:
    col1,col2=st.columns(2)
    with col1:
        st.subheader("Distribution of Target Variable: Is Fraud")
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        Y1.value_counts().plot(kind='pie', autopct='%.2f%%', colors=['purple', 'orange'], ax=ax1)
        ax1.set_title('Distribution of Is Fraud')
        ax1.set_ylabel('') 
        st.pyplot(fig1)
        plt.close(fig1)
    with col2:
        st.subheader("Distribution of Transaction Type")
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        df['type'].value_counts().plot(kind='pie', autopct='%.2f%%', ax=ax2)
        ax2.set_title('Distribution of Transaction Type')
        ax2.set_ylabel('')
        st.pyplot(fig2)
        plt.close(fig2)

df_encoded = pd.get_dummies(df, columns=['type'], drop_first=True) 
X = df_encoded[["step","amount","oldbalanceOrg","newbalanceOrig","oldbalanceDest","newbalanceDest"] + [col for col in df_encoded.columns if 'type' in col]]
print(X.head())

with tab3:
    st.subheader("Feature Correlation Heatmap")
    fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
    sns.heatmap(X.corr(), annot=True, cmap="coolwarm", ax=ax_corr)
    ax_corr.set_title("Correlation Matrix of Numerical Features")
    st.pyplot(fig_corr)
    plt.close(fig_corr)
    
    col1,col2=st.columns(2)
    with col1:
        st.subheader("Amount vs. New Balance Destination")
        fig_scatter1, ax_scatter1 = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x='amount', y='newbalanceDest', data=X, ax=ax_scatter1)
        ax_scatter1.set_title("Amount vs. New Balance Destination")
        st.pyplot(fig_scatter1)
        plt.close(fig_scatter1)
    with col2:    
        st.subheader("Old Balance Destination vs. New Balance Destination")
        fig_scatter2, ax_scatter2 = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x='oldbalanceDest', y='newbalanceDest', data=X, ax=ax_scatter2)
        ax_scatter2.set_title("Old Balance Destination vs. New Balance Destination")
        st.pyplot(fig_scatter2)
        plt.close(fig_scatter2)
    
    st.subheader("Pair Plot of Numerical Features")
    fig_pair = sns.pairplot(X)
    st.pyplot(fig_pair)
    



st.header("2. Model Training and Evaluation")

col1, col2 = st.columns(2)
with col1:
    test_percentage = st.number_input(label="Enter the test data percentage", min_value=0.0, max_value=1.0, value=0.25, step=0.05)
with col2:
    seed_number = st.number_input(label="Enter the seed number", min_value=0, max_value=1000, value=713, step=1)

#scaler=StandardScaler()
scaler=RobustScaler()
X_scaled=scaler.fit_transform(X)
print(X_scaled)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y1, test_size=test_percentage, random_state=seed_number)

st.info(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")

models = {
    "ExtraTreesClassifier": ExtraTreesClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Decision Tree" : DecisionTreeClassifier(),
    "MLPClassifier": MLPClassifier(), 
    "Logistic Regression": LogisticRegression(),
    "KNeighborsClassifier": KNeighborsClassifier(),    
}

@st.cache_resource
def trainingml(_models, X_train, y_train, X_test, y_test):
    results_df = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'F1 Score', 'MAE', 'RMSE', 'R2 Score'])
    model_tabs = st.tabs(list(_models.keys()))

    for i, (model_names, model) in enumerate(_models.items()):
        with model_tabs[i]: 
            st.subheader(f"Performance for {model_names}")
            model.fit(X_train, y_train.values.ravel())
            y_pred = model.predict(X_test) 
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = root_mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            f1=f1_score(y_test, y_pred,zero_division=0)
            
            new_row = pd.DataFrame([{
                'Model': model_names,
                'Accuracy': acc,
                'Precision': prec,
                'F1 Score': f1, 
                'MAE': mae,
                'RMSE': rmse,
                'R2 Score': r2
            }])
            results_df = pd.concat([results_df, new_row], ignore_index=True)
                
            st.write(f"**Accuracy:** {acc:.4f}")
            st.write(f"**Precision Score:** {prec:.4f}")
            st.write(f"**Mean Absolute Error:** {mae:.4f}")
            st.write(f"**Root Mean Squared Error:** {rmse:.4f}")
            st.write(f"**R2 Score:** {r2:.4f}")
            st.write(f"**F1 Score:** {f1:.4f}")
                
            st.markdown("---") 
            
            col1,col2,col3=st.columns(3)
            with col1:
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', ax=ax_cm) # Changed cmap for clarity
                ax_cm.set_title(f'{model_names} - Confusion Matrix')
                ax_cm.set_xlabel('Predicted')
                ax_cm.set_ylabel('True')
                st.pyplot(fig_cm)
                plt.close(fig_cm)      
    return results_df
      
    

results_df = trainingml(models, X_train, y_train, X_test, y_test)


st.markdown("---")
st.header("3. Overall Model Performance Summary")
st.subheader("Model Comparison (Sorted by R2 Score)")
st.dataframe(results_df.sort_values(by='R2 Score', ascending=False))
best_model_row = results_df.loc[results_df['R2 Score'].idxmax()]
st.success(f"The best performing model is **{best_model_row['Model']}** with an **R2 Score of {best_model_row['R2 Score']:.4f}**.")
results_df.sort_values(by='F1 Score', ascending=False)
best_model_row = results_df.loc[results_df['F1 Score'].idxmax()]
#st.markdown(f"---")
st.success(f"**{best_model_row['Model']}** is the best performing model with an **F1 Score of {best_model_row['F1 Score']:.2f}**.")