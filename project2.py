import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from scipy import stats
#import plotly.express as px
import io

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score,confusion_matrix, classification_report,mean_absolute_error,root_mean_squared_error,r2_score,f1_score
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder,StandardScaler,OneHotEncoder,RobustScaler,MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier    


@st.cache_data
def read_csvdata(filename):
    df = pd.read_csv(filename)
    df = pd.DataFrame(df)
    return df

def reset_state(list1):
    for key in list1:
        if key not in st.session_state:
            st.session_state[key] = None
list1=["df","encoded_df","dtypes_df","orignal_dtypes","target","catvar","disvar","convar"]
reset_state(list1)

st.set_page_config(layout="wide", page_title="Fraud Detection Dashboard")

st.title("Auto ML Dashboard")
st.markdown("This dashboard analyzes a dataset with ML to predict target variables given data of independent variables.")

uploaded_file=st.file_uploader("Upload a dataset as a csv file", type="csv", accept_multiple_files=False)

if uploaded_file is None:
    st.info("Please upload a dataset as a csv file.")
    st.stop()
if uploaded_file is not None:
    try:
        #df_upload = pd.read_csv(uploaded_file)
        df=read_csvdata(uploaded_file)
        st.session_state.df = df.copy() 
        st.session_state.original_dtypes = df.dtypes.astype(str) 
        st.session_state.encoded_df = None 
        st.success("CSV File Uploaded Successfully!")
    except Exception as e:
        st.error(f"Error reading or processing the CSV file: {e}")
        st.session_state.df = None 
        st.stop()
#st.write(df)

st.header("1. Data Exploration")
tab1, tab2, tab3 = st.tabs(["Dataset Info", "Data Distribution", "Correlation Analysis"])

with tab1:
    st.subheader("Raw Data Preview")
    st.metric(label="Total Columns", value=df.shape[1])
    st.metric(label="Total Rows", value=df.shape[0])
    
    st.dataframe(df.head()) 
    
    #st.subheader("Data Types")
    #st.write(df.dtypes)
    
    dtypes_df = df.dtypes.reset_index()
    dtypes_df.columns = ['Column', 'Detected Data Type']
    dtypes_df['Detected Data Type'] = dtypes_df['Detected Data Type'].astype(str) 
    st.write("Data Type Summary:")
    st.dataframe(dtypes_df, use_container_width=True)
    
    
    st.subheader("Descriptive Statistics")
    st.write(df.describe())
    
    st.subheader("Missing Value Counts per Column:")
    missing_values = df.isnull().sum().reset_index()
    missing_values.columns = ['Column', 'Missing Count']
    st.dataframe(missing_values, use_container_width=True,hide_index=True)
    
    if st.session_state.df is not None:
        df = st.session_state.df.copy()
        potential_categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if not potential_categorical_cols:
            st.warning("No obvious categorical columns (object or category dtype) found in the dataset.")
            st.info("If you have numerical columns that represent categories (e.g., IDs, codes), you might need to convert them to 'object' or 'category' type first using a separate process or manually before encoding here.")
        else:
            st.subheader("Select Categorical Columns to Encode:")
            selected_cols_to_encode = st.multiselect(
                "Choose columns:",
                options=potential_categorical_cols,default=[potential_categorical_cols[0]],
                help="These are columns with 'object' or 'category' data types."
            )
            

            if selected_cols_to_encode:
                st.subheader("Choose Encoding Method:")
                encoding_method = st.selectbox(
                    "Select one:",
                    ("No Encoding","One-Hot Encoding", "Label Encoding"),
                    #horizontal=True
                )
                if encoding_method == "One-Hot Encoding":
                    drop_first = st.checkbox("Drop first category (to avoid multicollinearity)", value=True)
                df_encoded = df.copy()
                if encoding_method == "One-Hot Encoding":
                    try:
                        df_encoded = pd.get_dummies(df_encoded, columns=selected_cols_to_encode, prefix=selected_cols_to_encode, drop_first=drop_first)
                        st.session_state.encoded_df = df_encoded
                        st.success("One-Hot Encoding Applied!")
                    except Exception as e:
                        st.error(f"Error during One-Hot Encoding: {e}")

                elif encoding_method == "Label Encoding":
                    label_encoders = {}
                    try:
                        for col in selected_cols_to_encode:
                            le = LabelEncoder()                                
                            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str)) 
                            label_encoders[col] = dict(zip(le.classes_, le.transform(le.classes_)))
                        st.session_state.encoded_df = df_encoded
                        st.success("Label Encoding Applied!")

                        if label_encoders:
                            st.subheader("Label Encoding Mappings:")
                            for col, mapping in label_encoders.items():
                                with st.expander(f"Mappings for '{col}'"):
                                    st.json(mapping)
                    
                    except Exception as e:
                        st.error(f"Error during Label Encoding: {e}")
            st.subheader("Select Columns to Remove:")
            potential_categorical_cols=list(col for col in potential_categorical_cols if col not in selected_cols_to_encode)
            st.write(df_encoded.columns)
            selected_cols_to_remove = st.multiselect(
                "Choose columns:",
                options=potential_categorical_cols)
            
            if selected_cols_to_remove:
                df_encoded.drop(selected_cols_to_remove,axis=1,inplace=True)
                st.session_state.encoded_df=df_encoded
            else:
                st.markdown("_Select one or more categorical columns to see encoding options._")
        
        
        if st.session_state.encoded_df is not None:
            
            st.markdown("---")
            st.header("Encoded Data")
            st.subheader("Preview of Encoded Data (First 5 Rows):")
            st.dataframe(st.session_state.encoded_df.head())

            st.subheader("Data Types After Encoding:")
            st.dataframe(st.session_state.encoded_df.dtypes.astype(str).rename("Data Type"), use_container_width=True)
            @st.cache_data 
            def convert_df_to_csv(df_to_convert):
                return df_to_convert.to_csv(index=False).encode('utf-8')

            csv_encoded = convert_df_to_csv(st.session_state.encoded_df)
            st.download_button(
                label="Download Encoded Data as CSV",
                data=csv_encoded,
                file_name=f"encoded_{uploaded_file.name if uploaded_file else 'data'}.csv",
                mime='text/csv',
            )
    else:
        st.info("Awaiting CSV file upload to begin...")
    
    
        


st.header("Choose Target Feature")
options=df_encoded.columns
index=len(options)-1
if st.session_state.target is not None:
    target=st.session_state.target
target=st.selectbox("Choose Target Variable",options,index=index)
if st.session_state.target is None:
    st.session_state.target=target
Y=df_encoded[target]
st.subheader("Target Variable")
st.write(Y.head())
st.subheader("Independent Variables")
X=df_encoded.drop(target,axis=1,inplace=False)

numeric_cols = list(X.select_dtypes(include=[np.number]).columns)
non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
print("Non-numeric columns:", non_numeric_cols)
#st.write("Non-numeric columns:", non_numeric_cols)


st.write(X.head())








with tab2:
    col1,col2,col3=st.columns(3)
    with col1:
        label="Choose a Categorical Variable to Visualize:"
        if st.session_state.catvar is not None:
            catvar=st.session_state.catvar
        catvar=st.selectbox(label,options,index=index)
        st.session_state.catvar=catvar
        st.subheader(f"Distribution of Categorical Variable: {catvar}")
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        df[catvar].value_counts().plot(kind='pie', autopct='%.2f%%', ax=ax1)
        ax1.set_title(f'Distribution of {catvar}')
        ax1.set_ylabel('') 
        st.pyplot(fig1)
        plt.close(fig1)
    with col2:
        label="Choose a continouus Variable to Visualize"
        if st.session_state.convar is not None:
            convar=st.session_state.convar
        convar=st.selectbox(label,options)
        st.subheader(f"Distribution of {convar} Variable")
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        #line', 'bar', 'barh', 'kde', 'density', 'area', 'hist', 'box', 'pie',
        # 'scatter', 'hexbin
        df[convar].sort_values().plot(kind='hist',bins=30, ax=ax2)
        st.session_state.convar=convar
        ax2.set_title(f'Distribution of {convar}')
        ax2.set_ylabel('')
        st.pyplot(fig2)
        plt.close(fig2)
    with col3:
        label="Choose a discrete numerical Variable to Visualize"
        if st.session_state.disvar is not None:
            disvar=st.session_state.disvar
        disvar=st.selectbox(label,options)
        st.session_state.disvar=disvar
        st.subheader(f"Distribution of {disvar} Variable")
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        #line', 'bar', 'barh', 'kde', 'density', 'area', 'hist', 'box', 'pie',
        # 'scatter', 'hexbin
        df[disvar].value_counts().plot(kind='bar', ax=ax2)
        ax2.set_title(f'Distribution of {disvar}')
        ax2.set_ylabel('')
        st.pyplot(fig2)
        plt.close(fig2)





with tab3:
    df_encoded.select_dtypes(include=[np.number]).hist(bins=50, figsize=(12, 8),color="purple")
    plt.show()
    fig_hist=plt.gcf()
    st.subheader("Histograms of Numerical Features")
    st.pyplot(fig_hist)
    
    st.subheader("Feature Correlation Heatmap")
    fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
    sns.heatmap(df_encoded.select_dtypes(include=[np.number]).corr(), annot=True, cmap="coolwarm", ax=ax_corr)
    #sns.heatmap(X.corr(), annot=True, cmap="coolwarm", ax=ax_corr)
    ax_corr.set_title("Correlation Matrix of Numerical Features")
    st.pyplot(fig_corr)
    plt.close(fig_corr)
        
    st.subheader("Pair Plot of Numerical Features")
    #fig = px.scatter_matrix(df_encoded, dimensions=df_encoded.select_dtypes(include='number').columns,color=target, opacity=0.8)
    #st.plotly_chart(fig,height=800)
    
    fig_pair = sns.pairplot(df_encoded, plot_kws={'alpha': 0.2})

    st.pyplot(fig_pair)
    
    
st.header("2. Model Training and Evaluation")

col1, col2,col3 = st.columns(3)
with col1:
    test_percentage = st.number_input(label="Enter the test data percentage", min_value=0.0, max_value=1.0, value=0.25, step=0.05)
with col2:
    seed_number = st.number_input(label="Enter the seed number", min_value=0, max_value=1000, value=958, step=1)
with col3:
    string=st.selectbox("Select a Scaler for Numerical Features:", ("RobustScaler", "StandardScaler","MinMaxScaler","No Scaling"),index=0)
    if string=="RobustScaler":
        scaler=RobustScaler()
    elif string=="StandardScaler":
        scaler=StandardScaler()
    elif string=="MinMaxScaler":
        scaler=MinMaxScaler()
    else:
        scaler=None
if scaler is not None:
    X_scaled=scaler.fit_transform(X)
else:
    X_scaled=X
print(X_scaled)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=test_percentage, random_state=seed_number)

st.info(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")


models = {
    "ExtraTreesClassifier": ExtraTreesClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Decision Tree" : DecisionTreeClassifier(),
    "MLPClassifier": MLPClassifier(), 
    "Logistic Regression": LogisticRegression(),
    "KNeighborsClassifier": KNeighborsClassifier(), 
    "SVC": SVC(probability=True, random_state=seed_number), 
    "Gaussian Naive Bayes": GaussianNB(), 
    "SGD Classifier": SGDClassifier(loss='log_loss', random_state=seed_number),
    "XGBoost":XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=seed_number),
    "LGBMClassifier":LGBMClassifier(random_state=seed_number),
    "CatBoostClassifier": CatBoostClassifier(verbose=0, random_state=seed_number)
}
   


@st.cache_resource
def trainingml(_models, X_train, y_train, X_test, y_test):
    results_df = pd.DataFrame(columns=['Model', 'Train Accuracy', 'Test Accuracy', 'Precision', 'F1 Score', 'MAE', 'RMSE', 'R2 Score','Cross Validation Score'])
    model_tabs = st.tabs(list(_models.keys()))

    for i, (model_names, model) in enumerate(_models.items()):
        with model_tabs[i]: 
            st.subheader(f"Performance for {model_names}")
            model.fit(X_train, y_train.values.ravel())
            y_pred_test = model.predict(X_test) 
            y_pred_train=model.predict(X_train)
            train_acc = accuracy_score(y_train, y_pred_train)
            test_acc = accuracy_score(y_test, y_pred_test)
            prec = precision_score(y_test, y_pred_test, zero_division=0)
            mae = mean_absolute_error(y_test, y_pred_test)
            rmse = root_mean_squared_error(y_test, y_pred_test)
            r2 = r2_score(y_test, y_pred_test)
            f1=f1_score(y_test, y_pred_test,zero_division=0)
            scores = cross_val_score(model, X_train, y_train, cv=5)
            score=np.mean(scores)
            
            new_row = pd.DataFrame([{
                'Model': model_names,
                'Train Accuracy': train_acc,
                'Test Accuracy': test_acc,
                'Precision': prec,
                'F1 Score': f1, 
                'MAE': mae,
                'RMSE': rmse,
                'R2 Score': r2 ,
                'Cross Validation Score': score
            }])
            results_df = pd.concat([results_df, new_row], ignore_index=True)
                
            st.write(f"**Train Accuracy:** {train_acc:.4f}")
            st.write(f"**Test Accuracy:** {test_acc:.4f}")
            st.write(f"**Precision Score:** {prec:.4f}")
            st.write(f"**Mean Absolute Error:** {mae:.4f}")
            st.write(f"**Root Mean Squared Error:** {rmse:.4f}")
            st.write(f"**R2 Score:** {r2:.4f}")
            st.write(f"**F1 Score:** {f1:.4f}")
            st.write(f"**Cross Validation Score:** {score:.4f}")
                
            st.markdown("---") 
            
            col1,col2,col3=st.columns(3)
            with col1:
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred_test)
                fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', ax=ax_cm) 
                ax_cm.set_title(f'{model_names} - Confusion Matrix')
                ax_cm.set_xlabel('Predicted')
                ax_cm.set_ylabel('True')
                st.pyplot(fig_cm)
                plt.close(fig_cm)   
            with col2:
                st.subheader("Classification Report (Test Set)")
                report = classification_report(y_test, y_pred_test, output_dict=True, zero_division=0)
                report1_df = pd.DataFrame(report).transpose()
                st.dataframe(report1_df.round(2))   
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
st.success(f"**{best_model_row['Model']}** is the best performing model with an **F1 Score of {best_model_row['F1 Score']:.2f}**.")
st.markdown(f"---")