import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import joblib

# App Title
st.title("Data Mining")

# Dataset upload
uploaded_file = st.sidebar.file_uploader("Upload your dataset", type=["csv", "xlsx"])
df = None
if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error("Error reading file: " + str(e))
else:
    st.warning("Please upload a dataset!")

# Sidebar for navigation
with st.sidebar:
    st.subheader("Navigation")
    menu = st.radio("Choose a task:", [
        "1. Dataset Info",
        "2. Describe Dataset",
        "3. Handle Missing Values",
        "4. Handle Duplicates",
        "5. Handle Outliers",
        "6. Modeling"
    ])

# Dataset Info
def display_dataset_info():
    """Display detailed information about the dataset."""
    st.markdown("### Dataset Information")
    if "df" not in st.session_state:
        st.session_state.df = df.copy()  # Copy original df to session state

    df_session = st.session_state.df
    # Create a DataFrame for dataset info
    dataset_info = {
        "Column Name": df_session.columns,
        "Non-Null Count": df_session.notnull().sum(),
        
        
        "Data Type": df_session.dtypes
    }
    
    dataset_info_df = pd.DataFrame(dataset_info).reset_index(drop=True)
    
    # Add memory usage
    memory_usage = df_session.memory_usage(deep=True).sum() / 1024 ** 2  # Convert to MB
    st.markdown(f"#### Total Memory Usage: {memory_usage:.2f} MB, Shape: { df.shape} .")
    
    # Display dataset info as a table with gradient coloring
    styled_info = dataset_info_df.style.background_gradient(cmap="coolwarm")
    st.table(styled_info)

def display_dataset_description():
    if "df" not in st.session_state:
        st.session_state.df = df.copy()  # Copy original df to session state

    df_session = st.session_state.df
    """Display descriptive statistics for numeric and object columns separately."""
    st.markdown("### Descriptive Statistics")

    # Numeric columns
    numeric_cols = df_session.select_dtypes(include=['float64', 'int64'])
    if not numeric_cols.empty:
        st.markdown("#### Numeric Columns")
        styled_numeric = numeric_cols.describe().style.background_gradient(cmap="coolwarm")
        st.table(styled_numeric)
    else:
        st.warning("No numeric columns found in the dataset.")

    # Categorical columns
    object_cols = df_session.select_dtypes(include=['object'])
    if not object_cols.empty:
        st.markdown("#### Categorical Columns")
        styled_object = object_cols.describe().style.set_properties(**{'text-align': 'center'})
        st.table(styled_object)
    else:
        st.warning("No categorical columns found in the dataset.")

    n_rows = st.number_input("Number of rows to display", min_value=1, max_value=len(df), value=10)
    st.subheader(f"First {n_rows} rows of the dataset")
    st.table(df.head(n_rows))


def handle_missing_values():
    # Initialize DataFrame in session state if not already present
    if "df" not in st.session_state:
        st.session_state.df = df.copy()  # Copy original df to session state

    df_session = st.session_state.df  # Reference the session-state DataFrame

    st.subheader("Handle Missing Values")

    # Display unique values in each column before handling
    st.write("### Unique Values in Each Column (before handling):")
    
    # Show missing value summary
    st.write("### Missing Value Summary Before:")
    st.write(df_session.isnull().sum())

    # Identify columns with missing values
    columns_with_missing = [col for col in df_session.columns if df_session[col].isnull().sum() > 0]
    if not columns_with_missing:
        st.success("No missing values found!")

    # Select multiple columns to handle
    selected_columns = st.selectbox("Select columns to handle:", df_session.columns)

    # UI for selecting actions
    action = st.selectbox(
        "Select action to apply:",
        [
            "No Action",
            "Convert to Numeric",
            "Fill with Mean",
            "Fill with Median",
            "Fill with Mode",
            "Drop Rows",
            "Drop Columns",
            "Sort Values",
            "Fill NaN based on Categorical Target"
        ]
    )

    if selected_columns:
        
        with st.expander(f"View Unique Values for {selected_columns}"):
            value_counts = df_session[selected_columns].value_counts(dropna=False)
            st.write(f"Unique Values and Frequencies : { value_counts.count() }")   
            st.table(value_counts)
        if action == "Fill NaN based on Categorical Target":
            target_col = st.selectbox("Select Categorical Target Column", df_session.columns)

            # Ensure the target column is not one of the selected columns
            if target_col in selected_columns:
                st.warning("The target column cannot be one of the selected columns. Please select a different column.")
            else:
                fill_action = st.selectbox("Choose fill action:", ["Mean", "Median", "Mode"])
                handle_button = st.button("Handle Missing Values")

                if handle_button:
                    
                    if fill_action == "Mean":
                        df_session[selected_columns] = df_session.groupby(target_col)[selected_columns].transform(lambda x: x.fillna(x.mean()))
                    elif fill_action == "Median":
                        df_session[selected_columns] = df_session.groupby(target_col)[selected_columns].transform(lambda x: x.fillna(x.median()))
                    elif fill_action == "Mode":
                        df_session[selected_columns] = df_session.groupby(target_col)[selected_columns].transform(lambda x: x.fillna(x.mode()[0]))
                    st.write(f"Filled NaN values in selected columns based on the '{target_col}' column using {fill_action}.")

        elif action != "No Action":
            handle_button = st.button("Handle Missing Values")

            if handle_button:
            
                if action == "Convert to Numeric":
                    df_session[selected_columns] = pd.to_numeric(
                        df_session[selected_columns].replace(r'[^\d.]', '', regex=True),
                        errors='coerce'
                    )
                    st.write(f"Converted column '{selected_columns}' to numeric. Invalid values replaced with NaN.")
                    st.session_state['data'] = df_session 

                elif action == "Fill with Mean":
                    df_session[selected_columns].fillna(df_session[selected_columns].mean(), inplace=True)
                elif action == "Fill with Median":
                    df_session[selected_columns].fillna(df_session[selected_columns].median(), inplace=True)
                elif action == "Fill with Mode":
                    df_session[selected_columns].fillna(df_session[selected_columns].mode()[0], inplace=True)
                elif action == "Drop Rows":
                    df_session.dropna(subset=[selected_columns], inplace=True)
                elif action == "Drop Columns":
                    df_session.drop(columns=selected_columns, inplace=True)
                    st.write(f"Columns '{selected_columns}' have been dropped.")
                    
                
                # Display success message
                st.success(f"Applied '{action}' to selected columns.")

    # Show updated missing value summary after applying actions
    st.write("### Missing Value Summary After:")
    st.write(df_session.isnull().sum())
    st.session_state.df = df_session 

# Handle Duplicates
def handle_duplicates():
    st.subheader("Handle Duplicates")
    duplicates = df[df.duplicated()]
    num_duplicates = duplicates.shape[0]
    st.write(f"Number of duplicate rows: {num_duplicates}")

    if num_duplicates > 0:
        st.write("Duplicate rows:")
        st.dataframe(duplicates)

        if st.button("Remove Duplicates"):
            df.drop_duplicates(inplace=True)
            st.session_state['data'] = df  # Save changes to session state
            st.success("Duplicates removed!")
            st.write(f"Number of duplicate rows after removal: {df.duplicated().sum()}")
            st.table(df.head(len(df)))

# Handle Outliers
def handle_outliers():
    # Initialize DataFrame in session state if not already present
    if "df" not in st.session_state:
        st.session_state.df = df.copy()  # Copy original df to session state

    df_session = st.session_state.df  # Reference the session-state DataFrame
    df_preview= df_session.copy()
    st.subheader("Handle Outliers")

    # Calculate IQR and detect columns with outliers
    outlier_columns = []
    for column in df_session.select_dtypes(include=['float64', 'int64']).columns:
        Q1 = df_session[column].quantile(0.25)
        Q3 = df_session[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Check if there are any outliers in the column
        outliers = df_session[(df_session[column] < lower_bound) | (df_session[column] > upper_bound)]
        if not outliers.empty:
            outlier_columns.append(column)

    if outlier_columns:
        st.write(f"Columns with outliers: {', '.join(outlier_columns)}")
    else:
        st.success("No outliers detected in any numeric columns.")

    # Select the column to handle
    selected_column = st.selectbox("Select a column to handle:", outlier_columns)

    # Show boxplot for the selected column
    if selected_column:
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.boxplot(x=df_session[selected_column], ax=ax)
        ax.set_title(f"Boxplot of {selected_column} (Before Handling Outliers)")
        st.pyplot(fig)

        Q1 = df_preview[selected_column].quantile(0.25)
        Q3 = df_preview[selected_column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Remove outliers from the column
        df_preview[selected_column] = df_preview[selected_column].where(
            (df_preview[selected_column] >= lower_bound) & (df_preview[selected_column] <= upper_bound),
            df_preview[selected_column].median()  # Replace outliers with median (or other method)
        )
        # Show updated boxplot after removing outliers
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.boxplot(x=df_preview[selected_column], ax=ax)
        ax.set_title(f"Boxplot of {selected_column} (After Handling Outliers)")
        st.pyplot(fig)
        # Provide button to remove outliers for the selected column
        if st.button(f"Remove Outliers in '{selected_column}'"):
            Q1 = df_session[selected_column].quantile(0.25)
            Q3 = df_session[selected_column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Remove outliers from the column
            df_session[selected_column] = df_session[selected_column].where(
                (df_session[selected_column] >= lower_bound) & (df_session[selected_column] <= upper_bound),
                df_session[selected_column].median()  # Replace outliers with median (or other method)
            )

            st.success(f"Outliers removed in '{selected_column}'!")

            # Show updated dataframe and description
            st.write(f"**{selected_column}** - Descriptive Statistics After Removing Outliers:")
            st.write(df_session[selected_column].describe())

            # Show updated boxplot after removing outliers
            fig, ax = plt.subplots(figsize=(7, 6))
            sns.boxplot(x=df_session[selected_column], ax=ax)
            ax.set_title(f"Boxplot of {selected_column} (After Handling Outliers)")
            st.pyplot(fig)

            # Save changes to session state
            st.session_state.df = df_session
    else:
        st.write("No column selected to handle.")


def encode_columns():
    st.subheader("Encode Columns")
    
    if st.session_state.df is not None:
        df = st.session_state.df
        categorical_columns = df.select_dtypes(include=['object']).columns
            # Initialize encoded mappings in session state if not already present
        if "encoded_mappings" not in st.session_state:
            st.session_state.encoded_mappings = {}
        # Ensure that there are categorical columns
        if categorical_columns.any():
            col = st.selectbox("Select an **Object** column to encode:", categorical_columns)
            
            if col and st.button("Encode Column"):
                le = LabelEncoder()
                
                # Encode the selected column
                df[col] = le.fit_transform(df[col])
                st.session_state.encoded_mappings[col] = dict(zip( le.transform(le.classes_),le.classes_))

                st.success(f"'{col}' encoded successfully.")
                st.write("Mapping of encoded values:")
                st.table(st.session_state.encoded_mappings[col])
                st.session_state.df = df

        else:
            st.warning("No categorical columns found to encode.")
        
        target_col = st.selectbox("Select the target column for the split:", df.columns)
        y = df[target_col]
        x = df.drop(target_col, axis=1)  # Drop target column (corrected axis)
        try:
            # Split the data into train and test sets
            train_percentage = st.slider("Select the Percentage for training data", 1, 100)
            train_percentage /= 100
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=1-train_percentage, random_state=42, stratify=y)
            
            # Standardize the data
            scaler = StandardScaler()
            X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
            X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
            st.write("Training and testing sets have been split.")
        
        
            # Model selection
            models = {
                "KNN": KNeighborsClassifier(),
                "SVC": SVC(),
                "XGB": XGBClassifier(),
                "Logistic Regression": LogisticRegression()
            }
            
            model_name = st.selectbox("Select a model for training:", list(models.keys()))
            model = models[model_name]
        
            # Train and evaluate the selected model
            if model:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                # Calculate and display confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                st.write(f"Confusion Matrix for {model_name}:")
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                st.pyplot(fig)

                results = []
                # model.fit(X_train, y_train)
                # y_pred = model.predict(X_test)
                
                results.append({
                    "Model": model_name,
                    "Accuracy": accuracy_score(y_test, y_pred),
                    "F1 Score": f1_score(y_test, y_pred, average="weighted"),
                    "Precision": precision_score(y_test, y_pred, average="weighted"),
                    "Recall": recall_score(y_test, y_pred, average="weighted")
                })

                # Display Model Comparison
                results_df = pd.DataFrame(results)
                st.write("### Model Performance Comparison")
                st.table(results_df)
                
            if st.button(f"Save {model_name} Model"):
                joblib.dump(model, f'{model_name}.h5')

                joblib.dump(scaler, 'scaler.h5')
                st.success(f"Model saved as {model_name} and scaler .")
                
                # Save the trained model to session state
                st.session_state["trained_model"] = model
                st.session_state["scaler"] = scaler
                st.session_state["features"] = x.columns
            else:
                st.warning("No dataset loaded. Please upload a dataset.")
        except :
            st.warning("Please select more rows to training model.")


def predict_new_use_case():
    if "trained_model" in st.session_state:
        st.markdown("### Predict a New Use Case")
        
        trained_model = st.session_state["trained_model"]
        scaler = st.session_state["scaler"]
        features = st.session_state["features"]

        if 'df' not in st.session_state:
            st.warning("Dataset not found. Please upload the dataset first.")
            return
        
        df = st.session_state['df']

        noncat = [col for col in df.columns if df[col].nunique() > 4]
        cate = [col for col in df.columns if df[col].nunique() <= 10]
        
        # التعامل مع الأعمدة المشفرة وغير المشفرة
        
        input_data = []
        for feature in features:
            
            if feature in cate :
                
                # Display unique values and allow labeling
                unique_values = df[feature].unique()
                if feature not in st.session_state.encoded_mappings:
                    st.session_state.encoded_mappings[feature] = {}
                
                st.write(f"### Define labels for '{feature}' values:")
                for value in unique_values:

                    if value not in st.session_state.encoded_mappings[feature] :
                        label = st.text_input(
                            f"Label for '{value}' in '{feature}':", 
                            key=f"{feature}_{value}"
                        )
                        if label:
                            st.session_state.encoded_mappings[feature][value] = label

                st.write(f"### Current Mapping for '{feature}':")
                st.table(st.session_state.encoded_mappings[feature])

                

        
    
        input_data = []
        for feature in features:
            # تحقق إذا كان هناك ترميز مخصص للعمود
            if feature in st.session_state.encoded_mappings:
                st.write(f"Encoded Mapping for {feature}:")
                keys = list(st.session_state.encoded_mappings[feature].keys())
                valu = list(st.session_state.encoded_mappings[feature].values())
                
                # عرض القيم المشفرة للعمود الفئوي
                if feature in cate:
                    st.write(f"### Select encoded value for {feature}:")
                    selected_value = st.selectbox(f"### Select encoded value for {feature}:", keys)
                    input_data.append(selected_value)
                  # Display the value and column together
                    input_data_dict = {feature: value for feature, value in zip(features, input_data)}
                else:  # في حالة الأعمدة الرقمية
                    st.write(f"### Enter value for {feature}:")
                    selected_value = st.number_input(f"### Enter value for {feature}:",
                                                     min_value=float(df[feature].min()), 
                                                     max_value=float(df[feature].max()), 
                                                     step=0.01)
                    input_data.append(selected_value)
                    input_data_dict = {feature: value for feature, value in zip(features, input_data)}

                # طباعة القيم المشفرة مع المفاتيح الخاصة بها
                for key, value in zip(keys, valu):
                    st.write(f"Key: {key}, Value: {value}")
            else:
                st.warning(f"No encoded mapping found for feature: {feature}")
                
                # في حالة الأعمدة الفئوية بدون ترميز مخصص، سيتم عرض القيم الفريدة في selectbox
                if feature in cate:
                    st.write(f"### Select encoded value for {feature}:")
                    selected_value = st.selectbox(f"Select value for {feature}:", df[feature].unique())
                    input_data.append(selected_value)
                    input_data_dict = {feature: value for feature, value in zip(features, input_data)}
                else:
                    st.write(f"### Enter value for {feature}:")
                    selected_value = st.number_input(f"Enter value for {feature}:",
                                                     min_value=float(df[feature].min()),
                                                     max_value=float(df[feature].max()),
                                                     step=0.01)
                    input_data.append(selected_value)
                    input_data_dict = {feature: value for feature, value in zip(features, input_data)}
        st.write("### Input Data:")
        st.markdown("### Input Data Preview:")
        st.dataframe(pd.DataFrame(input_data_dict, index=[0]).T.style.set_properties(**{'text-align': 'center'}))

        # التنبؤ بناءً على المدخلات
        if st.button("Predict"):
            try:
                # تطبيع المدخلات بناءً على المقياس المستخدم في التدريب
                input_data_scaled = scaler.transform([input_data])
                prediction = trained_model.predict(input_data_scaled)
                st.success(f"Predicted Output: {prediction[0]}")
            except Exception as e:
                st.error(f"Error during prediction: {e}")
    else:
        st.warning("Train a model first to enable prediction.")


# Task Navigation
if df is not None:
    if menu == "1. Dataset Info":
        display_dataset_info()
    elif menu == "2. Describe Dataset":
        display_dataset_description()
    elif menu == "3. Handle Missing Values":
        handle_missing_values()
    elif menu == "4. Handle Duplicates":
        handle_duplicates()
    elif menu == "5. Handle Outliers":
        handle_outliers()
    elif menu == "6. Modeling":
        encode_columns()
        predict_new_use_case()

else:
    st.info("Upload a dataset to begin.")

# Download modified dataset
if df is not None:

    st.sidebar.download_button(
        label="Download Modified Dataset",
        data=st.session_state.df.to_csv(index=False),
        file_name="modified_dataset.csv",
        mime="text/csv"
    )



