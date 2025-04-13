import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
import google.generativeai as genai
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Custom CSS for styling the app
st.markdown(
    """
    <style>
    .reportview-container {
        background-color: #f4f6f9;
        color: #333;
    }
    .sidebar .sidebar-content {
        background-color: #20232a;
        color: white;
    }
    h1 {
        font-family: 'Arial', sans-serif;
        color: #F39C12;
    }
    .stButton>button {
        background-color: #F39C12;
        color: white;
        font-weight: bold;
    }
    .stSlider>div>label {
        color: #F39C12;
    }
    .stTextInput>label {
        color: #F39C12;
    }
    </style>
    """, unsafe_allow_html=True
)

def main():
    st.title("ML/AI Explorer")
    
    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Select Task",
        ["Regression", "Clustering", "Neural Network", "LLM Q&A"]
    )
    
    if page == "Regression":
        regression_page()
    elif page == "Clustering":
        clustering_page()
    elif page == "Neural Network":
        neural_network_page()
    elif page == "LLM Q&A":
        llm_page()

def regression_page():
    st.header("Regression Analysis")
    
    # File upload
    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:")
        st.dataframe(df.head())
        
        # Check for numeric columns only
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        
        if len(numeric_columns) < 2:
            st.error("Your dataset needs at least 2 numeric columns for regression analysis.")
            return
            
        # Select target column FIRST (before preprocessing)
        target_col = st.selectbox("Select target column (numeric only)", numeric_columns)
        
        # Select feature columns SECOND
        remaining_numeric = [col for col in numeric_columns if col != target_col]
        feature_cols = st.multiselect("Select feature columns (numeric only)", remaining_numeric)
        
        # NOW do preprocessing (after feature_cols is defined)
        if feature_cols:  # Only show preprocessing options if features are selected
            st.subheader("Data Preprocessing")
            preprocessing_options = st.multiselect("Select preprocessing steps:", 
                                                ["Remove outliers", "Normalize features", "Log transform"])

            if "Remove outliers" in preprocessing_options:
                # Simple IQR-based outlier removal
                for col in numeric_columns:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    df = df[~((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))]
                st.success(f"Outliers removed. Remaining rows: {len(df)}")

            if "Normalize features" in preprocessing_options:
                for col in feature_cols:  # Now feature_cols is defined
                    df[col] = (df[col] - df[col].mean()) / df[col].std()
                st.success("Features normalized (z-score)")

            if "Log transform" in preprocessing_options:
                for col in feature_cols:  # Now feature_cols is defined
                    if (df[col] > 0).all():
                        df[col] = np.log(df[col])
                st.success("Applied log transformation to positive feature columns")
        
        if feature_cols and target_col:
            X = df[feature_cols]
            y = df[target_col]
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Display metrics
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            st.write(f"Mean Absolute Error: {mae:.2f}")
            st.write(f"R² Score: {r2:.2f}")
            
            # Plot predictions vs actual
            fig = px.scatter(x=y_test, y=y_pred, 
                           labels={'x': 'Actual Values', 'y': 'Predicted Values'},
                           title='Predictions vs Actual Values')
            fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], 
                                   y=[y_test.min(), y_test.max()],
                                   mode='lines', name='Perfect Prediction'))
            st.plotly_chart(fig)
            
            # Custom prediction
            st.subheader("Make Custom Predictions")
            input_data = {}
            for col in feature_cols:
                input_data[col] = st.number_input(f"Enter {col}")
            
            if st.button("Predict"):
                custom_input = pd.DataFrame([input_data])
                prediction = model.predict(custom_input)[0]
                st.write(f"Predicted {target_col}: {prediction:.2f}")

def clustering_page():
    st.header("K-Means Clustering")
    
    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:")
        st.dataframe(df.head())
        
        # Select features for clustering
        feature_cols = st.multiselect("Select features for clustering", df.columns)
        
        if len(feature_cols) >= 2:
            # Check for categorical columns and encode them
            X = df[feature_cols].copy()
            categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
            
            if categorical_cols:
                st.info(f"Encoding categorical features: {', '.join(categorical_cols)}")
                # Apply label encoding to categorical columns
                from sklearn.preprocessing import LabelEncoder
                
                for col in categorical_cols:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col])
                
                st.write("Encoded data preview:")
                st.dataframe(X.head())
            
            # Number of clusters
            n_clusters = st.slider("Select number of clusters", 2, 10, 3)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(X)
            
            # Visualize clusters
            if len(feature_cols) == 2:
                # Create DataFrame with original data and cluster labels
                plot_df = pd.DataFrame({
                    'x': X[feature_cols[0]],  # Use encoded values for plotting
                    'y': X[feature_cols[1]],  # Use encoded values for plotting
                    'cluster': clusters
                })
                
                # Add original values for hover info
                for col in feature_cols:
                    plot_df[col] = df[col]  # Add original values as separate columns
                
                # Create plot with hover data
                fig = px.scatter(
                    plot_df, x='x', y='y', 
                    color='cluster',
                    labels={'x': feature_cols[0], 'y': feature_cols[1]},
                    title='Cluster Visualization',
                    hover_data=[col for col in feature_cols]  # Now these columns exist
                )
                
                # Add centroids
                centroids = kmeans.cluster_centers_
                fig.add_trace(go.Scatter(
                    x=centroids[:, 0], y=centroids[:, 1],
                    mode='markers',
                    marker=dict(symbol='x', size=15, color='black'),
                    name='Centroids'
                ))
                
                st.plotly_chart(fig)
            
            elif len(feature_cols) == 3:
                # Create 3D visualization
                plot_df = pd.DataFrame({
                    'x': df[feature_cols[0]],
                    'y': df[feature_cols[1]],
                    'z': df[feature_cols[2]],
                    'cluster': clusters
                })
                
                fig = px.scatter_3d(
                    plot_df, x='x', y='y', z='z',
                    color='cluster',
                    labels={'x': feature_cols[0], 'y': feature_cols[1], 'z': feature_cols[2]},
                    title='3D Cluster Visualization'
                )
                
                # Add centroids to 3D plot
                centroids = kmeans.cluster_centers_
                fig.add_trace(go.Scatter3d(
                    x=centroids[:, 0], y=centroids[:, 1], z=centroids[:, 2],
                    mode='markers',
                    marker=dict(symbol='x', size=8, color='black'),
                    name='Centroids'
                ))
                
                st.plotly_chart(fig)
            
            else:
                st.warning("For visualization, please select either 2 or 3 features.")
            
            # Add cluster labels to dataframe
            df['Cluster'] = clusters
            
            # Download clustered data
            st.download_button(
                "Download Clustered Data",
                df.to_csv(index=False).encode('utf-8'),
                "clustered_data.csv",
                "text/csv"
            )

def neural_network_page():
    st.header("Neural Network Training")
    
    # Initialize session state for model persistence
    if 'nn_model' not in st.session_state:
        st.session_state.nn_model = None
        st.session_state.scaler = None
        st.session_state.target_encoder = None
        st.session_state.classes = None
        st.session_state.feature_cols = None
        st.session_state.categorical_cols = None
        st.session_state.target_col = None
        st.session_state.is_categorical_target = False
        st.session_state.trained_feature_cols = None  # Store features used for training

    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:")
        st.dataframe(df.head())
        
        # Select target column
        target_col = st.selectbox("Select target column", df.columns)
        feature_cols = st.multiselect("Select feature columns", 
                                     [col for col in df.columns if col != target_col])
        
        if feature_cols and target_col:
            # Create a copy of the features dataframe
            X = df[feature_cols].copy()
            
            # Check for categorical columns and encode them
            categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
            
            if categorical_cols:
                st.info(f"Encoding categorical features: {', '.join(categorical_cols)}")
                # Apply label encoding to categorical columns
                from sklearn.preprocessing import LabelEncoder
                
                for col in categorical_cols:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col])
                
                st.write("Encoded data preview:")
                st.dataframe(X.head())
            
            # Save feature selection to session state
            st.session_state.feature_cols = feature_cols
            st.session_state.categorical_cols = categorical_cols
            st.session_state.target_col = target_col
            
            # Hyperparameters
            epochs = st.slider("Number of epochs", 10, 100, 50)
            learning_rate = st.number_input("Learning rate", 0.0001, 0.1, 0.001)
            
            # Train button - now only retrains when explicitly clicked
            train_button = st.button("Train Model")
            
            # Only train if button is clicked or no model exists yet
            if train_button or st.session_state.nn_model is None:
                # Scale features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Encode target if categorical
                is_categorical_target = df[target_col].dtype == 'object'
                st.session_state.is_categorical_target = is_categorical_target
                
                if is_categorical_target:
                    target_encoder = LabelEncoder()
                    y = target_encoder.fit_transform(df[target_col])
                    # Store class names for later prediction display
                    classes = target_encoder.classes_
                    st.session_state.target_encoder = target_encoder
                    st.session_state.classes = classes
                    st.info(f"Target classes: {', '.join(classes)}")
                else:
                    y = df[target_col]
                
                # Create and train MLP model
                with st.spinner("Training model..."):
                    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), 
                                      activation='relu',
                                      solver='adam',
                                      learning_rate_init=learning_rate,
                                      max_iter=epochs,
                                      random_state=42,
                                      verbose=True)
                    
                    mlp.fit(X_scaled, y)
                
                # Store trained objects in session state
                st.session_state.nn_model = mlp
                st.session_state.scaler = scaler
                
                # Store which features were used for training
                st.session_state.trained_feature_cols = feature_cols.copy()
                
                # Plot training loss
                plt.figure(figsize=(10, 6))
                plt.plot(mlp.loss_curve_)
                plt.title('Training Loss')
                plt.xlabel('Iterations')
                plt.ylabel('Loss')
                st.pyplot(plt)
            
            # Only show the prediction UI if we have a trained model
            if st.session_state.nn_model is not None:
                # Show warning if current features don't match training features
                if st.session_state.trained_feature_cols != feature_cols:
                    st.warning(f"Warning: The model was trained with different features. Please click 'Train Model' to update the model with your current feature selection.")
                
                # Use the training features for the prediction UI
                prediction_features = st.session_state.trained_feature_cols
                
                # Create input UI based on training features
                input_data = {}
                for col in prediction_features:
                    # For categorical columns, use a selectbox with the original values
                    if col in categorical_cols:
                        original_values = df[col].unique().tolist()
                        # Add a unique key for each selectbox
                        input_val = st.selectbox(f"Select {col}", original_values, key=f"select_{col}")
                        # Encode the selected value
                        le = LabelEncoder().fit(df[col])
                        input_data[col] = le.transform([input_val])[0]
                    else:
                        # For numeric columns, use a number input with unique key
                        input_data[col] = st.number_input(f"Enter {col}", key=f"input_{col}")
                
                # Unique key for predict button
                if st.button("Predict", key="predict_button"):
                    custom_input = np.array([[input_data[col] for col in prediction_features]])
                    custom_input_scaled = st.session_state.scaler.transform(custom_input)
                    
                    # Make prediction
                    if st.session_state.is_categorical_target:
                        # For categorical targets, show class probabilities
                        probs = st.session_state.nn_model.predict_proba(custom_input_scaled)[0]
                        st.write("Prediction probabilities:")
                        for i, prob in enumerate(probs):
                            st.write(f"{st.session_state.classes[i]}: {prob:.4f}")
                        pred_class = st.session_state.classes[np.argmax(probs)]
                        st.success(f"Predicted class: {pred_class}")
                    else:
                        # For numeric targets
                        prediction = st.session_state.nn_model.predict(custom_input_scaled)[0]
                        st.success(f"Predicted {target_col}: {prediction}")

def process_document(uploaded_file, embeddings):
    """Process either PDF or CSV documents and create a vector store"""
    if uploaded_file.type == "application/pdf":
        # Extract text from PDF
        document_text = ""
        reader = PyPDF2.PdfReader(uploaded_file)
        for page in reader.pages:
            document_text += page.extract_text() + "\n"
        st.success(f"PDF document loaded: {len(reader.pages)} pages")
        
        # Create text chunks for better retrieval
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_text(document_text)
        
        # Display sample of processed text
        st.subheader("Sample of processed text:")
        st.write(document_text[:500] + "...")
        
    elif uploaded_file.type == "text/csv":
        df = pd.read_csv(uploaded_file)
        st.write("CSV data loaded:")
        st.dataframe(df.head())
        
        # Process CSV data
        texts = []
        for index, row in df.iterrows():
            # Convert each row to a text description
            text = f"Row {index}: "
            for col in df.columns:
                text += f"{col}: {row[col]}. "
            texts.append(text)
        
        # Join texts and split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_text("\n".join(texts))
    
    # Create vector store with text chunks
    return FAISS.from_texts(chunks, embeddings)

def generate_answer(question, vector_store):
    """Generate answer using RAG with Gemini"""
    # Retrieve relevant document chunks
    docs = vector_store.similarity_search(question, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    
    with st.spinner("Generating answer with Gemini..."):
        # Configure the model
        generation_config = {
            "temperature": 0.2,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 1024,
        }
        
        # Create the prompt with context and question
        prompt = f"""You are a helpful assistant that answers questions based on the context provided.
        
        Context:
        {context}
        
        Question: {question}
        
        Provide a detailed answer based only on the information in the context. If the answer is not in the context, say "I don't have enough information to answer that question."
        """
        
        # Generate response from Gemini
        model = genai.GenerativeModel(model_name="gemini-1.5-pro", generation_config=generation_config)
        response = model.generate_content(prompt)
        
        # Display answer
        st.subheader("Answer:")
        st.write(response.text)
        
        # Display relevant context
        with st.expander("Relevant document sections"):
            st.write(context)
    
    return context

def llm_page():
    st.header("Large Language Model Q&A")
    
    # Configure Gemini API
    genai.configure(api_key="AIzaSyCBGaG28fdZ9oCtNp2DzWcSDkK4Tnmo93E")
    
    try:
        # File upload for context
        uploaded_file = st.file_uploader("Upload your document (PDF/CSV)", type=['pdf', 'csv'])
        
        if uploaded_file is not None:
            # Load embeddings model
            with st.spinner("Loading embedding model..."):
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            
            # Process document and create vector store
            vector_store = process_document(uploaded_file, embeddings)
            
            # Question input
            question = st.text_input("Enter your question about the document")
            
            if question and st.button("Get Answer"):
                generate_answer(question, vector_store)
        
        else:
            st.info("Please upload a PDF or CSV file to ask questions about its content.")
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Try using a smaller file or check your internet connection.")

if __name__ == "__main__":
    main()
