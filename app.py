# Section 1: Importing Libraries
import tkinter as tk
from tkinter import messagebox, filedialog
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier  # Import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import torch
import re
import nltk

# Section 2: NLTK Stop Words Setup
nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('english'))

# Section 3: Global Variables
trained_rf_model = None
trained_vectorizer = None
trained_dt_model = None  # Variable to store the trained Decision Tree model
df = None  # Global dataframe to store CSV data
text_entry = None  # Declare text_entry globally
result_text = None  # Declare result_text globally
X_train, X_test, y_train, y_test = None, None, None, None  # For holding training/testing splits

# Section 4: Functions for Data Preprocessing and Model Prediction
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # Remove URLs
    text = re.sub(r"[^A-Za-z\s]", "", text)  # Remove non-alphabetical characters
    text = text.lower()  # Convert to lowercase
    text = " ".join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

def load_csv():
    global df
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if file_path:
        df = pd.read_csv(file_path)
        if 'statement' in df.columns and 'status' in df.columns:
            messagebox.showinfo("Success", "CSV loaded successfully.")
        else:
            messagebox.showerror("Error", "CSV must contain 'statement' and 'status' columns.")

def process_data():
    global df, X_train, X_test, y_train, y_test
    if df is None:
        messagebox.showerror("Error", "Please load a CSV file first.")
        return
    
    df['statement'] = df['statement'].fillna('')  # Handle missing data
    df['processed_text'] = df['statement'].apply(preprocess_text)
    
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(df['processed_text'])
    y = df['status']

    # Split the data into 80% training and 20% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train RandomForest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Train DecisionTree model
    dt_model = DecisionTreeClassifier(random_state=42)  # Decision Tree classifier
    dt_model.fit(X_train, y_train)

    # Save trained models and vectorizer
    global trained_rf_model, trained_vectorizer, trained_dt_model
    trained_rf_model = rf_model
    trained_vectorizer = vectorizer
    trained_dt_model = dt_model

    # Display training/testing split info
    result_text.delete(1.0, tk.END)  # Clear previous output
    result_text.insert(tk.END, f"Data split: 80% for training, 20% for testing.\n")

def predict_rf():
    if trained_rf_model is None or trained_vectorizer is None:
        messagebox.showwarning("Model Error", "Please load data and train the model first.")
        return

    # Get the RandomForest model accuracy on the test set
    rf_accuracy = trained_rf_model.score(X_test, y_test)
    
    # Display the accuracy of the RandomForest model
    result_text.delete(1.0, tk.END)  # Clear previous output
    result_text.insert(tk.END, f"RandomForest Accuracy on Test Set: {rf_accuracy * 100:.2f}%\n")

def predict_dt():
    if trained_dt_model is None or trained_vectorizer is None:
        messagebox.showwarning("Model Error", "Please load data and train the model first.")
        return

    # Get the Decision Tree model accuracy on the test set
    dt_accuracy = trained_dt_model.score(X_test, y_test)
    
    # Display the accuracy of the Decision Tree model
    result_text.delete(1.0, tk.END)  # Clear previous output
    result_text.insert(tk.END, f"Decision Tree Accuracy on Test Set: {dt_accuracy * 100:.2f}%\n")

def predict_statement():
    statement = text_entry.get("1.0", tk.END).strip()
    if not statement:
        messagebox.showwarning("Input Error", "Please enter a statement for prediction.")
        return

    processed_statement = preprocess_text(statement)
    statement_vector = trained_vectorizer.transform([processed_statement])
    
    # Predict using RandomForest model
    rf_prediction = trained_rf_model.predict(statement_vector)[0]
    rf_prob = trained_rf_model.predict_proba(statement_vector)[0]
    rf_pred_prob = max(rf_prob) * 100

    # Predict using DecisionTree model
    dt_prediction = trained_dt_model.predict(statement_vector)[0]
    dt_prob = trained_dt_model.predict_proba(statement_vector)[0]
    dt_pred_prob = max(dt_prob) * 100

    # Display the results in the result_text box
    result_text.delete(1.0, tk.END)  # Clear previous output
    result_text.insert(tk.END, f"Prediction for Statement: \n")
    result_text.insert(tk.END, f"RandomForest Prediction: {rf_prediction}, Probability: {rf_pred_prob:.2f}%\n")
    result_text.insert(tk.END, f"DecisionTree Prediction: {dt_prediction}, Probability: {dt_pred_prob:.2f}%\n")

# Section 5: Tkinter GUI Setup
def create_gui():
    global text_entry, result_text  # Declare as global to access in functions

    # Tkinter window setup
    root = tk.Tk()
    root.title("Mental Health Disorder Detection")
    root.geometry("800x600")
    root.config(bg="#9B1B30")  # Set background color to merone (hex code)

    # Title Label with background color
    # Title Label with background color and spacing
    title_label = tk.Label(root, 
                       text="Detection and Prediction of Future Mental Disorder from social media data using ML, Ensemble learning and LLM", 
                       font=("Arial", 17, "bold"), 
                       bg="white", 
                       fg="black", 
                       padx=20,  # Horizontal padding (left and right space)
                       pady=20)  # Vertical padding (top and bottom space)
    title_label.pack(pady=20)

    

    # Frame for Buttons (Left aligned)
    button_frame = tk.Frame(root, bg="#9B1B30")
    button_frame.pack(side=tk.LEFT, padx=20, pady=20, anchor="n")

    # Upload CSV button
    load_button = tk.Button(button_frame, text="Load CSV File", font=("Arial", 12), command=load_csv)
    load_button.pack(pady=10, anchor="w")

    # Train and Test button
    train_button = tk.Button(button_frame, text="Train and Test", font=("Arial", 12), command=process_data)
    train_button.pack(pady=10, anchor="w")

    # Run RandomForest model button
    rf_button = tk.Button(button_frame, text="Run RandomForest Algorithm", font=("Arial", 12), command=predict_rf)
    rf_button.pack(pady=10, anchor="w")

    # Run Decision Tree model button
    dt_button = tk.Button(button_frame, text="Run Decision Tree Algorithm", font=("Arial", 12), command=predict_dt)
    dt_button.pack(pady=10, anchor="w")

    # Predict button (new)
    predict_button = tk.Button(button_frame, text="Predict", font=("Arial", 12), command=predict_statement)
    predict_button.pack(pady=10, anchor="w")

    # Frame for Textbox (Right aligned)
    text_frame = tk.Frame(root, bg="#9B1B30")
    text_frame.pack(side=tk.RIGHT, padx=20, pady=20, anchor="n")

    # Statement input field for prediction
    text_entry_label = tk.Label(text_frame, text="Enter Statement for Prediction:", font=("Arial", 13), bg="#9B1B30", fg="white")
    text_entry_label.pack()

    text_entry = tk.Text(text_frame, height=6, width=50, font=("Arial", 12))
    text_entry.pack(pady=10)

    # Result Text Box (single box for all outputs)
    result_text = tk.Text(text_frame, height=10, width=50, font=("Arial", 12), bg="#f5f5f5")
    result_text.pack(pady=10)

    # Start the Tkinter event loop
    root.mainloop()

# Main function to initialize GUI
if __name__ == "__main__":
    create_gui()
