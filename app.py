# Section 1: Importing Libraries
import tkinter as tk
from tkinter import messagebox, filedialog
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import re
import nltk

# Section 2: NLTK Stop Words Setup
try:
    stop_words = set(nltk.corpus.stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(nltk.corpus.stopwords.words('english'))

# Section 3: Global Variables
trained_rf_model = None
trained_vectorizer = None
trained_dt_model = None
df = None
text_entry = None
result_text = None
X_train, X_test, y_train, y_test = None, None, None, None

# Section 4: Functions for Data Preprocessing and Model Prediction
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^A-Za-z\s]", "", text)
    text = text.lower()
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

def load_csv():
    global df
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if file_path:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.lower()  # normalize column names
        if 'statement' in df.columns and 'status' in df.columns:
            messagebox.showinfo("Success", "CSV loaded successfully.")
        else:
            messagebox.showerror("Error", "CSV must contain 'statement' and 'status' columns.")

def process_data():
    global df, X_train, X_test, y_train, y_test
    if df is None:
        messagebox.showerror("Error", "Please load a CSV file first.")
        return
    
    df['statement'] = df['statement'].fillna('')
    df['processed_text'] = df['statement'].apply(preprocess_text)
    
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(df['processed_text'])
    y = df['status']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)

    global trained_rf_model, trained_vectorizer, trained_dt_model
    trained_rf_model = rf_model
    trained_vectorizer = vectorizer
    trained_dt_model = dt_model

    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, f"Data split: 80% training, 20% testing.\n")

def predict_rf():
    if trained_rf_model is None or trained_vectorizer is None:
        messagebox.showwarning("Model Error", "Please load data and train the model first.")
        return
    rf_accuracy = trained_rf_model.score(X_test, y_test)
    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, f"RandomForest Accuracy: {rf_accuracy * 100:.2f}%\n")

def predict_dt():
    if trained_dt_model is None or trained_vectorizer is None:
        messagebox.showwarning("Model Error", "Please load data and train the model first.")
        return
    dt_accuracy = trained_dt_model.score(X_test, y_test)
    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, f"Decision Tree Accuracy: {dt_accuracy * 100:.2f}%\n")

def predict_statement():
    if trained_rf_model is None or trained_dt_model is None or trained_vectorizer is None:
        messagebox.showwarning("Model Error", "Please train the models first.")
        return

    statement = text_entry.get("1.0", tk.END).strip()
    if not statement:
        messagebox.showwarning("Input Error", "Please enter a statement for prediction.")
        return

    processed_statement = preprocess_text(statement)
    statement_vector = trained_vectorizer.transform([processed_statement])
    
    rf_prediction = trained_rf_model.predict(statement_vector)[0]
    rf_prob = trained_rf_model.predict_proba(statement_vector)[0]
    rf_pred_prob = max(rf_prob) * 100

    dt_prediction = trained_dt_model.predict(statement_vector)[0]
    dt_prob = trained_dt_model.predict_proba(statement_vector)[0]
    dt_pred_prob = max(dt_prob) * 100

    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, f"Prediction:\n")
    result_text.insert(tk.END, f"RandomForest → {rf_prediction}, Prob: {rf_pred_prob:.2f}%\n")
    result_text.insert(tk.END, f"DecisionTree → {dt_prediction}, Prob: {dt_pred_prob:.2f}%\n")

# Section 5: Tkinter GUI Setup
def create_gui():
    global text_entry, result_text

    root = tk.Tk()
    root.title("Mental Health Disorder Detection")
    root.geometry("800x600")
    root.config(bg="#9B1B30")

    title_label = tk.Label(root, 
        text="Detection and Prediction of Future Mental Disorder\nfrom social media data using ML, Ensemble learning and LLM", 
        font=("Arial", 16, "bold"), bg="white", fg="black", padx=20, pady=20)
    title_label.pack(pady=20)

    button_frame = tk.Frame(root, bg="#9B1B30")
    button_frame.pack(side=tk.LEFT, padx=20, pady=20, anchor="n")

    tk.Button(button_frame, text="Load CSV File", font=("Arial", 12), command=load_csv).pack(pady=10, anchor="w")
    tk.Button(button_frame, text="Train and Test", font=("Arial", 12), command=process_data).pack(pady=10, anchor="w")
    tk.Button(button_frame, text="Run RandomForest", font=("Arial", 12), command=predict_rf).pack(pady=10, anchor="w")
    tk.Button(button_frame, text="Run Decision Tree", font=("Arial", 12), command=predict_dt).pack(pady=10, anchor="w")
    tk.Button(button_frame, text="Predict", font=("Arial", 12), command=predict_statement).pack(pady=10, anchor="w")

    text_frame = tk.Frame(root, bg="#9B1B30")
    text_frame.pack(side=tk.RIGHT, padx=20, pady=20, anchor="n")

    tk.Label(text_frame, text="Enter Statement for Prediction:", font=("Arial", 13), bg="#9B1B30", fg="white").pack()
    text_entry = tk.Text(text_frame, height=6, width=50, font=("Arial", 12))
    text_entry.pack(pady=10)

    result_text = tk.Text(text_frame, height=10, width=50, font=("Arial", 12), bg="#f5f5f5")
    result_text.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    create_gui()
