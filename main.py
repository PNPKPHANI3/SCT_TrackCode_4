# main.py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from preprocess import load_and_preprocess_data, split_data
from train_model import train_svm, save_model
from evaluate import evaluate_model, plot_confusion_matrix

# 📁 Paths to the dataset
train_csv = 'data/sign_mnist_train.csv'
test_csv = 'data/sign_mnist_test.csv'

# 🔄 Load and preprocess the dataset
print("📥 Loading and preprocessing data...")
X_train_full, X_test, y_train_full, y_test = load_and_preprocess_data(train_csv, test_csv)
X_train, X_val, y_train, y_val = split_data(X_train_full, y_train_full)

# 🎯 Train the SVM model
print("🧠 Training the SVM model...")
model = train_svm(X_train, y_train)

# 💾 Save the trained model
print("💾 Saving the model...")
save_model(model, 'models/svm_model.pkl')

# 📊 Evaluate the model
print("🔍 Evaluating the model...")
conf_matrix = evaluate_model(model, X_val, y_val)

# 📉 Plot confusion matrix
print("🖼️ Saving confusion matrix plot...")
plot_confusion_matrix(conf_matrix, classes=[chr(i) for i in range(65, 91) if i != 74])  # A–Z excluding J

print("✅ Task Completed Successfully!")
