# Telecom Customer Churn Predictor

A full-stack machine learning application that predicts customer churn in telecommunications using an Artificial Neural Network (ANN) built with PyTorch. The project includes both backend API and frontend web interface for real-time churn prediction.

## 🔗 Live Demo
**Deployed Application**: [telecom-churn-predictor-1.onrender.com](https://telecom-churn-predictor-1.onrender.com)

## 📊 Project Overview

This project addresses the critical business problem of customer churn prediction in the telecom industry. By analyzing customer demographics, service usage, and billing information, the model identifies customers who are likely to discontinue their service, enabling proactive retention strategies.

### Key Features
- **Machine Learning Model**: Custom ANN implemented with PyTorch
- **REST API**: FastAPI backend with real-time prediction endpoints
- **Web Interface**: React.js frontend for user-friendly interaction
- **Advanced Training**: Includes Focal Loss, early stopping, and learning rate scheduling
- **Model Deployment**: TorchScript model serialization for production deployment

## 🏗️ Project Structure

```
telecom-churn-predictor/
├── backend/
│   ├── weights/
│   │   ├── churn_ann.pt          # Trained PyTorch model
│   │   └── scaler.pkl            # Feature scaler
│   ├── model.ipynb               # Model training notebook
│   ├── pre-processing.ipynb      # Data preprocessing notebook
│   ├── main.py                   # FastAPI application
│   ├── customer_churn.csv        # Original dataset
│   ├── preprocessed_churn.csv    # Cleaned dataset
│   └── requirements.txt          # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── App.jsx              # Main React component
│   │   └── App.css              # Styling
│   ├── package.json             # Node.js dependencies
│   └── index.html               # Entry point
└── README.md
```

## 🧠 Machine Learning Pipeline

### Dataset Features
The model uses **19 input features** including:
- **Demographics**: Gender, Senior Citizen status, Partner, Dependents
- **Service Details**: Phone Service, Multiple Lines, Internet Service type
- **Account Information**: Contract type, Payment method, Billing preferences
- **Usage Metrics**: Tenure, Monthly Charges, Total Charges
- **Add-on Services**: Online Security, Backup, Device Protection, Tech Support, Streaming services

### Data Preprocessing
- **Data Cleaning**: Removed rows with missing `TotalCharges` (11 rows)
- **Feature Engineering**: 
  - Binary encoding for Yes/No fields
  - One-hot encoding for categorical variables (Internet Service, Contract, Payment Method)
  - MinMax scaling for numerical features (tenure, charges)
- **Final Dataset**: 7,032 samples with 26 engineered features

### Model Architecture
**Advanced Neural Network Design**:
```
Input Layer: 26 features
├── Dense(64) + BatchNorm + ReLU + Dropout(0.3)
├── Dense(32) + BatchNorm + ReLU + Dropout(0.3)  
├── Dense(16) + BatchNorm + SELU + Dropout(0.2)
├── Dense(8) + BatchNorm + SELU + Dropout(0.2)
└── Dense(1) + Sigmoid
```

### Training Configuration
- **Loss Function**: Focal Loss (α=1.0, γ=2.0) for handling class imbalance
- **Optimizer**: AdamW with weight decay (1e-4)
- **Learning Rate**: Warm-up + Cosine Annealing scheduler
- **Training Strategy**: Weighted sampling for balanced training
- **Early Stopping**: Patience of 10 epochs on validation loss

## 📈 Model Performance

### Final Results
- **Validation Accuracy**: 78.89%
- **ROC-AUC Score**: 0.8286

### Confusion Matrix
```
                Predicted
Actual    No Churn  Churn
No Churn     867     166
Churn        131     243
```

### Class-wise Performance
- **Class 0 (No Churn)**: Precision 0.87, Recall 0.84, F1-Score 0.85
- **Class 1 (Churn)**: Precision 0.69, Recall 0.84, F1-Score 0.79

## 🚀 Technology Stack

### Backend
- **Framework**: FastAPI
- **ML Library**: PyTorch
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Model Serialization**: TorchScript, Joblib
- **Server**: Uvicorn

### Frontend
- **Framework**: React.js 19.1.1
- **Build Tool**: Vite
- **Language**: JavaScript/JSX
- **Styling**: CSS3

## 🛠️ Installation & Setup

### Backend Setup
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

### Docker Deployment (Optional)
The application can be containerized and deployed on cloud platforms like Render, Heroku, or AWS.

## 📝 API Documentation

### Prediction Endpoint
**POST** `/predict`

**Request Body**:
```json
{
  "gender": "Male",
  "SeniorCitizen": 1,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 43,
  "PhoneService": "Yes",
  "MultipleLines": "Yes",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "Yes",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 103.0,
  "TotalCharges": 4414.3
}
```

**Response**:
```json
{
  "churn_probability": 0.7234,
  "prediction": "Churn"
}
```

## 💡 Key Technical Achievements

1. **Advanced Architecture**: Implemented sophisticated neural network with batch normalization and multiple activation functions
2. **Robust Training**: Used Focal Loss and weighted sampling to handle class imbalance effectively
3. **Production Ready**: TorchScript model serialization enables efficient inference
4. **Full-Stack Integration**: Seamless communication between React frontend and FastAPI backend
5. **Real-time Predictions**: Sub-second prediction response times
6. **Scalable Design**: Modular architecture supports easy model updates and feature additions

## 🔍 Business Impact

This solution enables telecom companies to:
- **Identify At-Risk Customers**: Proactively detect customers likely to churn
- **Optimize Retention Strategies**: Target high-risk customers with personalized offers
- **Reduce Revenue Loss**: Prevent customer attrition through timely interventions
- **Improve Customer Experience**: Address service issues before customers leave

## 🚀 Future Enhancements

- **Model Improvements**: Experiment with ensemble methods and feature selection
- **Real-time Monitoring**: Implement model drift detection and retraining pipeline  
- **Advanced Analytics**: Add customer segmentation and lifetime value prediction
- **Mobile App**: Develop mobile interface for field sales teams
- **A/B Testing**: Integrate framework for testing retention strategies

## 📊 Dataset Information

- **Source**: Telecom customer data with churn labels
- **Size**: 7,032 customers after cleaning
- **Features**: 19 original features expanded to 26 engineered features
- **Target Distribution**: Imbalanced dataset with minority churn class


