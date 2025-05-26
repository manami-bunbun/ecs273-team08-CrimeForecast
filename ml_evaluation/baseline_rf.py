import os
import glob
import logging
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import validation_curve, learning_curve
from sklearn.metrics import classification_report, confusion_matrix, log_loss
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RandomForestBaseline:
    def __init__(self):
        csv_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "csv")
        csv_files = glob.glob(os.path.join(csv_dir, "sf_crime_*.csv.gz"))

        if csv_files:
            self.csv_path = max(csv_files, key=os.path.getctime)
            logger.info(f"Using CSV file: {self.csv_path}")
        else:
            self.csv_path = None
            logger.warning("No CSV files found in data/csv directory")

        if self.csv_path and os.path.exists(self.csv_path):
            self.df = pd.read_csv(self.csv_path)
            logger.info(f"Successfully loaded CSV data from {self.csv_path}")
        else:
            logger.error("CSV file not found or not accessible")
            self.df = None

    def preprocess_data(self):
        if self.df is None:
            raise ValueError("Data not loaded.")

        self.df['incident_datetime'] = pd.to_datetime(self.df['incident_datetime'], errors='coerce')
        self.df = self.df.dropna(subset=['incident_category', 'latitude', 'longitude', 'incident_datetime'])

        self.df['incident_hour'] = self.df['incident_datetime'].dt.hour
        self.df['incident_day_of_week'] = self.df['incident_datetime'].dt.weekday
        self.df['incident_month'] = self.df['incident_datetime'].dt.month

        self.df['police_district'] = LabelEncoder().fit_transform(self.df['police_district'].fillna('Unknown'))
        self.df['analysis_neighborhood'] = LabelEncoder().fit_transform(self.df['analysis_neighborhood'].fillna("Unknown"))

        label_counts = self.df['incident_category'].value_counts()
        min_samples = 20 
        valid_labels = label_counts[label_counts >= min_samples].index
        
        logger.info(f"Original categories: {len(label_counts)}")
        logger.info(f"Categories with >= {min_samples} samples: {len(valid_labels)}")
        
        self.df = self.df[self.df['incident_category'].isin(valid_labels)]

        le_label = LabelEncoder()
        self.df['incident_category'] = le_label.fit_transform(self.df['incident_category'])

        self.df['supervisor_district'] = self.df['supervisor_district'].fillna(-1).astype(int)

        features = [
            'incident_hour',
            'incident_day_of_week',
            'incident_month',
            'latitude',
            'longitude',
            'police_district',
            'analysis_neighborhood',
            'supervisor_district'
        ]

        X = self.df[features]
        y = self.df['incident_category']

        self.label_encoder = le_label
        return train_test_split(X, y, test_size=0.3, random_state=2, stratify=y)

    def train_model(self, X_train, X_test, y_train, y_test):
        train_counts = pd.Series(y_train).value_counts()
        logger.info("Training set class distribution:")
        for label, count in train_counts.items():
            category_name = self.label_encoder.classes_[label]
            logger.info(f"  {category_name}: {count} samples")
        
        min_train_samples = 20
        valid_train_labels = train_counts[train_counts >= min_train_samples].index
        
        train_mask = y_train.isin(valid_train_labels)
        test_mask = y_test.isin(valid_train_labels)
        
        X_train_filtered = X_train[train_mask]
        y_train_filtered = y_train[train_mask]
        X_test_filtered = X_test[test_mask]
        y_test_filtered = y_test[test_mask]
        
        logger.info(f"Filtered training samples: {len(X_train_filtered)} (from {len(X_train)})")
        logger.info(f"Filtered test samples: {len(X_test_filtered)} (from {len(X_test)})")
        
        # SMOTE 處理
        try:
            min_class_size = pd.Series(y_train_filtered).value_counts().min()
            k_neighbors = min(10, min_class_size - 1)
            
            logger.info(f"Using k_neighbors={k_neighbors} for SMOTE")
            sm = SMOTE(random_state=2, k_neighbors=k_neighbors)
            X_train_res, y_train_res = sm.fit_resample(X_train_filtered, y_train_filtered)
            
            logger.info(f"After SMOTE: {len(X_train_res)} training samples")
            
        except Exception as e:
            logger.warning(f"SMOTE failed: {e}. Training without oversampling.")
            X_train_res, y_train_res = X_train_filtered, y_train_filtered

        # 1. Learning Curve - Shows how the model performs as more training samples are added
        logger.info("Generating learning curves...")
        train_sizes, train_scores, val_scores = learning_curve(
            RandomForestClassifier(n_estimators=100, random_state=2, class_weight='balanced', n_jobs=-1),
            X_train_res, y_train_res,
            cv=5,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='accuracy',
            random_state=2
        )
        
        # 2. Validation curves - showing the impact of different n_estimators on model performance
        logger.info("Generating validation curves...")
        param_range = [10, 25, 50, 75, 100, 150, 200, 300]
        train_scores_val, test_scores_val = validation_curve(
            RandomForestClassifier(random_state=2, class_weight='balanced', n_jobs=-1),
            X_train_res, y_train_res,
            param_name='n_estimators',
            param_range=param_range,
            cv=5,
            scoring='accuracy',
            random_state=2
        )
        
        # 3. Track the loss during training
        logger.info("Training model with loss tracking...")
        
        # To keep track of the loss, we need to use the predict_proba method.
        clf = RandomForestClassifier(n_estimators=100, random_state=2, class_weight='balanced', n_jobs=-1)
        
        # Training in batches to track progress (simulating epochs)
        epoch_losses = []
        epoch_val_losses = []
        epochs = []
        
        # Use different n_estimators to simulate epochs
        n_estimators_range = range(10, 101, 10)  # 10, 20, 30, ..., 100
        
        for n_est in n_estimators_range:
            # Train model
            temp_clf = RandomForestClassifier(
                n_estimators=n_est, 
                random_state=2, 
                class_weight='balanced',
                n_jobs=-1
            )
            temp_clf.fit(X_train_res, y_train_res)
            
            # Compute loss
            train_proba = temp_clf.predict_proba(X_train_res)
            val_proba = temp_clf.predict_proba(X_test_filtered)
            
            # Use log loss as loss function
            train_loss = log_loss(y_train_res, train_proba)
            val_loss = log_loss(y_test_filtered, val_proba)
            
            epoch_losses.append(train_loss)
            epoch_val_losses.append(val_loss)
            epochs.append(n_est)
            
            logger.info(f"n_estimators={n_est}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        
        # Last modek we used
        clf.fit(X_train_res, y_train_res)
        y_pred = clf.predict(X_test_filtered)

        # Plot the training curves
        self.plot_training_curves(
            train_sizes, train_scores, val_scores,
            param_range, train_scores_val, test_scores_val,
            epochs, epoch_losses, epoch_val_losses
        )
        
        # Evaluation
        labels = np.unique(y_test_filtered)
        target_names = [self.label_encoder.classes_[i] for i in labels]

        logger.info("Classification Report:")
        print(classification_report(y_test_filtered, y_pred, labels=labels, target_names=target_names, zero_division=0))

        cm = confusion_matrix(y_test_filtered, y_pred, labels=labels)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, cmap="Blues", xticklabels=target_names, yticklabels=target_names, annot=False)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        os.makedirs("outputs", exist_ok=True)
        plt.savefig("outputs/rf_confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Balanced model evaluation completed and confusion matrix saved.")

    def plot_training_curves(self, train_sizes, train_scores, val_scores, 
                            param_range, train_scores_val, test_scores_val,
                            epochs, epoch_losses, epoch_val_losses):
        """繪製訓練曲線"""
        
        # Setting
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Training Analysis', fontsize=16, fontweight='bold')
        
        # 1. Learning Curve
        ax1 = axes[0, 0]
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        ax1.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Accuracy')
        ax1.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
        ax1.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Accuracy')
        ax1.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
        
        ax1.set_xlabel('Training Set Size')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Learning Curve')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Evaluation Curve
        ax2 = axes[0, 1]
        train_mean_val = np.mean(train_scores_val, axis=1)
        train_std_val = np.std(train_scores_val, axis=1)
        test_mean_val = np.mean(test_scores_val, axis=1)
        test_std_val = np.std(test_scores_val, axis=1)
        
        ax2.plot(param_range, train_mean_val, 'o-', color='blue', label='Training Accuracy')
        ax2.fill_between(param_range, train_mean_val - train_std_val, train_mean_val + train_std_val, alpha=0.1, color='blue')
        ax2.plot(param_range, test_mean_val, 'o-', color='red', label='Validation Accuracy')
        ax2.fill_between(param_range, test_mean_val - test_std_val, test_mean_val + test_std_val, alpha=0.1, color='red')
        
        ax2.set_xlabel('Number of Estimators')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Validation Curve (n_estimators)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Loss Curve
        ax3 = axes[1, 0]
        ax3.plot(epochs, epoch_losses, '-', color='blue', linewidth=2, label='Training Loss')
        ax3.plot(epochs, epoch_val_losses, '-', color='red', linewidth=2, label='Validation Loss')
        ax3.set_xlabel('Epoch (n_estimators)')
        ax3.set_ylabel('Loss')
        ax3.set_title('Cross-validation Loss of the Model')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(bottom=0)
        
        # 4. Loss Difference
        ax4 = axes[1, 1]
        loss_diff = np.array(epoch_val_losses) - np.array(epoch_losses)
        ax4.plot(epochs, loss_diff, '-', color='green', linewidth=2)
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax4.set_xlabel('Epoch (n_estimators)')
        ax4.set_ylabel('Validation Loss - Training Loss')
        ax4.set_title('Overfitting Indicator')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        os.makedirs("outputs", exist_ok=True)
        plt.savefig("outputs/training_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Training analysis plots saved to outputs/training_analysis.png")
        
        # Loss Curve
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, epoch_val_losses, '-', color='#2E86AB', linewidth=2.5)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Cross-validation loss of the base model', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.ylim(bottom=0)
        
        # Set x 
        plt.xticks(range(0, max(epochs)+1, 10))
        
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_color('#CCCCCC')
        plt.gca().spines['bottom'].set_color('#CCCCCC')
        
        plt.tight_layout()
        plt.savefig("outputs/cv_loss_curve.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Cross-validation loss curve saved to outputs/cv_loss_curve.png")

if __name__ == "__main__":
    try:
        rf_base = RandomForestBaseline()
        X_train, X_test, y_train, y_test = rf_base.preprocess_data()
        rf_base.train_model(X_train, X_test, y_train, y_test)
    except Exception as e:
        logger.error(f"Error in main: {e}")
        import traceback
        traceback.print_exc()