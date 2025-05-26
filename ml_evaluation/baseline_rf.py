import os
import glob
import logging
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
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

        le_neigh = LabelEncoder()
        self.df['neighborhood_encoded'] = le_neigh.fit_transform(self.df['analysis_neighborhood'].fillna('Unknown'))
        self.df['district_encoded'] = LabelEncoder().fit_transform(self.df['police_district'].fillna("Unknown"))

        label_counts = self.df['incident_category'].value_counts()
        min_samples = 20  # 增加最小樣本數
        valid_labels = label_counts[label_counts >= min_samples].index
        
        logger.info(f"Original categories: {len(label_counts)}")
        logger.info(f"Categories with >= {min_samples} samples: {len(valid_labels)}")
        
        self.df = self.df[self.df['incident_category'].isin(valid_labels)]

        le_label = LabelEncoder()
        self.df['label'] = le_label.fit_transform(self.df['incident_category'])

        self.df['supervisor_district'] = self.df['supervisor_district'].fillna(-1).astype(int)

        features = [
            'incident_hour',
            'incident_day_of_week',
            'incident_month',
            'latitude',
            'longitude',
            'neighborhood_encoded',
            'district_encoded',
            'supervisor_district'
        ]

        X = self.df[features]
        y = self.df['label']

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
        

        try:
            min_class_size = pd.Series(y_train_filtered).value_counts().min()
            k_neighbors = min(5, min_class_size - 1)
            
            logger.info(f"Using k_neighbors={k_neighbors} for SMOTE")
            sm = SMOTE(random_state=2, k_neighbors=k_neighbors)
            X_train_res, y_train_res = sm.fit_resample(X_train_filtered, y_train_filtered)
            
            logger.info(f"After SMOTE: {len(X_train_res)} training samples")
            
        except Exception as e:
            logger.warning(f"SMOTE failed: {e}. Training without oversampling.")
            X_train_res, y_train_res = X_train_filtered, y_train_filtered

        clf = RandomForestClassifier(n_estimators=100, random_state=2, class_weight='balanced')
        clf.fit(X_train_res, y_train_res)
        y_pred = clf.predict(X_test_filtered)

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

if __name__ == "__main__":
    try:
        rf_base = RandomForestBaseline()
        X_train, X_test, y_train, y_test = rf_base.preprocess_data()
        rf_base.train_model(X_train, X_test, y_train, y_test)
    except Exception as e:
        logger.error(f"Error in main: {e}")
        import traceback
        traceback.print_exc()