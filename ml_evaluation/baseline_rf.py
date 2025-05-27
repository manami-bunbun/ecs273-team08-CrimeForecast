import os
import glob
import logging
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score
from imblearn.over_sampling import SMOTE
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RandomForestPipeline:
    def __init__(self):
        csv_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "csv")
        csv_files = glob.glob(os.path.join(csv_dir, "sf_crime_*.csv.gz"))

        if csv_files:
            self.csv_path = max(csv_files, key=os.path.getctime)
            logger.info(f"Using CSV file: {self.csv_path}")
        else:
            raise FileNotFoundError("No CSV files found in data/csv directory")

        self.df = pd.read_csv(self.csv_path)
        logger.info("CSV file loaded successfully")

    def categorize_crime_violent(self, category):
        violent = {
            'Assault', 'Robbery', 'Homicide', 'Rape',
            'Weapons Offense', 'Weapons Carrying Etc',
            'Sex Offense', 'Offences Against The Family And Children',
            'Suicide', 'Human Trafficking (A), Commercial Sex Acts',
            'Human Trafficking, Commercial Sex Acts'
        }
        return 'Violent' if category in violent else 'Non-Violent'

    def categorize_crime_occupation(self, category):
        blue_collar = {
            'Assault', 'Robbery', 'Burglary', 'Motor Vehicle Theft', 'Larceny Theft',
            'Malicious Mischief', 'Vandalism', 'Drug Offense',
            'Weapons Offense', 'Weapons Carrying Etc', 'Homicide', 'Rape', 'Sex Offense',
            'Disorderly Conduct', 'Prostitution', 'Suicide',
            'Human Trafficking (A), Commercial Sex Acts', 'Human Trafficking, Commercial Sex Acts',
            'Traffic Violation Arrest', 'Traffic Collision', 'Arson', 'Stolen Property',
            'Civil Sidewalks', 'Suspicious', 'Recovered Vehicle'
        }
        return 'Blue-Collar' if category in blue_collar else 'White-Collar'

    def create_super_categories(self):
        mapping = {
            'Property Crime': ['Larceny Theft', 'Burglary', 'Motor Vehicle Theft', 'Robbery'],
            'Violence': ['Assault', 'Homicide', 'Rape'],
            'Drugs': ['Drug Offense', 'Drug Violation'],
            'Public Order': ['Disorderly Conduct', 'Vandalism'],
            'Administrative': ['Non-Criminal', 'Other Offenses'],
            'Family': ['Suicide', 'Human Trafficking']
        }
        reverse_map = {cat: super_cat for super_cat, cats in mapping.items() for cat in cats}
        self.df['crime_type_super'] = self.df['incident_category'].map(reverse_map).fillna('Administrative')
        self.df['crime_type_violent'] = self.df['incident_category'].apply(self.categorize_crime_violent)
        self.df['crime_type_occupation'] = self.df['incident_category'].apply(self.categorize_crime_occupation)

    def create_features(self):
        self.df['incident_datetime'] = pd.to_datetime(self.df['incident_datetime'], errors='coerce')
        self.df.dropna(subset=['incident_datetime', 'latitude', 'longitude'], inplace=True)
        self.df['hour'] = self.df['incident_datetime'].dt.hour
        self.df['month'] = self.df['incident_datetime'].dt.month
        self.df['dayofweek'] = self.df['incident_datetime'].dt.weekday

        le_neigh = LabelEncoder()
        self.df['neigh_enc'] = le_neigh.fit_transform(self.df['analysis_neighborhood'].fillna('Unknown'))

        self.df['geo_grid'] = pd.cut(self.df['latitude'], bins=10, labels=False) * 10 + pd.cut(self.df['longitude'], bins=10, labels=False)

    def run_multiple_targets(self, label_cols):
        results = []
        
        for label_col in label_cols:
            logger.info(f"\n=== Training with label: {label_col} ===")
            
            try:
                X_train, X_test, y_train, y_test, label_encoder = self.preprocess(label_col)

                # Check class distribution and decide whether to use SMOTE
                class_counts = Counter(y_train)
                min_samples = min(class_counts.values())
                
                logger.info(f"Class distribution: {dict(class_counts)}")
                logger.info(f"Smallest class has {min_samples} samples")

                # Only use SMOTE if we have enough samples and class imbalance
                use_smote = False
                if min_samples >= 6:  # Need at least 6 samples for k_neighbors=5
                    max_samples = max(class_counts.values())
                    imbalance_ratio = max_samples / min_samples
                    if imbalance_ratio > 2:  # Only apply SMOTE if there's significant imbalance
                        use_smote = True
                        smote_k = min(5, min_samples - 1)
                        logger.info(f"Applying SMOTE with k_neighbors={smote_k} (imbalance ratio: {imbalance_ratio:.2f})")
                        sm = SMOTE(random_state=42, k_neighbors=smote_k)
                        X_train_balanced, y_train_balanced = sm.fit_resample(X_train, y_train)
                    else:
                        logger.info(f"Skipping SMOTE - classes are relatively balanced (ratio: {imbalance_ratio:.2f})")
                else:
                    logger.info(f"Skipping SMOTE - insufficient samples in smallest class ({min_samples} < 6)")

                if not use_smote:
                    X_train_balanced, y_train_balanced = X_train, y_train

                start = time.time()
                model, metrics = self.train_and_evaluate(
                    X_train_balanced, X_test, y_train_balanced, y_test, 
                    label_encoder, suffix=label_col
                )
                duration = time.time() - start

                metrics['label'] = label_col
                metrics['training_time_sec'] = duration
                metrics['used_smote'] = use_smote
                results.append(metrics)
                
            except Exception as e:
                logger.error(f"Error processing label '{label_col}': {str(e)}")
                # Add a placeholder result to maintain consistency
                results.append({
                    'label': label_col,
                    'accuracy': 0.0,
                    'f1_score': 0.0,
                    'precision': 0.0,
                    'training_time_sec': 0.0,
                    'used_smote': False,
                    'error': str(e)
                })

        df_results = pd.DataFrame(results)
        os.makedirs("outputs", exist_ok=True)
        df_results.to_csv("outputs/model_comparison.csv", index=False)
        logger.info("\n=== Summary ===\n" + str(df_results))

    def preprocess(self, target_col):
        self.create_super_categories()
        self.create_features()

        features = ['hour', 'month', 'dayofweek', 'latitude', 'longitude', 'neigh_enc', 'geo_grid']
        X = self.df[features]
        y_raw = self.df[target_col]

        # Filter out classes with very few samples - increase minimum to ensure stratification works
        class_counts = y_raw.value_counts()
        min_samples_per_class = max(10, int(len(y_raw) * 0.3 * 0.01))  # At least 10 or 1% of 30% of data
        valid_classes = class_counts[class_counts >= min_samples_per_class].index
        
        logger.info(f"Original classes: {len(class_counts)}")
        logger.info(f"Min samples per class: {min_samples_per_class}")
        logger.info(f"Valid classes (>={min_samples_per_class} samples): {len(valid_classes)}")
        
        if len(valid_classes) < 2:
            # Fallback: use classes with at least 6 samples
            min_samples_per_class = 6
            valid_classes = class_counts[class_counts >= min_samples_per_class].index
            logger.warning(f"Fallback: Using classes with >={min_samples_per_class} samples: {len(valid_classes)}")
            
            if len(valid_classes) < 2:
                raise ValueError(f"Not enough valid classes for classification. Only {len(valid_classes)} classes have >={min_samples_per_class} samples.")
        
        # Filter data to only include valid classes
        mask = y_raw.isin(valid_classes)
        X = X[mask]
        y_raw = y_raw[mask]

        # Encode labels
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        logger.info(f"Using label column '{target_col}': {len(le.classes_)} classes")
        logger.info(f"Class distribution after filtering: {pd.Series(y).value_counts().sort_index().to_dict()}")
        self.label_encoder = le

        # Feature selection
        selector = SelectKBest(score_func=mutual_info_classif, k='all')
        X_new = selector.fit_transform(X, y)
        self.selected_features = [features[i] for i in selector.get_support(indices=True)]

        # Use stratified split with better error handling
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_new, y, test_size=0.3, random_state=42 + attempt, stratify=y
                )
                
                # Verify that all classes are present in both splits
                train_classes = set(y_train)
                test_classes = set(y_test)
                all_classes = set(y)
                
                if train_classes == all_classes and len(test_classes & all_classes) >= len(all_classes) * 0.8:
                    logger.info(f"Successful stratified split on attempt {attempt + 1}")
                    break
                else:
                    logger.warning(f"Attempt {attempt + 1}: Some classes missing in splits")
                    if attempt == max_attempts - 1:
                        raise ValueError("Classes not properly distributed")
                        
            except ValueError as e:
                logger.warning(f"Stratified split attempt {attempt + 1} failed: {e}")
                if attempt == max_attempts - 1:
                    logger.warning("Using random split as final fallback")
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_new, y, test_size=0.3, random_state=42
                    )
        
        return X_train, X_test, y_train, y_test, le

    def train_and_evaluate(self, X_train, X_test, y_train, y_test, label_encoder, suffix=""):
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Get unique labels that actually appear in test set
        unique_labels = sorted(list(set(y_test) | set(y_pred)))
        
        # Create label names for only the classes that appear
        target_names = [label_encoder.classes_[i] for i in unique_labels]
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        
        logger.info(f"Accuracy: {acc:.4f}, F1 Score: {f1:.4f}, Precision: {precision:.4f}")

        # Get per-class metrics with proper label handling
        try:
            report_dict = classification_report(
                y_test, y_pred, 
                labels=unique_labels,
                target_names=target_names, 
                output_dict=True,
                zero_division=0
            )
            
            # Create detailed classification report table
            self.create_classification_report_table(report_dict, suffix)
            
            # Print classification report
            report = classification_report(
                y_test, y_pred, 
                labels=unique_labels,
                target_names=target_names,
                zero_division=0
            )
            logger.info("\n" + report)
            
        except Exception as e:
            logger.warning(f"Could not generate classification report: {e}")

        # Use proper label mapping for confusion matrix
        self.plot_confusion(y_test, y_pred, unique_labels, target_names, suffix)
        self.plot_scores(acc, f1, precision, suffix)
        self.plot_learning_curve(model, X_train, y_train, suffix)
        self.plot_feature_importance(model, suffix)

        return model, {'accuracy': acc, 'f1_score': f1, 'precision': precision}

    def plot_confusion(self, y_true, y_pred, unique_labels, target_names, suffix=""):
        cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
        
        # Handle large number of classes
        if len(target_names) > 20:
            figsize = (max(12, len(target_names) * 0.6), max(10, len(target_names) * 0.5))
            annot = False  # Don't show numbers for large matrices
        else:
            figsize = (8, 6)
            annot = True
            
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=annot, fmt='d', cmap='Reds',
                    xticklabels=target_names,
                    yticklabels=target_names)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {suffix}')
        
        # Rotate labels if too many classes
        if len(target_names) > 10:
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
        
        plt.tight_layout()
        os.makedirs("outputs", exist_ok=True)
        plt.savefig(f"outputs/confusion_matrix_{suffix}.png", dpi=300, bbox_inches='tight')
        plt.close()

    def create_classification_report_table(self, report_dict, suffix=""):
        """Create and save detailed classification report as table"""
        # Extract per-class metrics
        classes = [key for key in report_dict.keys() if key not in ['accuracy', 'macro avg', 'weighted avg']]
        
        class_data = []
        for class_name in classes:
            class_data.append({
                'Class': class_name,
                'Precision': round(report_dict[class_name]['precision'], 4),
                'Recall': round(report_dict[class_name]['recall'], 4),
                'F1-Score': round(report_dict[class_name]['f1-score'], 4),
                'Support': int(report_dict[class_name]['support'])
            })
        
        # Add summary statistics
        class_data.append({
            'Class': '--- MACRO AVG ---',
            'Precision': round(report_dict['macro avg']['precision'], 4),
            'Recall': round(report_dict['macro avg']['recall'], 4),
            'F1-Score': round(report_dict['macro avg']['f1-score'], 4),
            'Support': int(report_dict['macro avg']['support'])
        })
        
        class_data.append({
            'Class': '--- WEIGHTED AVG ---',
            'Precision': round(report_dict['weighted avg']['precision'], 4),
            'Recall': round(report_dict['weighted avg']['recall'], 4),
            'F1-Score': round(report_dict['weighted avg']['f1-score'], 4),
            'Support': int(report_dict['weighted avg']['support'])
        })
        
        # Create DataFrame and save
        class_df = pd.DataFrame(class_data)
        os.makedirs("outputs", exist_ok=True)
        class_df.to_csv(f"outputs/classification_report_{suffix}.csv", index=False)
        
        # Log the table
        logger.info(f"\nDetailed Classification Report for {suffix}:")
        logger.info(class_df.to_string(index=False))
        
        # Create visualization of per-class performance
        self.plot_class_performance(class_df, suffix)

    def plot_class_performance(self, class_df, suffix=""):
        """Create bar chart for per-class performance"""
        # Filter out summary rows for visualization
        viz_df = class_df[~class_df['Class'].str.contains('---', na=False)]
        
        if len(viz_df) > 1:  # Only create plot if there are multiple classes
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot 1: Precision, Recall, F1-Score by class
            x = range(len(viz_df))
            width = 0.25
            
            ax1.bar([i - width for i in x], viz_df['Precision'], width, 
                   label='Precision', alpha=0.8, color='skyblue')
            ax1.bar(x, viz_df['Recall'], width, 
                   label='Recall', alpha=0.8, color='lightgreen')
            ax1.bar([i + width for i in x], viz_df['F1-Score'], width, 
                   label='F1-Score', alpha=0.8, color='lightcoral')
            
            ax1.set_xlabel('Classes')
            ax1.set_ylabel('Score')
            ax1.set_title(f'Per-Class Performance - {suffix}')
            ax1.set_xticks(x)
            ax1.set_xticklabels(viz_df['Class'], rotation=45, ha='right')
            ax1.legend()
            ax1.grid(axis='y', alpha=0.3)
            ax1.set_ylim(0, 1)
            
            # Plot 2: Support (sample count) by class
            bars = ax2.bar(viz_df['Class'], viz_df['Support'], 
                          alpha=0.8, color='Set3', edgecolor='grey')
            ax2.set_xlabel('Classes')
            ax2.set_ylabel('Number of Samples')
            ax2.set_title(f'Sample Distribution by Class - {suffix}')
            ax2.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + max(viz_df['Support'])*0.01,
                        f'{int(height)}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(f"outputs/class_performance_{suffix}.png", dpi=300, bbox_inches='tight')
            plt.close()

    def plot_scores(self, acc, f1, precision, suffix=""):
        # Create bar chart for metrics
        plt.figure(figsize=(8, 6))
        metrics = [acc, f1, precision]
        names = ['Accuracy', 'F1 Score', 'Precision']
        colors = ['skyblue', 'lightgreen', 'lightcoral']
        
        bars = plt.bar(names, metrics, color=colors, alpha=0.7, edgecolor='black')
        plt.ylim(0, 1)
        plt.ylabel('Score')
        plt.title(f'Model Performance Metrics - {suffix}')
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        os.makedirs("outputs", exist_ok=True)
        plt.savefig(f"outputs/metrics_plot_{suffix}.png", dpi=300)
        plt.close()
        
        # Create and save metrics table
        metrics_df = pd.DataFrame({
            'Metric': names,
            'Score': metrics
        })
        metrics_df['Score'] = metrics_df['Score'].round(4)
        metrics_df.to_csv(f"outputs/metrics_table_{suffix}.csv", index=False)
        logger.info(f"\nMetrics Table for {suffix}:")
        logger.info(metrics_df.to_string(index=False))

    def plot_learning_curve(self, model, X, y, suffix=""):
        train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5, scoring='accuracy')
        train_mean = np.mean(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)

        plt.figure(figsize=(6, 4))
        plt.plot(train_sizes, train_mean, label='Training Accuracy')
        plt.plot(train_sizes, test_mean, label='Validation Accuracy')
        plt.xlabel('Training Set Size')
        plt.ylabel('Accuracy')
        plt.title('Learning Curve')
        plt.legend()
        plt.tight_layout()
        os.makedirs("outputs", exist_ok=True)
        plt.savefig(f"outputs/learning_curve_{suffix}.png")
        plt.close()

    def plot_feature_importance(self, model, suffix=""):
        importances = model.feature_importances_
        feature_names = self.selected_features
        plt.figure(figsize=(8, 5))
        sns.barplot(x=importances, y=feature_names, palette='Set3')
        plt.title("Feature Importances")
        plt.tight_layout()
        os.makedirs("outputs", exist_ok=True)
        plt.savefig(f"outputs/feature_importance_{suffix}.png")
        plt.close()

if __name__ == "__main__":
    pipeline = RandomForestPipeline()
    label_columns = ['incident_category', 'crime_type_super', 'crime_type_violent', 'crime_type_occupation']
    pipeline.run_multiple_targets(label_columns)