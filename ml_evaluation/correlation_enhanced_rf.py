import os
import glob
import logging
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from openai import OpenAI
from dotenv import load_dotenv
from collections import Counter
import json
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CorrelationEnhancedRandomForest:
    def __init__(self):
  
        load_dotenv()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        csv_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "csv")
        csv_files = glob.glob(os.path.join(csv_dir, "sf_crime_*.csv.gz"))
        
        if csv_files:
            self.csv_path = max(csv_files, key=os.path.getctime)
            logger.info(f"Using CSV file: {self.csv_path}")
        else:
            raise FileNotFoundError("No CSV files found in data/csv directory")
            
        self.df = pd.read_csv(self.csv_path)
        logger.info(f"CSV file loaded successfully. Columns: {self.df.columns.tolist()}")
        logger.info(f"Dataset shape: {self.df.shape}")
        
        self.create_derived_targets()
        self.handle_missing_values()
        
        self.category_scores = {}
        
    def create_derived_targets(self):
        violent_crimes = [
            'Homicide', 'Rape', 'Assault', 'Robbery', 
            'Human Trafficking (A), Commercial Sex Acts',
            'Human Trafficking, Commercial Sex Acts'
        ]
        self.df['crime_type_violent'] = self.df['incident_category'].isin(violent_crimes).astype(str)
        

        super_categories = {
            'VIOLENT': violent_crimes,
            'PROPERTY': ['Burglary', 'Larceny Theft', 'Motor Vehicle Theft', 'Arson', 'Vandalism'],
            'DRUG': ['Drug Offense', 'Drug Violation'],
            'FINANCIAL': ['Fraud', 'Embezzlement', 'Forgery And Counterfeiting'],
            'OTHER': ['Other', 'Other Offenses', 'Other Miscellaneous']
        }
        
        self.df['crime_type_super'] = 'MISC'
        for category, crimes in super_categories.items():
            mask = self.df['incident_category'].isin(crimes)
            self.df.loc[mask, 'crime_type_super'] = category
        
        # Map occupation-related crimes
        occupation_crimes = [
            'Prostitution', 'Gambling', 'Liquor Laws', 
            'Drug Offense', 'Drug Violation',
            'Forgery And Counterfeiting', 'Embezzlement'
        ]
        self.df['crime_type_occupation'] = self.df['incident_category'].isin(occupation_crimes).astype(str)
        
        logger.info("Created derived target columns")
        logger.info(f"Violent crimes distribution: \n{self.df['crime_type_violent'].value_counts()}")
        logger.info(f"Super categories distribution: \n{self.df['crime_type_super'].value_counts()}")
        logger.info(f"Occupation crimes distribution: \n{self.df['crime_type_occupation'].value_counts()}")
    
    def handle_missing_values(self):
        categorical_columns = ['incident_category', 'analysis_neighborhood']
        for col in categorical_columns:
            self.df[col] = self.df[col].fillna('Unknown')
        
        numerical_columns = ['latitude', 'longitude']
        for col in numerical_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(self.df[col].median())
        
        logger.info("Missing values handled")
    
    def generate_category_scores(self):
        unique_categories = [str(cat) for cat in self.df['incident_category'].unique() if pd.notna(cat)]
        logger.info(f"Found {len(unique_categories)} unique categories")
        

        categories_list = "\n".join(unique_categories)
        prompt = f"""Analyze each crime incident category below and provide the following scores (1-10):
1. Severity: How serious is the crime
2. Social Impact: Effect on community
3. Resource Intensity: Police resources needed
4. Prevention Score: How preventable is this type of crime
5. Recurrence Risk: Likelihood of repeat offenses

Format the response as a JSON object where each category is a key with a dictionary containing these scores.
Example format:
{{"ASSAULT": {{"severity": 8, "impact": 7, "resources": 6, "prevention": 5, "recurrence": 7}}}}

Categories:
{categories_list}"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a crime analysis expert. Respond only with the requested JSON format, without any markdown formatting or additional text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            response_text = response.choices[0].message.content.strip()
            response_text = response_text.replace('```json', '').replace('```', '').strip()
            logger.info(f"LLM Response: {response_text[:100]}...")
            
            try:
                self.category_scores = json.loads(response_text)
                logger.info(f"Successfully parsed scores for {len(self.category_scores)} categories")
            except json.JSONDecodeError as e:
                logger.error(f"JSON parse error: {e}")
                logger.error(f"Response text: {response_text}")
                self.category_scores = {str(cat): {
                    'severity': 5, 'impact': 5, 'resources': 5,
                    'prevention': 5, 'recurrence': 5
                } for cat in unique_categories}
            
        except Exception as e:
            logger.error(f"Error generating category scores: {e}")
            self.category_scores = {str(cat): {
                'severity': 5, 'impact': 5, 'resources': 5,
                'prevention': 5, 'recurrence': 5
            } for cat in unique_categories}
    
    def create_temporal_features(self):
        """Create enhanced temporal features"""
        self.df['incident_datetime'] = pd.to_datetime(self.df['incident_datetime'])
        self.df['hour'] = self.df['incident_datetime'].dt.hour
        self.df['month'] = self.df['incident_datetime'].dt.month
        self.df['dayofweek'] = self.df['incident_datetime'].dt.weekday
        
     
        self.df['is_night'] = (self.df['hour'] >= 22) | (self.df['hour'] <= 5)
        self.df['is_rush_hour'] = self.df['hour'].isin([7, 8, 9, 16, 17, 18])
        self.df['is_weekend'] = self.df['dayofweek'].isin([5, 6])


        hour_crime_counts = self.df.groupby('hour').size()
        total_crimes = len(self.df)
        self.df['time_risk_score'] = self.df['hour'].map(hour_crime_counts / total_crimes)

    def create_spatial_features(self):
        le_neigh = LabelEncoder()
        self.df['neigh_enc'] = le_neigh.fit_transform(self.df['analysis_neighborhood'].fillna('Unknown'))
        

        neigh_crime_counts = self.df.groupby('analysis_neighborhood').size()
        total_crimes = len(self.df)
        self.df['neighborhood_risk_score'] = self.df['analysis_neighborhood'].map(neigh_crime_counts / total_crimes)
        

        if 'latitude' in self.df.columns and 'longitude' in self.df.columns:
            self.df['geo_grid'] = pd.cut(self.df['latitude'], bins=10, labels=False) * 10 + \
                                pd.cut(self.df['longitude'], bins=10, labels=False)
            
            
            grid_crime_counts = self.df.groupby('geo_grid').size()
            self.df['grid_risk_score'] = self.df['geo_grid'].map(grid_crime_counts / total_crimes)

    def create_correlation_features(self):
        for score_type in ['severity', 'impact', 'resources', 'prevention', 'recurrence']:
            self.df[f'category_{score_type}'] = self.df['incident_category'].astype(str).map(
                lambda x: self.category_scores.get(x, {score_type: 5})[score_type] / 10.0
            )
        
        # Combined risk score with weighted components
        self.df['combined_risk_score'] = (
            self.df['time_risk_score'] * 0.3 +
            self.df['neighborhood_risk_score'] * 0.3 +
            self.df['category_severity'] * 0.2 +
            self.df['category_recurrence'] * 0.2
        )
        
        # Time-adjusted severity with more factors
        time_multiplier = np.where(
            self.df['is_night'], 1.2,  # Night crimes
            np.where(
                self.df['is_weekend'], 1.1,  # Weekend crimes
                np.where(
                    self.df['is_rush_hour'], 1.15,  # Rush hour crimes
                    1.0  # Normal hours
                )
            )
        )
        self.df['time_adjusted_severity'] = self.df['category_severity'] * time_multiplier
        
        # Location-adjusted impact with prevention score
        self.df['location_adjusted_impact'] = (
            self.df['category_impact'] * 
            (1 + self.df['neighborhood_risk_score']) *
            (2 - self.df['category_prevention'])  # Higher impact for less preventable crimes
        )
        
        # Resource intensity score
        self.df['resource_intensity_score'] = (
            self.df['category_resources'] *
            (1 + self.df['neighborhood_risk_score'] * 0.5) *
            (1 + self.df['time_risk_score'] * 0.3)
        )
        
        # Composite risk indicators
        self.df['prevention_effectiveness'] = (
            self.df['category_prevention'] *
            (1 - self.df['neighborhood_risk_score'] * 0.3) *
            (1 - self.df['time_risk_score'] * 0.2)
        )
        
        self.df['recurrence_risk_score'] = (
            self.df['category_recurrence'] *
            (1 + self.df['neighborhood_risk_score'] * 0.4) *
            (1 + self.df['time_risk_score'] * 0.3)
        )

    def create_features(self):
        """Create all features including correlation-based ones"""
        # Generate base category scores (one-time LLM call)
        self.generate_category_scores()
        
        # Create basic features
        self.create_temporal_features()
        self.create_spatial_features()
        
        # Create correlation-based features
        self.create_correlation_features()

    def preprocess(self, target_col):
        """Preprocess data for training"""
        if target_col not in self.df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset")
            
        self.create_features()

        features = [
            # Basic temporal features
            'hour', 'month', 'dayofweek',
            'is_night', 'is_rush_hour', 'is_weekend',
            'time_risk_score',
            
            # Spatial features
            'neigh_enc', 'neighborhood_risk_score',
            'grid_risk_score',
            
            # Correlation-based features
            'category_severity',
            'category_impact',
            'combined_risk_score',
            'time_adjusted_severity',
            'location_adjusted_impact',
            'resource_intensity_score',
            'prevention_effectiveness',
            'recurrence_risk_score'
        ]
        
        # Add location features if available
        if 'latitude' in self.df.columns and 'longitude' in self.df.columns:
            features.extend(['latitude', 'longitude', 'geo_grid'])
        
        X = self.df[features]
        y_raw = self.df[target_col]
        
        # Handle any remaining missing values in features
        imputer = SimpleImputer(strategy='median')
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        
        # Filter out classes with very few samples
        class_counts = y_raw.value_counts()
        min_samples = max(10, int(len(y_raw) * 0.001))  # At least 10 samples or 0.1% of data
        valid_classes = class_counts[class_counts >= min_samples].index
        
        # Filter data
        mask = y_raw.isin(valid_classes)
        X = X[mask]
        y_raw = y_raw[mask]
        
        # Encode labels
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        return X_train, X_test, y_train, y_test, le

    def train_and_evaluate(self, X_train, X_test, y_train, y_test, label_encoder, suffix=""):
        """Train and evaluate the model with optimized hyperparameter tuning"""
        # Apply SMOTE if needed
        class_counts = Counter(y_train)
        min_samples = min(class_counts.values())
        
        if min_samples >= 6:
            max_samples = max(class_counts.values())
            imbalance_ratio = max_samples / min_samples
            
            if imbalance_ratio > 2:
                smote_k = min(5, min_samples - 1)
                sm = SMOTE(random_state=42, k_neighbors=smote_k)
                X_train, y_train = sm.fit_resample(X_train, y_train)
        
        # Define optimized hyperparameter search space
        param_dist = {
            'n_estimators': randint(50, 200),  # Reduced range
            'max_depth': randint(5, 20),       # Reduced range
            'min_samples_split': randint(2, 10),
            'min_samples_leaf': randint(1, 5),
            'max_features': ['sqrt', 'log2'],   # Removed None option
            'class_weight': ['balanced', 'balanced_subsample'],
            'bootstrap': [True]  # Fixed to True as it's generally better
        }
        
        # Initialize base model with some good defaults
        base_model = RandomForestClassifier(
            random_state=42,
            n_jobs=-1,  # Parallel processing within each model
            verbose=0
        )
        
        # Initialize RandomizedSearchCV with optimized settings
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_dist,
            n_iter=10,     # Reduced number of iterations
            cv=3,          # Reduced number of folds
            scoring='f1_weighted',
            n_jobs=2,      # Limit parallel jobs to avoid memory issues
            random_state=42,
            verbose=1
        )
        
        # Fit RandomizedSearchCV
        logger.info("Starting hyperparameter tuning...")
        random_search.fit(X_train, y_train)
        logger.info(f"Best parameters found: {random_search.best_params_}")
        
        # Get best model
        model = random_search.best_estimator_
        
        # Predict and evaluate
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        
        logger.info(f"Accuracy: {acc:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        
        # Save detailed results
        os.makedirs("outputs", exist_ok=True)
        
        # Feature importance analysis
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        feature_importance.to_csv(f"outputs/correlation_feature_importance_{suffix}.csv", index=False)
        
        return model, {
            'accuracy': acc,
            'f1_score': f1,
            'precision': precision,
            'feature_importance': feature_importance,
            'best_params': random_search.best_params_
        }

    def run_experiment(self, label_columns):
        """Run the experiment with multiple target columns"""
        results = []
        
        for label_col in label_columns:
            logger.info(f"\nTraining model for target: {label_col}")
            
            try:
                X_train, X_test, y_train, y_test, label_encoder = self.preprocess(label_col)
                
                model, metrics = self.train_and_evaluate(
                    X_train, X_test, y_train, y_test,
                    label_encoder, suffix=label_col
                )
                
                metrics['label'] = label_col
                results.append(metrics)
                
            except Exception as e:
                logger.error(f"Error processing {label_col}: {e}")
                results.append({
                    'label': label_col,
                    'accuracy': 0,
                    'f1_score': 0,
                    'precision': 0,
                    'error': str(e)
                })
        
        # Save results
        df_results = pd.DataFrame(results)
        df_results.to_csv("outputs/correlation_enhanced_results.csv", index=False)
        logger.info("\nFinal Results:\n" + str(df_results))
        
        return df_results

if __name__ == "__main__":
    # Initialize with your API key
    pipeline = CorrelationEnhancedRandomForest()
    
    # Get available target columns - focusing on super categories and violent crimes
    available_targets = [
        col for col in [
            'crime_type_super',    # Main categories only
            'crime_type_violent'   # Binary classification for violent crimes
        ] if col in pipeline.df.columns
    ]
    
    logger.info(f"Available target columns: {available_targets}")
    
    # Run experiment with available targets
    results = pipeline.run_experiment(available_targets) 
