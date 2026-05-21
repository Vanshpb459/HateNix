import joblib
import re
import nltk
import emoji
import contractions
import os
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
from transformers import pipeline, RobertaTokenizer, RobertaForSequenceClassification
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from indicnlp.tokenize import indic_tokenize
from collections import defaultdict


class HateSpeechModel:
    def __init__(self, model_path='hate_speech_model.pkl', dataset_path=None):
        # Initialize NLTK resources
        self._init_nltk()

        # Initialize with comprehensive Hindi/Urdu slurs list
        self.HINDI_SLURS = {
            'rand', 'chuche','Chuche', 'Bhokachoda','bondu','Aandi mandi sandi','Chucha' ,'gasti', 'lund', 'chut', 'madarchod', 'bhenchod',
            'kutta', 'kamina', 'harami', 'gandu', 'lauda', 'chutiya', 'bhosdike',
            'lavde', 'jhaat', 'gaand', 'fuddu', 'randi', 'kuttiya', 'besharm',
            'kamine', 'chod', 'chodu', 'chudai', 'gand', 'jhat', 'loda', 'lodu',
            'mader', 'mammey', 'rape', 'rapist', 'sala', 'sali', 'suar', 'tatti',
            'chhinal', 'pataka', 'saala', 'kanjar', 'dalla', 'bhadwa', 'bhosda',
            'raand', 'kameena', 'jhandu', 'bakwass', 'ghatiya', 'dhatura', 'bevkuf',
            'pagla', 'nalaayak', 'dhokhebaaz', 'badtamiz', 'chutmarani', 'gandmasti',
            'luchcha', 'lafanga', 'bhadwi', 'chutmar', 'jhaantu', 'bakloli', 'suar ka',
            'gandi', 'chhati', 'maadarchod', 'bhen ka', 'khotta', 'chhinalpan',
            'randibaaz', 'lodey', 'gandiya', 'chutki', 'bhosdi', 'kallu', 'habshi',
            'chamar', 'bhangi', 'kuttey', 'sala kutta', 'saali randi', 'chut ka',
            'gandu ka', 'lavda', 'jhaati', 'bhosdika', 'randipana', 'kutta kamina',
            'haramkhor', 'chhinal ki', 'gand ka', 'loda ka', 'chutad', 'bhadwe',
            'saala harami', 'kutti', 'chhinala', 'gandmar', 'jhatu', 'luchchi',
            'bakwas ki', 'ghatiyapan', 'chut ke', 'randi ka', 'bhosdi ka', 'kameenapan',
            'suar ki', 'tatti ka', 'gand ka diwana', 'chutmarika', 'lode ka', 'jhaat ka',
            'bhadwa pan', 'chhinalpana', 'randi rona', 'gandu pan', 'kutta pan',
            'harami pan', 'chutiya pan', 'bhosdi pan', 'gand mara', 'jhaat mara',
            'loda mara', 'chut mara', 'saala bhadwa', 'kamina pan', 'suar pan',
            'tatti pan', 'ghatiya pan', 'bakwas pan', 'nalayak pan', 'dhokebaaz pan',
            'badtameez pan', 'chhinal ka', 'pataka ka', 'kanjar pan', 'dalla ka',
            'chutiyapa', 'gandupana', 'bhosdiki', 'randikhana', 'kutta ka', 'suar kutta',
            'chut ka pujari', 'gand ka pujari', 'jhaat ka pujari', 'loda pujari',
            'chutmar ke', 'gandmar ke', 'bhadwa ka', 'saala kanjar', 'kutti ki',
            'randi ke', 'chhinal ke', 'bhosda ka', 'tatti ki', 'suar ke', 'kutta ke',
            'harami ka', 'kamina ka', 'nalayak ka', 'dhokebaaz ka', 'badtameez ka',
            'gandi baat', 'chut ki baat', 'gand ki baat', 'jhaat ki baat', 'loda ki baat',
            'bhosda ki baat', 'randi ki baat', 'chhinal ki baat', 'bhadwa ki baat',
            'kutta ki baat', 'suar ki baat', 'tatti ki baat', 'harami ki baat',
            'kamina ki baat', 'nalayak ki baat', 'ghatiya baat', 'bakwas baat',
            'chutiya baat', 'gandu baat', 'bhosdi baat', 'randipana ka', 'chhinalpana ka',
            'gandupana ka', 'chutiyapa ka', 'bhadwa panti', 'kanjar panti', 'dalla panti',
            'sala panti', 'saali panti', 'kutta panti', 'suar panti', 'randi panti',
            'chhinal panti', 'harami panti', 'kamina panti', 'nalayak panti',
            'ghatiya panti', 'bakwas panti', 'chutiya panti', 'gandu panti', 'bhosdi panti',
            'chut ka raja', 'gand ka raja', 'jhaat ka raja', 'loda ka raja', 'bhadwa raja',
            'randi rani', 'chhinal rani', 'kutti rani', 'saali rani', 'gandi rani',
            'tatti raja', 'suar raja', 'kutta raja', 'harami raja', 'kamina raja',
            'nalayak raja', 'ghatiya raja', 'bakwas raja', 'chutiya raja', 'gandu raja',
            'bhosdi raja', 'chut ka khel', 'gand ka khel', 'jhaat ka khel', 'loda ka khel',
            'bhadwa khel', 'randi khel', 'chhinal khel', 'kutta khel', 'suar khel',
            'tatti khel', 'harami khel', 'kamina khel', 'nalayak khel', 'ghatiya khel',
            'bakwas khel', 'chutiya khel', 'gandu khel', 'bhosdi khel', 'chut ka deewana',
            'gand ka deewana', 'jhaat ka deewana', 'loda ka deewana', 'bhadwa deewana',
            'randi deewani', 'chhinal deewani', 'kutti deewani', 'saali deewani',
            'gandi deewani', 'tatti deewana', 'suar deewana', 'kutta deewana',
            'harami deewana', 'kamina deewana', 'nalayak deewana', 'ghatiya deewana',
            'bakwas deewana', 'chutiya deewana', 'gandu deewana', 'bhosdi deewana'
        }

        # Load models
        self.roberta_model = self._load_roberta_model()
        self.custom_model = self._load_custom_model(model_path, dataset_path)

        # Initialize counters for debugging
        self.slur_detections = defaultdict(int)
        self.total_predictions = 0

    def _init_nltk(self):
        """Initialize NLTK resources"""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
            print("NLTK resources already downloaded")
        except LookupError:
            print("Downloading NLTK resources...")
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')

        # Initialize stopwords
        self.stop_words = set(stopwords.words('english'))
        self.hindi_stopwords = set(stopwords.words('hindi')) if 'hindi' in stopwords.fileids() else set()

    def _load_roberta_model(self):
        """Initialize RoBERTa hate speech model"""
        try:
            print("Loading RoBERTa model...")
            model_name = "facebook/roberta-hate-speech-dynabench-r4-target"
            tokenizer = RobertaTokenizer.from_pretrained(model_name)
            model = RobertaForSequenceClassification.from_pretrained(model_name)
            return pipeline('text-classification', model=model, tokenizer=tokenizer)
        except Exception as e:
            print(f"Failed to load RoBERTa model: {e}")
            return None

    def _load_custom_model(self, model_path, dataset_path):
        """Load custom trained model with validation"""
        try:
            pipeline = joblib.load(model_path)
            print(f"Loaded custom model from {model_path}")

            # Validate model if dataset available
            default_dataset = '/Users/vansh./PycharmProjects/model/hindi-hinglish-hate-speech-dataset.txt'
            dataset_to_use = dataset_path or default_dataset

            if os.path.exists(dataset_to_use):
                print(f"Validating model with dataset {dataset_to_use}...")
                self._validate_model(pipeline, dataset_to_use)

            return pipeline
        except Exception as e:
            print(f"Custom model load failed: {e}")
            if dataset_path and os.path.exists(dataset_path):
                print("Training new model...")
                return self._train_new_model(dataset_path, model_path)
            return None

    def _validate_model(self, pipeline, dataset_path):
        """Validate model performance on test dataset"""
        df = self._load_dataset(dataset_path)
        if len(df) < 50:
            print("Insufficient data for validation")
            return

        X = df['text'].apply(self.preprocess)
        y = df['label']

        # Test on sample data
        sample = df.sample(min(100, len(df)))
        predictions = pipeline.predict(sample['text'].apply(self.preprocess))

        # Calculate accuracy
        accuracy = (predictions == sample['label']).mean()
        print(f"Model validation accuracy: {accuracy:.2f}")

        if accuracy < 0.7:
            print("Warning: Model accuracy is below 70%")

    def _train_new_model(self, dataset_path, model_path):
        """Train new model from dataset"""
        try:
            df = self._load_dataset(dataset_path)
            if len(df) < 100:
                raise ValueError("Insufficient training data")

            # Preprocess and balance data
            df['cleaned'] = df['text'].apply(self.preprocess)
            class_weights = compute_class_weight('balanced', classes=np.unique(df['label']), y=df['label'])

            # Build pipeline
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(
                    max_features=50000,
                    ngram_range=(1, 3),
                    min_df=2,
                    max_df=0.9,
                    sublinear_tf=True
                )),
                ('clf', RandomForestClassifier(
                    class_weight={0: class_weights[0], 1: class_weights[1]},
                    n_estimators=300,
                    max_depth=100,
                    random_state=42
                ))
            ])

            # Train and evaluate
            X_train, X_test, y_train, y_test = train_test_split(
                df['cleaned'], df['label'], test_size=0.2, random_state=42
            )

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            print(classification_report(y_test, y_pred))

            # Save model
            joblib.dump(pipeline, model_path)
            print(f"Model saved to {model_path}")
            return pipeline

        except Exception as e:
            print(f"Training failed: {e}")
            return None

    def preprocess(self, text):
        """Enhanced preprocessing with Hindi slur protection"""
        if not isinstance(text, str):
            return ""

        text = text.lower()

        # Preserve slurs by converting to special tokens
        for slur in self.HINDI_SLURS:
            if slur in text:
                text = text.replace(slur, f"SLUR_{slur.upper()}")

        # Standard cleaning
        text = emoji.demojize(text, delimiters=(' ', ' '))
        text = contractions.fix(text)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'[^\w\sऀ-ॿ0-9]', ' ', text)

        # Tokenize with Hindi support
        try:
            tokens = indic_tokenize.trivial_tokenize(text, lang='hi')
        except:
            tokens = word_tokenize(text)

        # Filter stopwords but keep SLUR tokens
        stop_words = self.stop_words.union(self.hindi_stopwords)
        custom_stop = {'hai', 'ho', 'hain', 'main', 'mera', 'ki', 'ka', 'ke', 'ko'}
        tokens = [word for word in tokens if (word not in stop_words and len(word) > 1) or word.startswith('SLUR_')]

        return ' '.join(tokens)

    def predict(self, comments, threshold=0.7):
        """Enhanced prediction with explicit slur detection and model fusion"""
        if not comments:
            return []

        results = []
        self.total_predictions += len(comments)

        for comment in comments:
            # First check for explicit slurs
            lower_comment = comment.lower()
            detected_slurs = [slur for slur in self.HINDI_SLURS if slur in lower_comment]

            if detected_slurs:
                for slur in detected_slurs:
                    self.slur_detections[slur] += 1
                results.append((1, 0.95))  # High confidence for explicit slurs
                continue

            # If no slurs, use model prediction
            try:
                # Custom model prediction
                custom_pred, custom_score = 0, 0.0
                if self.custom_model:
                    clean_text = self.preprocess(comment)
                    proba = self.custom_model.predict_proba([clean_text])[0][1]
                    custom_pred = int(proba >= threshold)
                    custom_score = proba

                # RoBERTa prediction
                roberta_pred, roberta_score = 0, 0.0
                if self.roberta_model:
                    roberta_res = self.roberta_model(comment)[0]
                    roberta_score = roberta_res['score']
                    roberta_pred = 1 if roberta_res['label'] == 'LABEL_1' else 0

                # Fusion logic
                if custom_pred + roberta_pred >= 1:  # At least one model says hate
                    final_score = max(custom_score, roberta_score)
                    # Penalize if only one model detected
                    if custom_pred + roberta_pred == 1:
                        final_score *= 0.8
                    results.append((1, final_score))
                else:
                    results.append((0, 1 - max(custom_score, roberta_score)))

            except Exception as e:
                print(f"Prediction error for comment: {e}")
                results.append((0, 0.0))

        return results

    def get_detection_stats(self):
        """Return statistics about detected slurs"""
        return {
            'total_predictions': self.total_predictions,
            'slur_detections': dict(self.slur_detections),
            'slur_ratio': sum(self.slur_detections.values()) / max(1, self.total_predictions)
        }

    def _load_dataset(self, filepath):
        """Load dataset from file with validation"""
        data = []
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                reader = csv.reader(file)
                next(reader, None)  # Skip header
                for row in reader:
                    if len(row) >= 2:
                        try:
                            data.append({
                                'text': row[0].strip(),
                                'label': int(row[1].strip())
                            })
                        except (ValueError, IndexError):
                            continue
        except Exception as e:
            print(f"Error loading dataset: {e}")

        return pd.DataFrame(data)
