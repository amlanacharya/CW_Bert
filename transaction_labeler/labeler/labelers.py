from .singleton import SingletonMeta
import logging
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import re
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.nn.functional import softmax
import numpy as np
from sklearn.preprocessing import LabelEncoder
import json
import os
from datetime import datetime

INCOME_LABELS = [
    'I_MEMBERSHIP',     # Monthly/daily memberships, hot desks
    'I_MEETING_ROOMS',  # Conference/meeting room bookings
    'I_SERVICES',       # Printing, coffee, events, lockers
    'I_DEPOSITS',       # Security deposits, advances
    'I_OTHER_INCOME'    # Partnerships, sponsorships, misc
]

# Expense Categories (E_label1 to E_label5)
EXPENSE_LABELS = [
    'E_RENT_UTILITIES', # Rent, electricity, water, internet
    'E_MAINTENANCE',    # Cleaning, repairs, equipment maintenance
    'E_OPERATIONS',     # Staff salary, supplies, coffee/tea
    'E_MARKETING',      # Advertising, events, promotions
    'E_OTHER_EXPENSE'   # Insurance, taxes, misc expenses
]

# Define keywords for each category
LABEL_KEYWORDS = {
    # Income Keywords
    'I_MEMBERSHIP': [
        'monthly subscription', 'hot desk', 'dedicated desk', 
        'private cabin', 'membership fee', 'subscription'
    ],
    'I_MEETING_ROOMS': [
        'conference room', 'meeting room', 'board room',
        'event space', 'training room'
    ],
    'I_SERVICES': [
        'printing', 'scanning', 'coffee', 'locker rent',
        'event ticket', 'catering', 'business address'
    ],
    'I_DEPOSITS': [
        'security deposit', 'advance', 'key deposit',
        'refundable', 'booking advance'
    ],
    'I_OTHER_INCOME': [
        'partnership', 'sponsor', 'commission', 'late fee',
        'penalty', 'miscellaneous'
    ],

    # Expense Keywords
    'E_RENT_UTILITIES': [
        'rent', 'lease', 'electricity', 'water', 'internet',
        'wifi', 'broadband', 'property tax'
    ],
    'E_MAINTENANCE': [
        'repair', 'cleaning', 'plumbing', 'electrical work',
        'air conditioning', 'pest control', 'furniture'
    ],
    'E_OPERATIONS': [
        'salary', 'wages', 'coffee', 'tea', 'pantry',
        'stationery', 'office supplies', 'toilet'
    ],
    'E_MARKETING': [
        'advertising', 'promotion', 'social media',
        'event expense', 'marketing', 'branding'
    ],
    'E_OTHER_EXPENSE': [
        'insurance', 'legal', 'accounting', 'bank charges',
        'miscellaneous', 'travel', 'training'
    ]
}
    
class KeywordLabeler:
    def __init__(self, label_keywords):
        self.label_keywords = label_keywords

    def label(self, description, transaction_type='DR'):
        description = description.lower()
        prefix = 'E_' if transaction_type == 'DR' else 'I_'

        for label, keywords in self.label_keywords.items():
            if label.startswith(prefix):
                if any(keyword in description for keyword in keywords):
                    return label, 70  

        return f'{prefix}label5', 50  

class RegexLabeler:
    def __init__(self, regex_patterns):
        self.regex_patterns = regex_patterns

    def label(self, description, transaction_type='DR'):
        description = description.lower()
        prefix = 'E_' if transaction_type == 'DR' else 'I_'

        for label, patterns in self.regex_patterns.items():
            if label.startswith(prefix):
                for pattern in patterns:
                    if re.match(pattern, description):
                        return label, 90  
        return None, 0
    

class FuzzyLabeler:
    def __init__(self, known_transactions, similarity_threshold=85):
        self.known_transactions = known_transactions
        self.similarity_threshold = similarity_threshold

    def label(self, description):
        if not self.known_transactions:
            return None, 0

        closest_match, score = process.extractOne(
            description,
            self.known_transactions.keys(),
            scorer=fuzz.token_sort_ratio
        )

        if score >= self.similarity_threshold:
            return self.known_transactions[closest_match], score
        return None, 0

    def remember(self, description, label):
        self.known_transactions[description] = label


class MLLabeler:
    def __init__(self, model_path='transaction_model.pkl', vectorizer_path='vectorizer.pkl'):
        self.model = None
        self.vectorizer = None
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.load_model()

    def load_model(self):
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            with open(self.vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            logging.info("ML model and vectorizer loaded successfully.")
        except FileNotFoundError as e:
            logging.error(f"Model or vectorizer file not found: {e}")
            self.model = MultinomialNB()
            self.vectorizer = TfidfVectorizer()
        except Exception as e:
            logging.error(f"Unexpected error during ML model loading: {e}")
            self.model = MultinomialNB()
            self.vectorizer = TfidfVectorizer()

    def train(self, descriptions, labels):
        X = self.vectorizer.fit_transform(descriptions)
        self.model.fit(X, labels)
        self.save_model()

    def predict(self, descriptions):
        if self.model is None or self.vectorizer is None:
            logging.warning("ML model or vectorizer is not loaded.")
            return [None] * len(descriptions)
        X = self.vectorizer.transform(descriptions)
        return self.model.predict(X)

    def save_model(self):
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
        with open(self.vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        logging.info("ML model and vectorizer saved successfully.")

class BERTLabeler:
    def __init__(self, model_path='transaction_bert'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.label_encoder = LabelEncoder()
        self.model = None
        self.model_path = model_path
        all_labels = INCOME_LABELS + EXPENSE_LABELS
        self.label_encoder.fit(all_labels)
        self.load_model()

    def load_model(self):
        try:
            if os.path.exists(self.model_path):
                self.model = BertForSequenceClassification.from_pretrained(self.model_path)
                self.model.to(self.device)
                logging.info("Loaded existing BERT model.")
            else:
                self.model = BertForSequenceClassification.from_pretrained(
                    'bert-base-uncased',
                    num_labels=len(self.label_encoder.classes_)
                )
                self.model.to(self.device)
                logging.info("Initialized new BERT model.")
        except Exception as e:
            logging.error(f"Error loading BERT model: {e}")
            self.model = None

    def predict(self, text):
        if not self.model:
            logging.warning("BERT model is not loaded.")
            return None, 0

        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = softmax(outputs.logits, dim=-1)
            prediction = torch.argmax(probs, dim=-1)
            confidence = float(torch.max(probs))

        predicted_label = self.label_encoder.inverse_transform([prediction.item()])[0]
        return predicted_label, confidence * 100

    def train(self, texts, labels, epochs=3):
        if not texts or not labels:
            logging.warning("No texts or labels provided for training.")
            return False

        encoded_labels = self.label_encoder.transform(labels)
        inputs = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        labels_tensor = torch.tensor(encoded_labels).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)

        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(**inputs, labels=labels_tensor)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

        self.save_model()
        return True

    def save_model(self):
        if self.model:
            self.model.save_pretrained(self.model_path)
            logging.info("BERT model saved successfully.")

class TransactionLabeler(metaclass=SingletonMeta):
    def __init__(self):
        self.init_labeler()

    def init_labeler(self):
        self.label_history = {}
        self.known_transactions = {}
        self.regex_patterns = {
            'I_label1': [
                r'SALARY[\/\s].*',
                r'SAL[\/\s].*',
                r'MONTHLY[\/\s]PAY.*'
            ],
            'E_label1': [
                r'GROCERY[\/\s].*',
                r'SUPER[\s]?MARKET.*',
                r'FOOD[\/\s].*'
            ],
        }
        self.keyword_labeler = KeywordLabeler(LABEL_KEYWORDS)
        self.regex_labeler = RegexLabeler(self.regex_patterns)
        self.fuzzy_labeler = FuzzyLabeler(self.known_transactions)
        self.ml_labeler = MLLabeler()
        self.bert_labeler = BERTLabeler()
        self.use_bert = True

    def label_transaction(self, description, transaction_type='DR'):
    
        if self.use_bert:
            bert_label, bert_confidence = self.bert_labeler.predict(description)
            if bert_label and bert_confidence > 90:
                self.fuzzy_labeler.remember(description, bert_label)
                return bert_label, bert_confidence
            
        fuzzy_label, confidence = self.fuzzy_labeler.label(description)
        if fuzzy_label:
            return fuzzy_label, confidence

        label, confidence = self.regex_labeler.label(description, transaction_type)
        if label:
            self.fuzzy_labeler.remember(description, label)
            return label, confidence
        
        label = self.pattern_based_label(description)
        if label:
            self.fuzzy_labeler.remember(description, label)
            return label, 85  # Good confidence for pattern matches

        label, confidence = self.keyword_labeler.label(description, transaction_type)
        self.fuzzy_labeler.remember(description, label)
        return label, confidence
