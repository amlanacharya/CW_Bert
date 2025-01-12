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

# Add these constants at the top of the file
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

class TransactionLabeler:
    _instance = None  # Singleton instance for TransactionLabeler too
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TransactionLabeler, cls).__new__(cls)
            cls._instance.init_labeler()
        return cls._instance
    
    def init_labeler(self):
        """Initialize the labeler only once"""
        self.label_history = {}
        self.model = None
        self.vectorizer = None
        self.load_model()
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
            # Add more patterns for other labels
        }
        self.similarity_threshold = 85
        self.known_transactions = {}
        self.bert_labeler = BERTLabeler()  # Will reuse existing instance
        self.use_bert = True

    def load_model(self):
        try:
            with open('transaction_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            with open('vectorizer.pkl', 'rb') as f:
                self.vectorizer = pickle.load(f)
        except FileNotFoundError:
            self.model = MultinomialNB()
            self.vectorizer = TfidfVectorizer()

    def save_model(self):
        with open('transaction_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        with open('vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)

    def keyword_based_label(self, description, transaction_type='DR'):
        description = description.lower()
        prefix = 'E_' if transaction_type == 'DR' else 'I_'
        
        for label, keywords in LABEL_KEYWORDS.items():
            if label.startswith(prefix):
                if any(keyword in description for keyword in keywords):
                    return label
        
        return f'{prefix}label5'  # Default to 'other' category

    def pattern_based_label(self, description):
        # Look up in historical patterns
        if description in self.label_history:
            return self.label_history[description]
        return None

    def ml_based_label(self, descriptions, labels=None):
        if labels is not None:  # Training
            X = self.vectorizer.fit_transform(descriptions)
            self.model.fit(X, labels)
            self.save_model()
        else:  # Prediction
            if self.model is None:
                return None
            X = self.vectorizer.transform(descriptions)
            return self.model.predict(X)

    def clean_description(self, description):
        # Remove special characters and normalize text
        return re.sub(r'[^a-zA-Z0-9\s]', '', description.lower())

    def regex_based_label(self, description, transaction_type='DR'):
        description = description.upper()
        prefix = 'E_' if transaction_type == 'DR' else 'I_'
        
        for label, patterns in self.regex_patterns.items():
            if label.startswith(prefix):
                for pattern in patterns:
                    if re.match(pattern, description):
                        return label
        return None

    def fuzzy_match_label(self, description):
        """Try to find similar transactions we've seen before"""
        if not self.known_transactions:  # First time seeing any transaction
            return None, 0

        # Find the closest match
        closest_match, score = process.extractOne(
            description,
            self.known_transactions.keys(),
            scorer=fuzz.token_sort_ratio
        )

        if score >= self.similarity_threshold:
            return self.known_transactions[closest_match], score
        return None, 0

    def remember_transaction(self, description, label):
        """Remember this transaction for future matching"""
        self.known_transactions[description] = label

    def label_transaction(self, description, transaction_type='DR'):
        # Try BERT first if enabled
        if self.use_bert:
            bert_label, bert_confidence = self.bert_labeler.predict(description)
            if bert_label and bert_confidence > 90:  # High confidence threshold for BERT
                self.remember_transaction(description, bert_label)
                return bert_label, bert_confidence

        # Try fuzzy matching first
        fuzzy_label, confidence = self.fuzzy_match_label(description)
        if fuzzy_label:
            return fuzzy_label, confidence

        # Try regex patterns
        label = self.regex_based_label(description, transaction_type)
        if label:
            self.remember_transaction(description, label)
            return label, 90  # High confidence for regex matches

        # Try pattern matching
        label = self.pattern_based_label(description)
        if label:
            self.remember_transaction(description, label)
            return label, 85  # Good confidence for pattern matches

        # Try keyword matching
        label = self.keyword_based_label(description, transaction_type)
        self.remember_transaction(description, label)
        return label, 70  # Lower confidence for keyword matches

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
    
    def save_model_pickle(self):
        if self.model:
            version = datetime.now().strftime("%Y%m%d_%H%M")
            filename = f'bert_model_{version}.pkl'
            model_state = {
                'state_dict': self.model.state_dict(),
                'config': self.model.config,
                'label_encoder': self.label_encoder,
                'version': version,
                'created_at': datetime.now().isoformat()
            }
            with open(filename, 'wb') as f:
                torch.save(model_state, f)
            return filename
        return None
    
    def load_model_pickle(self, filename='bert_model.pkl'):
        """Load model from a pickle file"""
        try:
            if os.path.exists(filename):
                model_state = torch.load(filename, map_location=self.device)
                self.model = BertForSequenceClassification.from_pretrained(
                    'bert-base-uncased',
                    num_labels=len(self.label_encoder.classes_),
                    state_dict=model_state['state_dict']
                )
                self.model.to(self.device)
                self.label_encoder = model_state['label_encoder']
                print("Loaded BERT model from pickle file")
                return True
        except Exception as e:
            print(f"Error loading pickle file: {str(e)}")
        return False

    def load_model(self):
        try:
            if os.path.exists(self.model_path):
                self.model = BertForSequenceClassification.from_pretrained(self.model_path)
                self.model.to(self.device)
                print("Loaded existing BERT model")
            else:
                self.model = BertForSequenceClassification.from_pretrained(
                    'bert-base-uncased',
                    num_labels=len(self.label_encoder.classes_)
                )
                self.model.to(self.device)
                print("Initialized new BERT model")
        except Exception as e:
            print(f"Error loading BERT model: {str(e)}")
            self.model = None

    def save_model(self):
        if self.model:
            self.model.save_pretrained(self.model_path)
            print("Saved BERT model")
    
    def predict(self, text):
        if not self.model:
            return None, 0
        
        # Prepare input
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = softmax(outputs.logits, dim=-1)
            prediction = torch.argmax(probs, dim=-1)
            confidence = float(torch.max(probs))
        
        # Convert prediction to label
        predicted_label = self.label_encoder.inverse_transform([prediction.item()])[0]
        return predicted_label, confidence * 100

    def train(self, texts, labels, epochs=3):
        if not texts or not labels:
            return False
        
        # Encode labels
        encoded_labels = self.label_encoder.transform(labels)
        
        # Prepare dataset
        inputs = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        labels_tensor = torch.tensor(encoded_labels).to(self.device)
        
        # Training settings
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(**inputs, labels=labels_tensor)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
        
        self.save_model()
        return True

def clean_amount(amount_str):
    if isinstance(amount_str, str):
        return float(amount_str.replace('"', '').replace('=', '').replace(',', ''))
    return amount_str

def process_bank_statement(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        start_row = 0
        end_row = len(lines)
        for i, line in enumerate(lines):
            if 'Sl. No.' in line:
                start_row = i
            if 'Opening balance' in line:
                end_row = i
                break
        
        df = pd.read_csv(file_path, skiprows=start_row, nrows=end_row-start_row)
        df.columns = df.columns.str.strip()
        
        print("\n=== Initial DataFrame ===")
        print("DataFrame shape:", df.shape)
        print("\nColumns:", df.columns.tolist())
        print("\nSample data:")
        print(df.head())
        print("\nDataFrame Info:")
        df.info()
        
        df['Amount'] = df['Amount'].apply(clean_amount)
        df['Balance'] = df['Balance'].apply(clean_amount)
        
        # Sort the main dataframe first by Date and then by Sl. No.
        df['Sl. No.'] = pd.to_numeric(df['Sl. No.'], errors='coerce')
        df = df.sort_values(['Date', 'Sl. No.'], ascending=[True, True])
        
        labeler = TransactionLabeler()
        
        # Process income transactions
        income = df[df['Dr / Cr'] == 'CR'].copy()
        income['Label'] = ''
        income['Sublabel'] = ''
        income['Confidence'] = 0  # Add confidence score
        income['Clean_Description'] = income['Description'].apply(labeler.clean_description)
        
        # Apply multiple labeling methods
        for idx, row in income.iterrows():
            label, confidence = labeler.label_transaction(row['Clean_Description'], 'CR')
            income.at[idx, 'Label'] = label
            income.at[idx, 'Confidence'] = confidence
        
        # Process expense transactions
        expenses = df[df['Dr / Cr'] == 'DR'].copy()
        expenses['Label'] = ''
        expenses['Sublabel'] = ''
        expenses['Confidence'] = 0  # Add confidence score
        expenses['Clean_Description'] = expenses['Description'].apply(labeler.clean_description)
        
        for idx, row in expenses.iterrows():
            label, confidence = labeler.label_transaction(row['Clean_Description'], 'DR')
            expenses.at[idx, 'Label'] = label
            expenses.at[idx, 'Confidence'] = confidence

        # Save directly without additional sorting
        income.to_csv('uc_income.csv', index=False)
        expenses.to_csv('uc_expenses.csv', index=False)
        
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
        df['Year_Month'] = df['Date'].dt.strftime('%B %Y')
        monthly_summary = []
        
        for month in sorted(df['Year_Month'].unique(), key=lambda x: pd.to_datetime(x, format='%B %Y')):
            month_data = df[df['Year_Month'] == month]
            month_income = month_data[month_data['Dr / Cr'] == 'CR']['Amount'].sum()
            month_expenses = month_data[month_data['Dr / Cr'] == 'DR']['Amount'].sum()
            net_amount = month_income - month_expenses
            profit_or_loss = "Profit" if net_amount >= 0 else "Loss"
            monthly_summary.append({
                'Month': month,
                'Income': month_income,
                'Expenses': month_expenses,
                'Net': net_amount
            })
        
        print("\n=== Monthly Summary ===")
        for month in monthly_summary:
            print(f"{month['Month']}:")
            print(f"  Income: ₹{month['Income']:,.2f}")
            print(f"  Expenses: ₹{month['Expenses']:,.2f}")
            print(f"  Net: ₹{abs(month['Net']):,.2f} ({'Profit' if month['Net'] >= 0 else 'Loss'})")
            print()

        return income, expenses, df, labeler
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None, None, None, None

if __name__ == "__main__":
    file_path = 'Bank Statment last FY.csv'
    income_df, expenses_df, full_df, labeler = process_bank_statement(file_path)

