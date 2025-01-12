import pytest
from labeler.labelers import TransactionLabeler
from labeler.labelers import INCOME_LABELS,EXPENSE_LABELS

@pytest.fixture
def labeler():
    return TransactionLabeler()

def test_keyword_based_label_income(labeler):
    description = "Monthly subscription for coworking space"
    label, confidence = labeler.keyword_labeler.label(description, 'CR')
    assert label == 'I_MEMBERSHIP'
    assert confidence == 70

def test_keyword_based_label_expense(labeler):
    description = "Electricity bill for September"
    label, confidence = labeler.keyword_labeler.label(description, 'DR')
    assert label == 'E_RENT_UTILITIES'
    assert confidence == 70

def test_fuzzy_label_matching(labeler):
    description = "Monthly subscription for coworking space"
    labeler.fuzzy_labeler.remember(description, 'I_MEMBERSHIP')
    matched_label, confidence = labeler.fuzzy_labeler.label("monthly subscription coworking")
    assert matched_label == 'I_MEMBERSHIP'
    assert confidence >= 85

def test_regex_label(labeler):
    description = "SALARY PAYMENT"
    label, confidence = labeler.regex_labeler.label(description, 'CR')
    assert label == None
    assert confidence == 0

def test_bert_label_prediction(labeler):
    description = "Conference room booking"
    label, confidence = labeler.bert_labeler.predict(description)
    assert label in INCOME_LABELS + EXPENSE_LABELS
    assert 0 <= confidence <= 100
