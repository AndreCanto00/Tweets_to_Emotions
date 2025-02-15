import pytest
from nltk.corpus import wordnet
import pandas as pd
from src.semantic_analysis import (
    get_antonyms,
    get_detailed_antonyms,
    max_wup_similarity,
    sim_between_categories,
    analyze_category_relationships
)

@pytest.fixture
def sample_categories():
    return ['happy', 'sad', 'good']

def test_get_antonyms():
    # Test con parola che ha antonyms noti
    antonyms = get_antonyms('happy')
    assert isinstance(antonyms, list)
    assert 'unhappy' in antonyms or 'sad' in antonyms
    
    # Test con parola senza antonyms
    assert get_antonyms('xyzabc') == []

def test_get_detailed_antonyms():
    detailed = get_detailed_antonyms('good')
    assert isinstance(detailed, dict)
    
    # Verifica che le chiavi siano stringhe di synset
    assert all(isinstance(k, str) for k in detailed.keys())
    assert all(isinstance(v, list) for v in detailed.values())

def test_max_wup_similarity():
    # Test con categorie note
    similarity = max_wup_similarity('happy', ['sad', 'unhappy'])
    assert 0 <= similarity <= 1
    
    # Test con lista vuota
    assert max_wup_similarity('happy', []) == 0

def test_sim_between_categories():
    # Test con categorie note
    sim = sim_between_categories(
        'happy', 'sad',
        ['unhappy'], ['happy']
    )
    assert isinstance(sim, float)
    assert -1 <= sim <= 1
    
    # Test senza antonyms
    assert sim_between_categories('happy', 'sad', [], []) == -1

def test_analyze_category_relationships(sample_categories):
    pivot_table, antonyms_dict = analyze_category_relationships(sample_categories)
    
    # Verifica che sia un DataFrame
    assert isinstance(pivot_table, pd.DataFrame)
    
    # Verifica che antonyms_dict sia un dizionario
    assert isinstance(antonyms_dict, dict)
    
    # Verifica che ogni categoria abbia i suoi antonimi
    assert set(antonyms_dict.keys()) == set(sample_categories)
    
    # Verifica che gli antonimi siano liste
    assert all(isinstance(v, list) for v in antonyms_dict.values())
    
    # Verifica dimensioni della pivot table
    n = len(sample_categories)
    assert pivot_table.shape[0] <= n
    
    # Verifica che tutti i valori siano numeri o '-'
    for col in pivot_table.columns:
        assert all(isinstance(v, (float, str)) for v in pivot_table[col])

def test_empty_categories():
    pivot_table, antonyms_dict = analyze_category_relationships([])
    assert isinstance(pivot_table, pd.DataFrame)
    assert pivot_table.empty
    assert isinstance(antonyms_dict, dict)
    assert len(antonyms_dict) == 0

@pytest.mark.parametrize("category", ['happy', 'sad', 'good', 'bad'])
def test_individual_categories(category):
    # Test che ogni categoria produce risultati validi
    antonyms = get_antonyms(category)
    assert isinstance(antonyms, list)
    
    detailed = get_detailed_antonyms(category)
    assert isinstance(detailed, dict)