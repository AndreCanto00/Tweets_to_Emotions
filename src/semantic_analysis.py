from nltk.corpus import wordnet
from typing import Dict, List, Tuple
import pandas as pd

def get_antonyms(word: str) -> List[str]:
    """
    Get all antonyms for a given word using WordNet.
    
    Args:
        word (str): Input word to find antonyms for
        
    Returns:
        List[str]: List of unique antonyms
    """
    antonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            if lemma.antonyms():
                antonyms.extend([antonym.name() for antonym in lemma.antonyms()])
    return list(set(antonyms))

def get_detailed_antonyms(word: str) -> Dict[str, List[str]]:
    """
    Get antonyms organized by synset for a given word.
    
    Args:
        word (str): Input word
        
    Returns:
        Dict[str, List[str]]: Dictionary mapping synset strings to their antonyms
    """
    detailed_antonyms = {}
    for synset in wordnet.synsets(word):
        detailed_antonyms[str(synset)] = []
        for lemma in synset.lemmas():
            for antonym in lemma.antonyms():
                detailed_antonyms[str(synset)].append(antonym.name())
    return detailed_antonyms

def max_wup_similarity(category: str, c_antonyms: List[str]) -> float:
    """
    Calculate maximum Wu-Palmer similarity between a category and its antonyms.
    
    Args:
        category (str): Category word
        c_antonyms (List[str]): List of antonyms
        
    Returns:
        float: Maximum similarity score
    """
    max_sim = 0
    for ant in c_antonyms:
        for synset_category in wordnet.synsets(category):
            for synset_ant in wordnet.synsets(ant):
                similarity = synset_category.wup_similarity(synset_ant)
                if similarity is not None and similarity > max_sim:
                    max_sim = similarity
    return max_sim

def sim_between_categories(ci: str, cj: str, 
                         ci_antonyms: List[str], 
                         cj_antonyms: List[str]) -> float:
    """
    Calculate similarity between two categories based on their antonyms.
    
    Args:
        ci (str): First category
        cj (str): Second category
        ci_antonyms (List[str]): Antonyms of first category
        cj_antonyms (List[str]): Antonyms of second category
        
    Returns:
        float: Similarity score, -1 if no antonyms found
    """
    if not ci_antonyms and not cj_antonyms:
        return -1
    return 1 - 0.5 * (max_wup_similarity(ci, ci_antonyms) + 
                      max_wup_similarity(cj, cj_antonyms))

def analyze_category_relationships(categories: List[str]) -> pd.DataFrame:
    """
    Analyze relationships between all pairs of categories.
    
    Args:
        categories (List[str]): List of category names
        
    Returns:
        pd.DataFrame: Pivot table of similarities between categories
    """
    # Get antonyms for all categories
    antonyms_category = {cat: get_antonyms(cat) for cat in categories}
    
    # Calculate similarities between all pairs
    result = []
    for i in range(len(categories)):
        for j in range(i + 1, len(categories)):
            sim = sim_between_categories(
                categories[i], categories[j],
                antonyms_category[categories[i]],
                antonyms_category[categories[j]]
            )
            result.append((categories[i], categories[j], sim))
    
    # Create DataFrame and pivot table
    df = pd.DataFrame(result, columns=["Cat1", "Cat2", "Similarity"])
    pivot_table = pd.pivot_table(
        df, 
        values="Similarity", 
        index="Cat1", 
        columns="Cat2", 
        fill_value="-"
    )
    
    return pivot_table