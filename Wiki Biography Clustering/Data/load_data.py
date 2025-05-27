import pandas as pd

def load_wikipedia_data(filepath):
    
    data = pd.read_csv(filepath, names=["URI", "name", "text"], skiprows=1)
    
    return data