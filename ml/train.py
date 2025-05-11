import numpy as np
import pandas as pd
import os
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from typing import List, Tuple

def load_player_data(filename: str = "player_observations.csv"):
    """Load player movement data from CSV file."""
    try:
        if not os.path.exists(filename):
            print(f"Data file {filename} not found.")
            return None, None
            
        data = []
        with open(filename, 'r') as f:
            for line in f:
                values = list(map(int, line.strip().split(',')))
                if len(values) >= 4:  # At least one position pair and target
                    features = values[:-2]
                    target = values[-2:]
                    data.append((features, target))
        
        if not data:
            print("No valid data found in file.")
            return None, None
            
        X = [item[0] for item in data]
        y = [item[1] for item in data]
        return X, y
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

def train_predictor_model(X, y, max_depth: int = 5):
    """Train a decision tree to predict player movement."""
    if X is None or y is None:
        print("No data available for training.")
        return None
        
    try:
        model = DecisionTreeClassifier(max_depth=max_depth)
        model.fit(X, y)
        print(f"Trained predictor model with {len(X)} samples.")
        return model
    except Exception as e:
        print(f"Error training predictor model: {e}")
        return None

def train_hotspot_model(positions: List[Tuple[int, int]], n_clusters: int = 4):
    """Train a clustering model to identify player hotspots."""
    if not positions:
        print("No position data available for clustering.")
        return None
        
    try:
        # Convert positions to numpy array
        X = np.array(positions)
        
        # Train KMeans model
        model = KMeans(n_clusters=n_clusters)
        model.fit(X)
        print(f"Trained hotspot model with {len(X)} positions and {n_clusters} clusters.")
        return model
    except Exception as e:
        print(f"Error training hotspot model: {e}")
        return None

def extract_positions_from_data(filename: str = "player_observations.csv"):
    """Extract all player positions from the data file."""
    positions = []
    try:
        if not os.path.exists(filename):
            print(f"Data file {filename} not found.")
            return positions
            
        with open(filename, 'r') as f:
            for line in f:
                values = list(map(int, line.strip().split(',')))
                # Add all positions including the target
                for i in range(0, len(values), 2):
                    if i + 1 < len(values):
                        positions.append((values[i], values[i+1]))
        
        return positions
    except Exception as e:
        print(f"Error extracting positions: {e}")
        return positions

def save_model(model, filename: str):
    """Save a trained model to a file."""
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved to {filename}")
    except Exception as e:
        print(f"Error saving model: {e}")

def train_and_save_models():
    """Train both models and save them."""
    # Create models directory if it doesn't exist
    os.makedirs("ml/models", exist_ok=True)
    
    # Train predictor model
    X, y = load_player_data()
    if X and y:
        predictor_model = train_predictor_model(X, y)
        if predictor_model:
            save_model(predictor_model, "ml/models/guard_predictor.pkl")
    
    # Train hotspot model
    positions = extract_positions_from_data()
    if positions:
        hotspot_model = train_hotspot_model(positions)
        if hotspot_model:
            save_model(hotspot_model, "ml/models/hotspots.pkl")

if __name__ == "__main__":
    train_and_save_models()
