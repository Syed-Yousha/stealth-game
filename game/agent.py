import numpy as np
import random
from typing import List, Tuple, Optional
import pickle
import os

class Agent:
    """Base class for all agents in the game."""
    
    def __init__(self, x: int, y: int, game_map):
        self.x = x
        self.y = y
        self.game_map = game_map
        self.symbol = '?'  # Default symbol
        
    def move(self, dx: int, dy: int) -> bool:
        """Move the agent by the given delta if possible."""
        new_x, new_y = self.x + dx, self.y + dy
        
        # Check if the new position is valid
        if self.game_map.is_valid_position(new_x, new_y):
            self.x, self.y = new_x, new_y
            return True
        return False
    
    def get_position(self) -> Tuple[int, int]:
        """Return the current position of the agent."""
        return (self.x, self.y)


class Player(Agent):
    """Player class controlled by the user."""
    
    def __init__(self, x: int, y: int, game_map):
        super().__init__(x, y, game_map)
        self.symbol = 'P'
        self.position_history = []  # Store last N positions for ML
        self.max_history = 5  # Number of positions to remember
        self.throw_cooldown = 0
        self.noise_position = None
        self.noise_duration = 0
        
    def update(self):
        """Update player state each game tick."""
        # Add current position to history
        self.position_history.append((self.x, self.y))
        if len(self.position_history) > self.max_history:
            self.position_history.pop(0)
            
        # Update cooldowns
        if self.throw_cooldown > 0:
            self.throw_cooldown -= 1
            
        # Update noise duration
        if self.noise_duration > 0:
            self.noise_duration -= 1
        else:
            self.noise_position = None
    
    def throw_rock(self, dx: int, dy: int) -> bool:
        """Throw a rock to create a distraction."""
        if self.throw_cooldown > 0:
            return False
        
        # Calculate target position (limited range of 5 cells)
        max_range = 5
        magnitude = (dx**2 + dy**2)**0.5
        if magnitude > 0:
            dx = int(dx * min(max_range, magnitude) / magnitude)
            dy = int(dy * min(max_range, magnitude) / magnitude)
        
        target_x, target_y = self.x + dx, self.y + dy
        
        # Check if target position is valid
        if self.game_map.is_valid_position(target_x, target_y):
            self.noise_position = (target_x, target_y)
            self.noise_duration = 5  # Noise lasts for 5 turns
            self.throw_cooldown = 10  # Cooldown before next throw
            return True
        return False


class Guard(Agent):
    """AI-controlled guard that hunts the player."""
    
    # Guard states
    PATROL = 0
    INVESTIGATE = 1
    CHASE = 2
    
    def __init__(self, x: int, y: int, game_map, model_dir: str = "ml/models"):
        super().__init__(x, y, game_map)
        self.symbol = 'G'
        self.state = self.PATROL
        self.suspicion = 0.0  # 0.0 to 1.0
        self.patrol_target = None
        self.last_player_position = None
        self.fov_range = 5  # How far the guard can see
        self.patrol_counter = 0
        self.model_dir = model_dir
        
        # Load ML models if available
        self.predictor_model = None
        self.hotspot_model = None
        self.load_models()
        
        # For data collection
        self.player_observations = []
        
    def load_models(self):
        """Load the trained ML models if they exist."""
        try:
            predictor_path = os.path.join(self.model_dir, "guard_predictor.pkl")
            if os.path.exists(predictor_path):
                with open(predictor_path, 'rb') as f:
                    self.predictor_model = pickle.load(f)
                print("Loaded predictor model")
                
            hotspot_path = os.path.join(self.model_dir, "hotspots.pkl")
            if os.path.exists(hotspot_path):
                with open(hotspot_path, 'rb') as f:
                    self.hotspot_model = pickle.load(f)
                print("Loaded hotspot model")
        except Exception as e:
            print(f"Error loading models: {e}")
    
    def update(self, player):
        """Update guard behavior based on current state and player position."""
        # Record player position if visible (for training data)
        if self.can_see_player(player):
            self.player_observations.append(player.position_history[-min(len(player.position_history), 5):])
            
        # Check if player is in line of sight
        if self.can_see_player(player):
            self.last_player_position = player.get_position()
            self.state = self.CHASE
            self.suspicion = 1.0
        # Check if there's a noise to investigate
        elif player.noise_position is not None:
            self.state = self.INVESTIGATE
            self.last_player_position = player.noise_position
            self.suspicion = 0.7
        # Gradually decrease suspicion
        elif self.suspicion > 0:
            self.suspicion -= 0.05
            if self.suspicion <= 0:
                self.state = self.PATROL
                self.last_player_position = None
        
        # Update patrol target periodically
        self.patrol_counter += 1
        if self.patrol_counter >= 30:  # Every 30 turns
            self.patrol_counter = 0
            self.update_patrol_target(player)
        
        # Move based on current state
        if self.state == self.CHASE and self.last_player_position:
            self.move_towards(self.last_player_position)
        elif self.state == self.INVESTIGATE and self.last_player_position:
            self.move_towards(self.last_player_position)
        else:  # PATROL state
            if self.patrol_target is None or self.get_position() == self.patrol_target:
                self.patrol_target = self.get_random_patrol_point()
            
            # If we have a predictor model, bias movement towards predicted player position
            if self.predictor_model is not None and len(player.position_history) >= 3:
                predicted_pos = self.predict_player_position(player.position_history)
                if predicted_pos:
                    # 30% chance to move towards predicted position, 70% towards patrol target
                    if random.random() < 0.3:
                        self.move_towards(predicted_pos)
                        return
            
            # Default patrol movement
            self.move_towards(self.patrol_target)
    
    def can_see_player(self, player) -> bool:
        """Check if the guard can see the player (line of sight)."""
        px, py = player.get_position()
        gx, gy = self.get_position()
        
        # Check distance
        distance = ((px - gx) ** 2 + (py - gy) ** 2) ** 0.5
        if distance > self.fov_range:
            return False
        
        # Check line of sight (Bresenham's line algorithm)
        line = self.game_map.get_line(gx, gy, px, py)
        for x, y in line[1:-1]:  # Skip first (guard) and last (player) positions
            if not self.game_map.is_valid_position(x, y) or self.game_map.is_obstacle(x, y):
                return False
        
        return True
    
    def move_towards(self, target_pos: Tuple[int, int]):
        """Move towards the target position."""
        tx, ty = target_pos
        gx, gy = self.get_position()
        
        # Calculate direction
        dx = 0 if tx == gx else (1 if tx > gx else -1)
        dy = 0 if ty == gy else (1 if ty > gy else -1)
        
        # Try to move in the primary direction first
        if abs(tx - gx) > abs(ty - gy):
            if not self.move(dx, 0):
                self.move(0, dy)
        else:
            if not self.move(0, dy):
                self.move(dx, 0)
    
    def get_random_patrol_point(self) -> Tuple[int, int]:
        """Get a random valid position for patrolling."""
        max_attempts = 20
        for _ in range(max_attempts):
            # Stay within a reasonable distance of current position
            dx = random.randint(-8, 8)
            dy = random.randint(-8, 8)
            new_x, new_y = self.x + dx, self.y + dy
            
            if self.game_map.is_valid_position(new_x, new_y) and not self.game_map.is_obstacle(new_x, new_y):
                return (new_x, new_y)
        
        # If no valid position found, return current position
        return self.get_position()
    
    def update_patrol_target(self, player):
        """Update patrol target based on hotspot model or random position."""
        if self.hotspot_model is not None:
            # Use clustering model to find hotspots
            try:
                # Get a random hotspot from the model
                hotspots = self.hotspot_model.cluster_centers_
                if len(hotspots) > 0:
                    # Choose a hotspot with higher probability for ones with more player visits
                    hotspot = tuple(map(int, random.choice(hotspots)))
                    if self.game_map.is_valid_position(hotspot[0], hotspot[1]) and not self.game_map.is_obstacle(hotspot[0], hotspot[1]):
                        self.patrol_target = hotspot
                        return
            except Exception as e:
                print(f"Error using hotspot model: {e}")
        
        # Fallback to random patrol point
        self.patrol_target = self.get_random_patrol_point()
    
    def predict_player_position(self, player_history):
        """Use the predictor model to guess the player's next position."""
        if self.predictor_model is None or len(player_history) < 3:
            return None
        
        try:
            # Format the input for the model
            # We need to flatten the last few positions into a single feature vector
            flattened_history = []
            for pos in player_history[-3:]:  # Use last 3 positions
                flattened_history.extend([pos[0], pos[1]])
            
            # Make prediction
            prediction = self.predictor_model.predict([flattened_history])
            predicted_x, predicted_y = prediction[0]
            
            # Ensure prediction is within map bounds
            predicted_x = max(0, min(int(predicted_x), self.game_map.width - 1))
            predicted_y = max(0, min(int(predicted_y), self.game_map.height - 1))
            
            return (predicted_x, predicted_y)
        except Exception as e:
            print(f"Error predicting player position: {e}")
            return None
    
    def save_observations(self, filename: str = "player_observations.csv"):
        """Save player observations for training."""
        if not self.player_observations:
            return
            
        try:
            with open(filename, 'w') as f:
                for obs in self.player_observations:
                    if len(obs) >= 2:  # Need at least 2 positions to predict movement
                        # Last position is the target, previous positions are features
                        features = ','.join([f"{x},{y}" for x, y in obs[:-1]])
                        target_x, target_y = obs[-1]
                        f.write(f"{features},{target_x},{target_y}\n")
            print(f"Saved {len(self.player_observations)} observations to {filename}")
        except Exception as e:
            print(f"Error saving observations: {e}")
