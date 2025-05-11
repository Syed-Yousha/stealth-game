import numpy as np
import random
from typing import List, Tuple, Set

class GameMap:
    """Represents the game map with obstacles and valid positions."""
    
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=int)
        self.obstacle_char = '#'
        self.empty_char = '.'
    
    def generate_random_map(self, obstacle_density: float = 0.2):
        """Generate a random map with obstacles."""
        self.grid = np.zeros((self.height, self.width), dtype=int)
        
        # Add random obstacles
        for y in range(self.height):
            for x in range(self.width):
                # Keep the edges clear
                if x == 0 or y == 0 or x == self.width - 1 or y == self.height - 1:
                    self.grid[y, x] = 0
                elif random.random() < obstacle_density:
                    self.grid[y, x] = 1
        
        # Ensure there's a path from top-left to bottom-right
        self._ensure_path()
    
    def _ensure_path(self):
        """Make sure there's at least one path through the map."""
        # Simple implementation: clear a path along the middle
        mid_x = self.width // 2
        mid_y = self.height // 2
        
        # Clear horizontal and vertical paths
        for x in range(self.width):
            self.grid[mid_y, x] = 0
        for y in range(self.height):
            self.grid[y, mid_x] = 0
    
    def load_from_file(self, filename: str):
        """Load a map from a text file."""
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()
            
            height = len(lines)
            width = max(len(line.strip()) for line in lines)
            
            self.height = height
            self.width = width
            self.grid = np.zeros((height, width), dtype=int)
            
            for y, line in enumerate(lines):
                for x, char in enumerate(line.strip()):
                    if char == self.obstacle_char:
                        self.grid[y, x] = 1
        except Exception as e:
            print(f"Error loading map: {e}")
            self.generate_random_map()
    
    def save_to_file(self, filename: str):
        """Save the current map to a text file."""
        try:
            with open(filename, 'w') as f:
                for y in range(self.height):
                    line = ''
                    for x in range(self.width):
                        if self.grid[y, x] == 1:
                            line += self.obstacle_char
                        else:
                            line += self.empty_char
                    f.write(line + '\n')
            print(f"Map saved to {filename}")
        except Exception as e:
            print(f"Error saving map: {e}")
    
    def is_valid_position(self, x: int, y: int) -> bool:
        """Check if a position is within the map bounds and not an obstacle."""
        return (0 <= x < self.width and 
                0 <= y < self.height and 
                not self.is_obstacle(x, y))
    
    def is_obstacle(self, x: int, y: int) -> bool:
        """Check if a position contains an obstacle."""
        return self.grid[y, x] == 1
    
    def get_random_empty_position(self) -> Tuple[int, int]:
        """Get a random position that is not an obstacle."""
        empty_positions = []
        for y in range(self.height):
            for x in range(self.width):
                if not self.is_obstacle(x, y):
                    empty_positions.append((x, y))
        
        if not empty_positions:
            # If no empty positions, clear a spot
            x, y = self.width // 2, self.height // 2
            self.grid[y, x] = 0
            return (x, y)
        
        return random.choice(empty_positions)
    
    def get_line(self, x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
        """Get all points on a line between two positions (Bresenham's algorithm)."""
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        while True:
            points.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
                
        return points
    
    def get_visible_positions(self, x: int, y: int, view_range: int) -> Set[Tuple[int, int]]:
        """Get all positions visible from a given position within view range."""
        visible = set()
        
        # Check all positions within the view range
        for dy in range(-view_range, view_range + 1):
            for dx in range(-view_range, view_range + 1):
                # Skip positions outside the view range
                if dx**2 + dy**2 > view_range**2:
                    continue
                
                target_x, target_y = x + dx, y + dy
                
                # Skip positions outside the map
                if not (0 <= target_x < self.width and 0 <= target_y < self.height):
                    continue
                
                # Check line of sight
                line = self.get_line(x, y, target_x, target_y)
                blocked = False
                
                for px, py in line[1:]:  # Skip the starting position
                    if self.is_obstacle(px, py):
                        blocked = True
                        break
                    if px == target_x and py == target_y:
                        break
                
                if not blocked:
                    visible.add((target_x, target_y))
        
        return visible
