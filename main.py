import os
import sys
import time
import random
import curses
from typing import List, Tuple, Optional
import numpy as np

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from game.map import GameMap
from game.agent import Player, Guard
from ml.train import train_and_save_models

class Game:
    """Main game class that handles the game loop and rendering."""
    
    def __init__(self, width: int = 20, height: int = 20, num_guards: int = 3):
        self.width = width
        self.height = height
        self.num_guards = num_guards
        self.game_map = GameMap(width, height)
        self.player = None
        self.guards = []
        self.running = True
        self.win_condition = False
        self.game_over = False
        self.turn_count = 0
        self.max_turns = 300  # Game ends after this many turns
        self.objective_position = None
        self.use_curses = True  # Set to False to use print-based rendering
        self.screen = None
        self.data_collection = True  # Set to True to collect player data
        
    def setup(self):
        """Initialize the game state."""
        # Generate or load map
        self.game_map.generate_random_map(obstacle_density=0.2)
        
        # Create player at random position
        player_pos = self.game_map.get_random_empty_position()
        self.player = Player(player_pos[0], player_pos[1], self.game_map)
        
        # Create guards at random positions
        self.guards = []
        for _ in range(self.num_guards):
            guard_pos = self.game_map.get_random_empty_position()
            # Make sure guards don't start too close to the player
            while ((guard_pos[0] - player_pos[0])**2 + 
                   (guard_pos[1] - player_pos[1])**2 < 100):  # Min distance of 10 cells
                guard_pos = self.game_map.get_random_empty_position()
            
            self.guards.append(Guard(guard_pos[0], guard_pos[1], self.game_map))
        
        # Set objective position
        self.objective_position = self.game_map.get_random_empty_position()
        # Make sure objective is not too close to player or guards
        while ((self.objective_position[0] - player_pos[0])**2 + 
               (self.objective_position[1] - player_pos[1])**2 < 100):
            self.objective_position = self.game_map.get_random_empty_position()
    
    def process_input(self, key) -> bool:
        """Process user input and return True if a valid move was made."""
        if key == curses.KEY_UP or key == 'w':
            return self.player.move(0, -1)
        elif key == curses.KEY_DOWN or key == 's':
            return self.player.move(0, 1)
        elif key == curses.KEY_LEFT or key == 'a':
            return self.player.move(-1, 0)
        elif key == curses.KEY_RIGHT or key == 'd':
            return self.player.move(1, 0)
        elif key == ' ':  # Space to throw rock
            # Get direction from player input
            self.screen.addstr(self.height + 2, 0, "Throw direction (WASD): ")
            self.screen.refresh()
            dir_key = self.screen.getch()
            
            dx, dy = 0, 0
            if dir_key == ord('w'):
                dy = -3
            elif dir_key == ord('s'):
                dy = 3
            elif dir_key == ord('a'):
                dx = -3
            elif dir_key == ord('d'):
                dx = 3
            
            if dx != 0 or dy != 0:
                return self.player.throw_rock(dx, dy)
        
        return False
    
    def update(self):
        """Update game state for one turn."""
        self.turn_count += 1
        
        # Update player
        self.player.update()
        
        # Check if player reached objective
        if self.player.get_position() == self.objective_position:
            self.win_condition = True
            self.game_over = True
            return
        
        # Update guards
        for guard in self.guards:
            guard.update(self.player)
            
            # Check if guard caught player
            if guard.get_position() == self.player.get_position():
                self.win_condition = False
                self.game_over = True
                return
        
        # Check if max turns reached
        if self.turn_count >= self.max_turns:
            self.win_condition = False
            self.game_over = True
    
    def render_curses(self):
        """Render the game using curses."""
        self.screen.clear()
        
        # Render map and agents
        for y in range(self.game_map.height):
            for x in range(self.game_map.width):
                if self.game_map.is_obstacle(x, y):
                    self.screen.addch(y, x, '#')
                else:
                    self.screen.addch(y, x, '.')
        
        # Render objective
        if self.objective_position:
            self.screen.addch(self.objective_position[1], self.objective_position[0], 'O')
        
        # Render player
        px, py = self.player.get_position()
        self.screen.addch(py, px, 'P')
        
        # Render guards
        for guard in self.guards:
            gx, gy = guard.get_position()
            # Use different symbols based on guard state
            if guard.state == Guard.CHASE:
                self.screen.addch(gy, gx, '!')
            elif guard.state == Guard.INVESTIGATE:
                self.screen.addch(gy, gx, '?')
            else:
                self.screen.addch(gy, gx, 'G')
        
        # Render noise
        if self.player.noise_position:
            nx, ny = self.player.noise_position
            self.screen.addch(ny, nx, '*')
        
        # Render status
        self.screen.addstr(self.height, 0, f"Turn: {self.turn_count}/{self.max_turns}")
        self.screen.addstr(self.height + 1, 0, "Controls: Arrow keys to move, Space to throw rock")
        
        # Render guard states
        for i, guard in enumerate(self.guards):
            state_str = "Patrol"
            if guard.state == Guard.CHASE:
                state_str = "Chase"
            elif guard.state == Guard.INVESTIGATE:
                state_str = "Investigate"
            self.screen.addstr(i, self.width + 2, f"Guard {i+1}: {state_str} ({guard.suspicion:.1f})")
        
        self.screen.refresh()
    
    def render_print(self):
        """Render the game using print statements (fallback)."""
        os.system('cls' if os.name == 'nt' else 'clear')
        
        # Create a grid representation
        grid = [[' ' for _ in range(self.game_map.width)] for _ in range(self.game_map.height)]
        
        # Fill in obstacles
        for y in range(self.game_map.height):
            for x in range(self.game_map.width):
                if self.game_map.is_obstacle(x, y):
                    grid[y][x] = '#'
                else:
                    grid[y][x] = '.'
        
        # Add objective
        if self.objective_position:
            ox, oy = self.objective_position
            grid[oy][ox] = 'O'
        
        # Add player
        px, py = self.player.get_position()
        grid[py][px] = 'P'
        
        # Add guards
        for guard in self.guards:
            gx, gy = guard.get_position()
            if guard.state == Guard.CHASE:
                grid[gy][gx] = '!'
            elif guard.state == Guard.INVESTIGATE:
                grid[gy][gx] = '?'
            else:
                grid[gy][gx] = 'G'
        
        # Add noise
        if self.player.noise_position:
            nx, ny = self.player.noise_position
            if 0 <= ny < len(grid) and 0 <= nx < len(grid[0]):
                grid[ny][nx] = '*'
        
        # Print the grid
        for row in grid:
            print(''.join(row))
        
        # Print status
        print(f"Turn: {self.turn_count}/{self.max_turns}")
        print("Controls: WASD to move, Space to throw rock")
        
        # Print guard states
        for i, guard in enumerate(self.guards):
            state_str = "Patrol"
            if guard.state == Guard.CHASE:
                state_str = "Chase"
            elif guard.state == Guard.INVESTIGATE:
                state_str = "Investigate"
            print(f"Guard {i+1}: {state_str} (Suspicion: {guard.suspicion:.1f})")
    
    def render(self):
        """Render the game state."""
        if self.use_curses and self.screen:
            self.render_curses()
        else:
            self.render_print()
    
    def run_curses(self, stdscr):
        """Run the game loop with curses interface."""
        self.screen = stdscr
        curses.curs_set(0)  # Hide cursor
        stdscr.timeout(100)  # Non-blocking input with 100ms timeout
        
        self.setup()
        
        while self.running and not self.game_over:
            self.render()
            
            # Get user input
            key = stdscr.getch()
            
            if key == ord('q'):
                self.running = False
            elif key != -1:  # -1 is returned if no key is pressed
                if self.process_input(key):
                    self.update()
            
            # Small delay to control game speed
            time.sleep(0.1)
        
        # Game over screen
        stdscr.clear()
        if self.win_condition:
            stdscr.addstr(self.height // 2, self.width // 2 - 5, "YOU WIN!")
        else:
            stdscr.addstr(self.height // 2, self.width // 2 - 5, "GAME OVER")
        
        stdscr.addstr(self.height // 2 + 1, self.width // 2 - 10, f"Survived {self.turn_count} turns")
        stdscr.addstr(self.height // 2 + 2, self.width // 2 - 10, "Press any key to exit")
        stdscr.refresh()
        stdscr.getch()
        
        # Save player data for ML training
        if self.data_collection:
            for guard in self.guards:
                guard.save_observations()
    
    def run_print(self):
        """Run the game loop with print-based interface."""
        self.setup()
        
        while self.running and not self.game_over:
            self.render()
            
            # Get user input
            key = input("Enter move (WASD, space to throw, q to quit): ")
            
            if key == 'q':
                self.running = False
            elif key:
                if self.process_input(key):
                    self.update()
            
            # Small delay to control game speed
            time.sleep(0.1)
        
        # Game over screen
        os.system('cls' if os.name == 'nt' else 'clear')
        if self.win_condition:
            print("\n\n" + " " * 20 + "YOU WIN!")
        else:
            print("\n\n" + " " * 20 + "GAME OVER")
        
        print(" " * 15 + f"Survived {self.turn_count} turns")
        print(" " * 15 + "Press Enter to exit")
        input()
        
        # Save player data for ML training
        if self.data_collection:
            for guard in self.guards:
                guard.save_observations()
    
    def run(self):
        """Run the game."""
        if self.use_curses:
            try:
                curses.wrapper(self.run_curses)
            except Exception as e:
                print(f"Error running curses: {e}")
                self.use_curses = False
                self.run_print()
        else:
            self.run_print()

def main():
    """Main function to start the game."""
    # Check if we should train models first
    if len(sys.argv) > 1 and sys.argv[1] == "--train":
        print("Training ML models...")
        train_and_save_models()
        return
    
    # Create and run the game
    game = Game(width=20, height=20, num_guards=3)
    game.run()

if __name__ == "__main__":
    main()
