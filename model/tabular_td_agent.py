import numpy as np
import pickle
import os

class TabularTDAgent:
    def __init__(self, alpha=0.1, gamma=0.99, n_actions=9, epsilon=1.0, 
                epsilon_end=0.01, epsilon_dec=0.999, state_bins=10,
                convergence_threshold=1e-5, convergence_window=50):
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.n_actions = n_actions
        self.actions = [i for i in range(n_actions)]
        self.epsilon = epsilon  # exploration rate
        self.epsilon_end = epsilon_end
        self.epsilon_dec = epsilon_dec
        
        # Number of bins for each state dimension
        self.state_bins = state_bins
        
        # Initialize Q-table
        # Since our state space is continuous, we need to discretize it
        # We'll use a dictionary to store Q-values
        self.q_table = {}
        
        # Convergence tracking parameters
        self.convergence_threshold = convergence_threshold
        self.convergence_window = convergence_window
        self.q_changes = []  # Track changes in Q-values
        self.is_converged = False  # Convergence flag
        
    def discretize_state(self, state):
        """Convert continuous state to discrete state for Q-table lookup"""
        # Ensure state is a numpy array and properly shaped
        state = np.array(state).flatten()  # Convert to 1D array
        
        discrete_state = []
        for i, value in enumerate(state):
            # Handle None values or NaN values
            if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)):
                # Default to 0 for None or NaN values
                bin_value = 0
            else:
                # Ensure value is in valid range [0, 1]
                safe_value = max(0, min(1, float(value)))
                # Discretize each dimension into bins
                bin_value = min(self.state_bins - 1, int(safe_value * self.state_bins))
            
            discrete_state.append(bin_value)
            
        # Convert list to tuple so it can be used as dictionary key
        return tuple(discrete_state)
  
    def choose_action(self, state):
        """Choose action using epsilon-greedy policy"""
        discrete_state = self.discretize_state(state)
        
        # Exploration
        if np.random.random() < self.epsilon:
            return np.random.choice(self.actions)
        
        # Exploitation
        # If we haven't seen this state before, initialize Q-values to zeros
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = np.zeros(self.n_actions)
            
        # Return action with highest Q-value
        return np.argmax(self.q_table[discrete_state])
    
    def learn(self, state, action, reward, next_state, done):
        """Update Q-value using TD learning"""
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)
        
        # Initialize Q-values for states if not seen before
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = np.zeros(self.n_actions)
        if discrete_next_state not in self.q_table:
            self.q_table[discrete_next_state] = np.zeros(self.n_actions)
            
        # Current Q-value estimate
        current_q = self.q_table[discrete_state][action]
        
        # Calculate TD target
        if done:
            target = reward
        else:
            # TD Target = reward + gamma * max(Q(s',a'))
            max_next_q = np.max(self.q_table[discrete_next_state])
            target = reward + self.gamma * max_next_q
            
        # Update Q-value
        # Q(s,a) = Q(s,a) + alpha * [target - Q(s,a)]
        q_change = self.alpha * (target - current_q)
        self.q_table[discrete_state][action] = current_q + q_change
        
        # Track absolute Q-value change for convergence detection
        self.q_changes.append(abs(q_change))
        if len(self.q_changes) > self.convergence_window:
            self.q_changes.pop(0)  # Remove oldest entry to maintain window size
        
        # Decay epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_dec
            
    def check_convergence(self):
        """Check if learning has converged based on recent Q-value changes"""
        if len(self.q_changes) < self.convergence_window:
            return False
            
        # Calculate average Q-value change over the window
        avg_change = np.mean(self.q_changes)
        
        # Check if average change is below threshold
        if avg_change < self.convergence_threshold:
            self.is_converged = True
            return True
        return False
    
    def save_model(self):
        """Save Q-table to file"""
        filepath = "./model/td_agent_qtable.pkl"
        with open(filepath, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f"Q-table saved with {len(self.q_table)} states to {filepath}")
    
    def load_model(self):
        """Load Q-table from file"""
        filepath = "./model/td_agent_qtable.pkl"  
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                self.q_table = pickle.load(f)
            print(f"Q-table loaded with {len(self.q_table)} states from {filepath}")
        else:
            print(f"No saved model found at {filepath}")
            self.q_table = {}
