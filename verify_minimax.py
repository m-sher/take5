import numpy as np
from minimax_agent import MinimaxAgent as PythonMinimaxAgent
import minimax_agent_c
import time

def create_random_observation(num_players=2):
    # Simulate a game state
    all_cards = np.arange(1, 105)
    np.random.shuffle(all_cards)
    
    table = np.zeros((4, 5), dtype=np.int64)
    table[:, 0] = all_cards[:4]
    
    hands = np.zeros((num_players, 10), dtype=np.int64)
    hands_cards = all_cards[4:4 + num_players * 10]
    hands[:, :] = hands_cards.reshape((num_players, 10))
        
    return {
        "table": table,
        "hands": hands,
        "num_players": np.array(num_players, dtype=np.int64)
    }

def verify():
    num_tests = 50
    depth = 2
    player_index = 0
    
    py_agent = PythonMinimaxAgent(depth=depth, player_index=player_index)
    
    print(f"Running {num_tests} verification tests at depth {depth}...")
    
    for i in range(num_tests):
        obs = create_random_observation()
        
        # Get Python result
        start_py = time.time()
        py_action = py_agent.choose_action(obs)
        py_time = time.time() - start_py
        
        # Get C result
        start_c = time.time()
        c_action = minimax_agent_c.choose_action(obs, depth, player_index)
        c_time = time.time() - start_c
        
        if py_action != c_action:
            print(f"Test {i} FAILED!")
            print(f"Observation:\nTable:\n{obs['table']}\nHands:\n{obs['hands']}")
            print(f"Python action: {py_action}")
            print(f"C action:      {c_action}")
            print(f"Python time: {py_time:.4f}s, C time: {c_time:.4f}s")
            raise AssertionError("Implementations produced different outputs")
        else:
            print(f"Test {i} passed! (Py: {py_time:.4f}s, C: {c_time:.4f}s, speedup: {py_time/c_time:.1f}x)")

    print(f"\nAll {num_tests} tests completed.")

if __name__ == "__main__":
    verify()
