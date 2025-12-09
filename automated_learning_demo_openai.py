import tkinter as tk
import random
import time
import threading

# --- Environment Setup ---
LANES = 3
ROAD_LENGTH = 15
car_lane = 1
car_pos = 0
epsilon = 1.0
alpha = 0.8
gamma = 0.9
episodes = 0
successes = 0

# Rewards
REWARD_GOAL = 10
REWARD_OBSTACLE = -10
REWARD_MOVE = -1

# Initialize Q-table
q_table = {}
def get_q(state):
    if state not in q_table:
        q_table[state] = [0, 0, 0]  # up, stay, down
    return q_table[state]

# --- Tkinter UI ---
root = tk.Tk()
root.title("Reinforcement Learning - Multi-Lane Road Simulation")
root.configure(bg="#1b1f3a")

canvas = tk.Canvas(root, width=800, height=200, bg="#101226", highlightthickness=0)
canvas.pack(pady=10)

info_frame = tk.Frame(root, bg="#1b1f3a")
info_frame.pack()

stats = {
    "Episode": tk.Label(info_frame, text="0", fg="white", bg="#6a0dad", font=("Arial", 16, "bold"), width=12),
    "Success Rate": tk.Label(info_frame, text="0.0%", fg="white", bg="#0f9d58", font=("Arial", 16, "bold"), width=12),
    "Avg Reward": tk.Label(info_frame, text="0.0", fg="white", bg="#4285f4", font=("Arial", 16, "bold"), width=12),
    "Exploration": tk.Label(info_frame, text="1.000", fg="white", bg="#fbbc04", font=("Arial", 16, "bold"), width=12),
}

for label in stats.values():
    label.pack(side=tk.LEFT, padx=5)

status_label = tk.Label(root, text="Status: Ready to train", fg="white", bg="#1b1f3a", font=("Arial", 12))
status_label.pack(pady=5)

button_frame = tk.Frame(root, bg="#1b1f3a")
button_frame.pack(pady=10)

def create_button(text, color, command):
    return tk.Button(button_frame, text=text, bg=color, fg="white", font=("Arial", 12, "bold"), width=12, command=command)

start_btn = create_button(" Start Training", "#0f9d58", lambda: threading.Thread(target=start_training).start())
demo_btn = create_button(" Demo", "#4285f4", lambda: threading.Thread(target=demo_run).start())
reset_btn = create_button(" Reset", "#5f6368", lambda: reset_env())
newroad_btn = create_button(" New Road", "#9c27b0", lambda: generate_new_road())

for btn in [start_btn, demo_btn, reset_btn, newroad_btn]:
    btn.pack(side=tk.LEFT, padx=5)

# --- Environment Drawing ---
car_icon = None
obstacles = []
goal_pos = ROAD_LENGTH - 1

def draw_env():
    canvas.delete("all")
    for lane in range(LANES):
        y = 60 * lane + 40
        canvas.create_line(0, y, 800, y, fill="#2b2d42", dash=(2,2))
    # Draw car
    x = car_pos * 50 + 50
    y = car_lane * 60 + 20
    global car_icon
    car_icon = canvas.create_rectangle(x-15, y-15, x+15, y+15, fill="#00bcd4")
    # Draw obstacles
    for obs in obstacles:
        ox = obs[1] * 50 + 50
        oy = obs[0] * 60 + 20
        canvas.create_rectangle(ox-10, oy-10, ox+10, oy+10, fill="#ff5252")
    # Draw goal
    gx = goal_pos * 50 + 50
    for lane in range(LANES):
        gy = lane * 60 + 20
        canvas.create_rectangle(gx-10, gy-10, gx+10, gy+10, fill="#4caf50")

def generate_new_road():
    global obstacles
    obstacles = []
    for _ in range(random.randint(3, 6)):
        lane = random.randint(0, LANES-1)
        pos = random.randint(2, ROAD_LENGTH-2)
        obstacles.append((lane, pos))
    draw_env()
    status_label.config(text="Status: New road generated")

def reset_env():
    global car_lane, car_pos, episodes, successes, epsilon
    car_lane, car_pos = 1, 0
    draw_env()
    status_label.config(text="Status: Environment reset")

# --- RL Training ---
def choose_action(state):
    if random.uniform(0,1) < epsilon:
        return random.randint(0, 2)
    return np.argmax(get_q(state))

def take_action(lane, pos, action):
    if action == 0 and lane > 0: lane -= 1
    elif action == 2 and lane < LANES-1: lane += 1
    pos += 1
    return lane, pos

import numpy as np
def start_training():
    global episodes, epsilon, successes
    status_label.config(text="Status: Training...")
    total_rewards = []
    for ep in range(50):
        lane, pos = 1, 0
        total_reward = 0
        while pos < ROAD_LENGTH - 1:
            state = (lane, pos)
            action = choose_action(state)
            next_lane, next_pos = take_action(lane, pos, action)

            reward = REWARD_MOVE
            if (next_lane, next_pos) in obstacles:
                reward = REWARD_OBSTACLE
                next_pos = ROAD_LENGTH - 1  # end
            elif next_pos == goal_pos:
                reward = REWARD_GOAL

            total_reward += reward
            q_vals = get_q(state)
            next_q = np.max(get_q((next_lane, next_pos)))
            q_vals[action] += alpha * (reward + gamma * next_q - q_vals[action])

            lane, pos = next_lane, next_pos
            draw_env()
            time.sleep(0.05)

            if reward == REWARD_OBSTACLE or reward == REWARD_GOAL:
                if reward == REWARD_GOAL:
                    successes += 1
                break

        episodes += 1
        epsilon = max(0.1, epsilon * 0.95)
        total_rewards.append(total_reward)
        avg_reward = np.mean(total_rewards[-10:])
        stats["Episode"].config(text=str(episodes))
        stats["Success Rate"].config(text=f"{(successes/episodes)*100:.1f}%")
        stats["Avg Reward"].config(text=f"{avg_reward:.1f}")
        stats["Exploration"].config(text=f"{epsilon:.3f}")
        root.update()

    status_label.config(text="Status: Training complete ✅")

def demo_run():
    global car_lane, car_pos
    status_label.config(text="Status: Demo running...")
    lane, pos = 1, 0
    while pos < ROAD_LENGTH - 1:
        state = (lane, pos)
        action = np.argmax(get_q(state))
        lane, pos = take_action(lane, pos, action)
        draw_env()
        time.sleep(0.2)
        if (lane, pos) in obstacles:
            status_label.config(text="Status: ❌ Crashed into obstacle!")
            return
    status_label.config(text="Status: 🏁 Successfully reached goal!")

# --- Initialize environment ---
generate_new_road()
draw_env()

root.mainloop()
