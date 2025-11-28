import tkinter as tk
from tkinter import ttk
import random
from collections import defaultdict

class RLDrivingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("RL Autonomous Driving Demo")
        self.root.geometry("1200x600")
        self.root.configure(bg="#1a1a2e")
        
        self.ROAD_LENGTH, self.NUM_LANES, self.GOAL = 15, 3, 14
        self.obstacles = self.generate_obstacles()
        self.q_table = defaultdict(float)
        self.episode = self.success_count = 0
        self.reward_history = []
        self.epsilon = 1.0
        self.is_training = False
        self.car_position = self.car_lane = 0
        self.reset_episode_state()
        
        self.max_episodes = 1200
        self.alpha, self.gamma = 0.1, 0.95
        self.epsilon_decay, self.min_epsilon = 0.995, 0.01
        
        self.setup_ui()
        
    def generate_obstacles(self):
        obstacles = set()
        for pos in range(3, self.ROAD_LENGTH - 1):
            if random.random() < 0.2:
                num_blocked = random.randint(1, 2)
                lanes = random.sample([0, 1, 2], num_blocked)
                obstacles.update((pos, lane) for lane in lanes)
        return obstacles
    
    def setup_ui(self):
        title_frame = tk.Frame(self.root, bg="#1a1a2e")
        title_frame.pack(pady=10)
        tk.Label(title_frame, text="Reinforcement Learning in Autonomous Driving", 
                font=("Arial", 18, "bold"), bg="#1a1a2e", fg="white").pack()
        
        road_frame = tk.Frame(self.root, bg="#2d2d44", relief=tk.RIDGE, borderwidth=3)
        road_frame.pack(pady=10, padx=40, fill=tk.BOTH)
        
        self.road_canvas = tk.Canvas(road_frame, width=1100, height=180, bg="#1a1a2e", highlightthickness=0)
        self.road_canvas.pack(pady=10)
        
        self.status_label = tk.Label(road_frame, text="Status: Ready to train", 
                                     font=("Arial", 10, "bold"), bg="#2d2d44", fg="#4a9eff")
        self.status_label.pack(pady=5)
        
        metrics_frame = tk.Frame(self.root, bg="#1a1a2e")
        metrics_frame.pack(pady=8, padx=40, fill=tk.X)
        
        self.metric_frames = []
        metrics = [("Episodes", "0", "#9333ea"), ("Success Rate", "0.0%", "#16a34a"),
                   ("Avg Reward", "0.0", "#2563eb"), ("Epsilon", "1.000", "#ea580c")]
        
        for i, (label, value, color) in enumerate(metrics):
            frame = tk.Frame(metrics_frame, bg=color, relief=tk.RAISED, borderwidth=2)
            frame.grid(row=0, column=i, padx=5, sticky="ew")
            metrics_frame.columnconfigure(i, weight=1)
            tk.Label(frame, text=label, font=("Arial", 9), bg=color, fg="white").pack(pady=3)
            metric_label = tk.Label(frame, text=value, font=("Arial", 18, "bold"), bg=color, fg="white")
            metric_label.pack(pady=3)
            self.metric_frames.append(metric_label)
        
        button_frame = tk.Frame(self.root, bg="#1a1a2e")
        button_frame.pack(pady=15)
        
        buttons = [
            ("Start Training", self.toggle_training, "#16a34a", 15),
            ("Demo Policy", self.demonstrate, "#2563eb", 12),
            ("Reset", self.reset, "#4b5563", 12),
            ("New Road", self.generate_new_road, "#7c3aed", 12)
        ]
        
        for i, (text, cmd, bg, width) in enumerate(buttons):
            btn = tk.Button(button_frame, text=text, command=cmd, font=("Arial", 13, "bold"),
                          bg=bg, fg="white", width=width, height=2, relief=tk.RAISED, borderwidth=3)
            btn.grid(row=0, column=i, padx=5)
            if i == 0:
                self.train_btn = btn
            elif i == 1:
                self.demo_btn = btn
        
        self.draw_road()
    
    def draw_road(self):
        self.road_canvas.delete("all")
        cell_w, lane_h = 50, 50
        
        for lane in range(self.NUM_LANES):
            y = 40 + lane * lane_h
            self.road_canvas.create_line(50, y - 20, 50 + self.ROAD_LENGTH * cell_w, y - 20, 
                                        fill="#4a4a6a", width=2, dash=(5, 5))
        
        for pos in range(self.ROAD_LENGTH):
            x = 50 + pos * cell_w
            for lane in range(self.NUM_LANES):
                y = 40 + lane * lane_h
                if pos == self.car_position and lane == self.car_lane:
                    self.road_canvas.create_oval(x-8, y-8, x+8, y+8, fill="#00ff00", outline="white", width=2)
                elif (pos, lane) in self.obstacles:
                    self.road_canvas.create_rectangle(x-10, y-10, x+10, y+10, fill="#ff4444", outline="#cc0000", width=2)
                elif pos == self.GOAL:
                    self.road_canvas.create_rectangle(x-12, y-20, x+12, y+20, fill="#ffcc00", outline="#ff9900", width=2)
        
        for lane in range(self.NUM_LANES):
            y = 40 + lane * lane_h
            self.road_canvas.create_text(20, y, text=f"L{lane}", 
                                        font=("Arial", 9), fill="#a0a0a0", anchor="e")
        self.root.update()
    
    def get_obstacles_ahead(self, pos):
        result = []
        for lane in range(self.NUM_LANES):
            obs = tuple(sorted([cp - pos for cp in range(pos + 1, min(pos + 4, self.ROAD_LENGTH)) 
                               if (cp, lane) in self.obstacles]))
            result.append(obs if obs else (0,))
        return tuple(result)
    
    def get_state(self, pos, lane):
        return f"{pos}-{lane}-{self.get_obstacles_ahead(pos)}"
    
    def get_q(self, state, action):
        return self.q_table.get(f"{state}-{action}", 0.0)
    
    def set_q(self, state, action, value):
        self.q_table[f"{state}-{action}"] = value
    
    def choose_action(self, state, eps):
        if random.random() < eps:
            return random.randint(0, 3)
        q_vals = [self.get_q(state, a) for a in range(4)]
        max_q = max(q_vals)
        return random.choice([i for i, q in enumerate(q_vals) if q == max_q])
    
    def take_action(self, pos, lane, action):
        return (pos + 1, lane, 1) if action == 0 else (pos, action - 1, 0)
    
    def calc_reward(self, pos, lane, base):
        reward = base - 0.05
        if (pos, lane) in self.obstacles:
            return -100, True
        if pos >= self.GOAL:
            return reward + 100, True
        if pos < 0 or pos > self.ROAD_LENGTH:
            return -50, True
        return reward, False

    
    def reset_episode_state(self):
        self.ep_pos = self.ep_lane = 0
        self.ep_steps = self.ep_reward = 0
        self.ep_done = False
        self.car_position, self.car_lane = 0, 1
        self.ep_pos, self.ep_lane = 0, 1
    
    def training_step(self):
        
        
        if not self.is_training:
            return
        if self.ep_done or self.ep_steps >= 100:
            if self.ep_pos >= self.GOAL:
                self.success_count += 1
            
            self.reward_history.append(self.ep_reward)
            self.episode += 1
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            
            sr = (self.success_count / self.episode) * 100
            recent = self.reward_history[-50:] if len(self.reward_history) >= 50 else self.reward_history
            avg_r = sum(recent) / len(recent) if recent else 0
            
            self.metric_frames[0].config(text=str(self.episode))
            self.metric_frames[1].config(text=f"{sr:.1f}%")
            self.metric_frames[2].config(text=f"{avg_r:.1f}")
            self.metric_frames[3].config(text=f"{self.epsilon:.3f}")
            
            if self.episode >= self.max_episodes:
                self.is_training = False
                self.train_btn.config(text="Start Training", bg="#16a34a")
                self.status_label.config(text="Status: Training Complete!", fg="#16a34a")
                self.demo_btn.config(state=tk.NORMAL)
                return
            
            self.reset_episode_state()
        
        state = self.get_state(self.ep_pos, self.ep_lane)
        action = self.choose_action(state, self.epsilon)
        new_pos, new_lane, act_r = self.take_action(self.ep_pos, self.ep_lane, action)
        
        self.ep_pos, self.ep_lane = new_pos, new_lane
        reward, done = self.calc_reward(self.ep_pos, self.ep_lane, act_r)
        self.ep_reward += reward
        self.ep_done = done
        
        next_state = self.get_state(self.ep_pos, self.ep_lane)
        curr_q = self.get_q(state, action)
        max_next_q = max([self.get_q(next_state, a) for a in range(4)])
        target = reward if done else reward + self.gamma * max_next_q
        self.set_q(state, action, curr_q + self.alpha * (target - curr_q))
        
        self.ep_steps += 1
        self.car_position, self.car_lane = self.ep_pos, self.ep_lane
        self.draw_road()
        self.root.after(20, self.training_step)
    
    def toggle_training(self):
        self.is_training = not self.is_training
        if self.is_training:
            self.train_btn.config(text="Stop Training", bg="#dc2626")
            self.status_label.config(text="Status: Training in progress...", fg="#eab308")
            self.demo_btn.config(state=tk.DISABLED)
            self.reset_episode_state()
            self.training_step()
        else:
            self.train_btn.config(text="Start Training", bg="#16a34a")
            self.status_label.config(text="Status: Training paused", fg="#4a9eff")
            self.demo_btn.config(state=tk.NORMAL)
    
    def demonstrate(self):
        if not self.q_table:
            self.status_label.config(text="Status: Train first before demo!", fg="#dc2626")
            return
        
        self.status_label.config(text="Status: Demonstrating learned policy...", fg="#eab308")
        self.demo_btn.config(state=tk.DISABLED)
        self.train_btn.config(state=tk.DISABLED)
        self.car_position, self.car_lane = 0, 1
        self.draw_road()
        self.root.after(300, self.demo_step)
    
    def demo_step(self):
        if (self.car_position >= self.GOAL or 
            (self.car_position, self.car_lane) in self.obstacles or 
            self.car_position > self.ROAD_LENGTH):
            self.draw_road()
            msg = "Status: Demo completed successfully" if self.car_position >= self.GOAL else "Status: Demo failed"
            color = "#16a34a" if self.car_position >= self.GOAL else "#dc2626"
            self.status_label.config(text=msg, fg=color)
            self.demo_btn.config(state=tk.NORMAL)
            self.train_btn.config(state=tk.NORMAL)
            self.car_position, self.car_lane = 0, 1
            self.draw_road()
            return
        
        state = self.get_state(self.car_position, self.car_lane)
        action = self.choose_action(state, 0.0)
        new_pos, new_lane, _ = self.take_action(self.car_position, self.car_lane, action)
        self.car_position, self.car_lane = new_pos, new_lane
        self.draw_road()
        self.root.after(300, self.demo_step)
    
    def generate_new_road(self):
        self.obstacles = self.generate_obstacles()
        self.reset(keep_q=False)
        self.status_label.config(text="Status: New road generated!", fg="#7c3aed")
    
    def reset(self, keep_q=False):
        self.is_training = False
        self.episode = self.success_count = 0
        self.reward_history = []
        self.epsilon = 1.0
        if not keep_q:
            self.q_table.clear()
        self.reset_episode_state()
        
        for i, val in enumerate(["0", "0.0%", "0.0", "1.000"]):
            self.metric_frames[i].config(text=val)
        
        self.train_btn.config(text="Start Training", bg="#16a34a", state=tk.NORMAL)
        self.demo_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Status: Ready to train", fg="#4a9eff")
        self.draw_road()

if __name__ == "__main__":
    root = tk.Tk()
    app = RLDrivingGUI(root)
    root.mainloop()