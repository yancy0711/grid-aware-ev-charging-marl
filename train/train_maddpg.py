# train_maddpg.py is the main training script for the MADDPG algorithm, this script handles the training loop and logging for 3 different training strategies.
# Example command to run this script:
# python train/train_maddpg.py --train_mode weekday --pkl_weekday ./data/events_augmented_weekday.pkl --episodes 500
import os, argparse, numpy as np, torch
from tqdm import trange
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env import CsvChargingEnv
from agents.maddpg import MADDPG
from rl_charge_extensions import DelayedReplayBuffer
import pickle, time, logging
import matplotlib.pyplot as plt
import random
import logging
from datetime import datetime

# add data augmentation strategy in script and support weekday/weekend/mixed training
def select_training_data(events_by_day, episode, strategy='mixed', weekday_data=None, weekend_data=None):
    """
    Select training data, balance sparse and dense data, support weekday/weekend mixed training
    Returns: (day_events, data_source) where data_source is 'weekday', 'weekend', or 'mixed'
    """
    # if weekday_data and weekend_data are provided, use mixed strategy
    if weekday_data is not None and weekend_data is not None:
        # Dynamic mixing ratio: focus on weekdays in the early stage, balance later
        if episode < len(events_by_day) // 4:
            # First 25%: 70% weekday, 30% weekend
            if np.random.random() < 0.7:
                return weekday_data[episode % len(weekday_data)], 'weekday'
            else:
                return weekend_data[episode % len(weekend_data)], 'weekend'
        elif episode < len(events_by_day) // 2:
            # 25%-50%: 60% weekday, 40% weekend
            if np.random.random() < 0.6:
                return weekday_data[episode % len(weekday_data)], 'weekday'
            else:
                return weekend_data[episode % len(weekend_data)], 'weekend'
        else:
            # Last 50%: 50% weekday, 50% weekend (balance training)
            if np.random.random() < 0.5:
                return weekday_data[episode % len(weekday_data)], 'weekday'
            else:
                return weekend_data[episode % len(weekend_data)], 'weekend'
    # If no separated data is available, use all weekday
    return events_by_day[episode % len(events_by_day)], 'weekday'


def get_curriculum_params(episode, total_episodes):
    """Curriculum Learning: gradually increase difficulty"""
    progress = episode / total_episodes
    
    if progress < 0.2:
        # First stage: learn basic acceptance/rejection
        return {
            'peak_penalty_scale': 0.5,
            'smoothness_weight': 0.1,
            'redirect_enabled': True,
            'future_horizon': 16
        }
    elif progress < 0.4:
        # Second stage: introduce smoothness requirements
        return {
            'peak_penalty_scale': 1.0,
            'smoothness_weight': 0.5,
            'redirect_enabled': True,
            'future_horizon': 16
        }
    elif progress < 0.6:
        # Third stage: enable redirection
        return {
            'peak_penalty_scale': 1.5,
            'smoothness_weight': 1.0,
            'redirect_enabled': True,
            'future_horizon': 16
        }
    else:
        # Fourth stage: full difficulty
        return {
            'peak_penalty_scale': 2.0,
            'smoothness_weight': 1.5,
            'redirect_enabled': True,
            'future_horizon': 16
        }

# Add real-time monitoring in the training script
class RedirectMonitor:
    """Monitor the effectiveness of redirection"""
    def __init__(self, n_sites):
        self.n_sites = n_sites
        self.reset()
    
    def reset(self):
        self.redirect_counts = np.zeros((self.n_sites, self.n_sites))
        self.load_before_redirect = []
        self.load_after_redirect = []
        self.queue_balance_improvements = []
        self.successful_redirects = 0
        self.total_redirects = 0
        self.effective_redirects = 0
    
    def record_redirect(self, from_site, to_site, load_before, load_after, queue_before, queue_after):
        """Record a redirection event"""
        self.redirect_counts[from_site, to_site] += 1
        self.total_redirects += 1

        # Calculate improvements
        load_std_before = np.std(load_before)
        load_std_after = np.std(load_after)
        queue_std_before = np.std(queue_before)
        queue_std_after = np.std(queue_after)
        
        self.load_before_redirect.append(load_std_before)
        self.load_after_redirect.append(load_std_after)

        # If improvements are made in load or queue balance, consider it a successful redirection
        if load_std_after < load_std_before or queue_std_after < queue_std_before:
            self.successful_redirects += 1
            self.effective_redirects += 1
            self.queue_balance_improvements.append(queue_std_before - queue_std_after)
    
    def get_summary(self):
        """Get summary of redirection effectiveness"""
        if self.total_redirects == 0:
            return {
                'total_redirects': 0,
                'success_rate': 0,
                'effective_redirects': 0,
                'avg_load_improvement': 0,
                'avg_queue_improvement': 0,
                'most_common_routes': []
            }

        # Success rate
        success_rate = self.successful_redirects / self.total_redirects

        # Average improvement
        if self.load_before_redirect:
            avg_load_imp = np.mean([
                (before - after) / before if before > 0 else 0
                for before, after in zip(self.load_before_redirect, self.load_after_redirect)
            ])
        else:
            avg_load_imp = 0
        
        avg_queue_imp = np.mean(self.queue_balance_improvements) if self.queue_balance_improvements else 0

        # Most common redirection routes
        redirect_pairs = []
        for i in range(self.n_sites):
            for j in range(self.n_sites):
                if self.redirect_counts[i, j] > 0:
                    redirect_pairs.append((i, j, self.redirect_counts[i, j]))
        
        redirect_pairs.sort(key=lambda x: x[2], reverse=True)
        most_common = redirect_pairs[:5]
        
        return {
            'total_redirects': self.total_redirects,
            'success_rate': success_rate,
            'effective_redirects': self.effective_redirects,
            'avg_load_improvement': avg_load_imp,
            'avg_queue_improvement': avg_queue_imp,
            'most_common_routes': most_common
        }

# Environment and Agent initialization
# Pass new parameters during environment initialization
class EnvArgs:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.enable_cross_site = args.enable_cross_site
        self.redirect_radius = args.redirect_radius
        self.redirect_penalty = args.redirect_penalty
        self.charger_kw = 22.0
        self.beta_peak = args.beta_peak
        self.peak_thresh = args.peak_thresh
        self.redirect_bonus = args.redirect_bonus if hasattr(args, 'redirect_bonus') else 100.0
# ----------------------Mini Config----------------------
# Mini configuration class for simplified parameter settings, centralized management of training parameters for easier modification and expansion
class Mini:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gamma = 0.99
        self.tau = 1e-2           # Slightly increase soft update rate
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.beta_entropy = args.beta_entropy
        self.lambda_match = args.lambda_match
        self.target_sigma = args.target_sigma
        self.batch_size = 64     
        self.buffer_size = 100000 
        self.hiddim = 64          
        self.beta_peak = args.beta_peak
        self.peak_thresh = args.peak_thresh
# ----------------------Training Setup----------------------
parser = argparse.ArgumentParser()
parser.add_argument('--pkl', type=str, default='./data/events_augmented.pkl')
parser.add_argument('--pkl_weekday', type=str, default='./data/events_augmented_weekday.pkl', help='weekday data file')
parser.add_argument('--pkl_weekend', type=str, default='./data/events_augmented_weekend.pkl', help='weekend data file')
parser.add_argument('--mixed_training', action='store_true', default=True, help='Whether to use mixed training for weekdays/weekends')
parser.add_argument('--train_mode', type=str, default='mixed', choices=['mixed', 'weekday', 'weekend'], help='Training mode: mixed/weekday/weekend')
parser.add_argument('--logdir', type=str, default='./logs')
parser.add_argument('--save_dir', type=str, default='./params')
parser.add_argument('--episodes', type=int, default=10)  # Increase the number of training episodes
parser.add_argument('--seed', type=int, default=42)

parser.add_argument('--actor_lr', type=float, default=1e-5)    
parser.add_argument('--critic_lr', type=float, default=1e-5)   # Slightly higher critic learning rate
parser.add_argument('--beta_entropy', type=float, default=0.01)  # Moderate entropy to reduce exploration and increase exploitation
parser.add_argument('--tau', type=float, default=0.005)  # Slower soft update
parser.add_argument('--lambda_match', type=float, default=0.1)   # Lower matching loss weight, higher means more focus on matching loss
parser.add_argument('--target_sigma', type=float, default=0.05)  # Lower target noise
parser.add_argument('--beta_peak', type=float, default=500.0)     # Lower peak penalty
parser.add_argument('--peak_thresh', type=float, default=0.7)   # Raise peak threshold
# New diversion-related parameters
parser.add_argument('--redirect_bonus', type=float, default=150.0)  # Additional reward for successful diversion

# New parameters
parser.add_argument('--warmup_episodes', type=int, default=5)   # Warm-up episodes
parser.add_argument('--update_freq', type=int, default=5)        # Update frequency
parser.add_argument('--eval_freq', type=int, default=10)         # Evaluation frequency

# Diversion
parser.add_argument('--enable_cross_site', type=bool, default=True)
parser.add_argument('--redirect_radius', type=int, default=5)
parser.add_argument('--redirect_penalty', type=float, default=0.2)
parser.add_argument('--max_queue', type=int, default=5)  # Increase queue length
# Add new parameters:
parser.add_argument('--smoothness_weight', type=float, default=50.0)  # Smoothness weight
parser.add_argument('--future_weight', type=float, default=30.0)     # Future prediction weight
parser.add_argument('--queue_penalty', type=float, default=100.0)     # Queue backlog penalty

args = parser.parse_args()

# Automatically select data file
if args.train_mode == 'weekday':
    pkl_path = args.pkl_weekday
elif args.train_mode == 'weekend':
    pkl_path = args.pkl_weekend
else:
    pkl_path = args.pkl

# Automatically identify weekday/weekend day indices
weekday_indices, weekend_indices = [], []
try:
    with open(args.pkl_weekday, 'rb') as f:
        weekday_data = pickle.load(f)
    weekday_events = weekday_data['events']
except Exception:
    weekday_events = []
try:
    with open(args.pkl_weekend, 'rb') as f:
        weekend_data = pickle.load(f)
    weekend_events = weekend_data['events']
except Exception:
    weekend_events = []

# Load main data and get: all_events_by_day
with open(pkl_path, 'rb') as f:
    main_data = pickle.load(f)
all_events_by_day = main_data['events']

# Match indices
def find_indices(subset, superset):
    indices = []
    for day in subset:
        for i, ref_day in enumerate(superset):
            if np.array_equal(np.array(day), np.array(ref_day)):
                indices.append(i)
                break
    return indices

if weekday_events:
    weekday_indices = find_indices(weekday_events, all_events_by_day)
if weekend_events:
    weekend_indices = find_indices(weekend_events, all_events_by_day)

env_args = EnvArgs()
env = CsvChargingEnv(args=env_args, pkl_path=pkl_path, max_queue=args.max_queue,
                     weekday_indices=weekday_indices, weekend_indices=weekend_indices)
dummy = env.reset(env.all_events_by_day[0])
obs_dim = dummy.shape[1]
print(f"[DEBUG] Observation dimension: {obs_dim}")
print(f"[DEBUG] Future dimension (d_future): {env.d_future}")

# Agent initialization
mini = Mini()
agent = MADDPG(obs_dim, n_agents=env.N, device=mini.device, env=env, args=mini, k=5)
agent.memory = DelayedReplayBuffer(capacity=mini.buffer_size, delay_steps=env.d_future)

# Load data - support for weekday/weekend separation
if args.train_mode == 'weekday':
    with open(args.pkl_weekday, 'rb') as f:
        day_data = pickle.load(f)
    events_by_day = day_data['events']
    data_mode = 'weekday'
elif args.train_mode == 'weekend':
    with open(args.pkl_weekend, 'rb') as f:
        day_data = pickle.load(f)
    events_by_day = day_data['events']
    data_mode = 'weekend'
else:
    with open(args.pkl, 'rb') as f:
        day_data = pickle.load(f)
    events_by_day = day_data['events']
    data_mode = 'mixed'
    # Load weekday/weekend separated data
    weekday_events = None
    weekend_events = None
    try:
        with open(args.pkl_weekday, 'rb') as f:
            weekday_data = pickle.load(f)
        weekday_events = weekday_data['events']
        print(f"Loaded weekday data: {len(weekday_events)} days")
    except FileNotFoundError:
        print(f"Weekday data file not found: {args.pkl_weekday}")
    try:
        with open(args.pkl_weekend, 'rb') as f:
            weekend_data = pickle.load(f)
        weekend_events = weekend_data['events']
        print(f"Loaded weekend data: {len(weekend_events)} days")
    except FileNotFoundError:
        print(f"Weekend data file not found: {args.pkl_weekend}")
    # Fix: If failed to load separated data, force mixed_training to False, otherwise True
    if weekday_events is None or weekend_events is None:
        print("Weekday/Weekend data loading failed, falling back to original data")
        args.mixed_training = False
    else:
        args.mixed_training = True

N_DAYS = len(events_by_day)
print(f"Training data statistics:")
print(f"  - Total: {N_DAYS} days")
if data_mode == 'mixed' and args.mixed_training:
    print(f"  - Weekday: {len(weekday_events) if weekday_events else 0} days")
    print(f"  - Weekend: {len(weekend_events) if weekend_events else 0} days")
    print(f"  - Using mixed training strategy: {args.mixed_training}")

# Logging setup (auto-increment or add timestamp to avoid overwriting)
log_dir = args.logdir
os.makedirs(log_dir, exist_ok=True)
# Automatically generate unique log file name

base_log_name = 'train_maddpg'
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = os.path.join(log_dir, f'{base_log_name}_{timestamp}.log')
idx = 2
while os.path.exists(log_file):
    log_file = os.path.join(log_dir, f'{base_log_name}_{timestamp}_{idx}.log')
    idx += 1
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Training variables
all_rewards, all_actor_loss, all_critic_loss = [], [], []
all_peak_violations = []  # Record peak violation counts
best_reward = -np.inf
best_model_path = None
best_metrics = None
best_losses = None
ep_metrics_last = None

# Training loop
for ep in trange(args.episodes):
    # Get current stage parameters
    curr_params = get_curriculum_params(ep, args.episodes)

    # Dynamically adjust environment parameters
    env.beta_peak = args.beta_peak * curr_params['peak_penalty_scale']
    #env.enable_cross_site = curr_params['redirect_enabled']
    #env.d_future = curr_params['future_horizon']

    # Adjust agent training parameters
    if ep % 100 == 0:
        print(f"\n[Curriculum] Episode {ep}: peak_scale={curr_params['peak_penalty_scale']:.1f}, "
            f"smoothness={curr_params['smoothness_weight']:.1f}, "
            f"redirect={curr_params['redirect_enabled']}, "
            f"horizon={curr_params['future_horizon']}")
    # Distinguish between three training modes
    if args.train_mode == 'weekend':
        day_events = events_by_day[ep % len(events_by_day)]
        data_source = 'weekend'
        if ep == 0:
            env.validate_reward_scale()
        s_t = env.reset(day_events, data_source=data_source)
    elif args.train_mode == 'weekday':
        day_events = events_by_day[ep % len(events_by_day)]
        data_source = 'weekday'
        if ep == 0:
            env.validate_reward_scale()
        s_t = env.reset(day_events, data_source=data_source)
    else:
        # Allow mixed sampling in mixed mode
        if args.mixed_training and weekday_events is not None and weekend_events is not None:
            day_events, data_source = select_training_data(events_by_day, ep, strategy='mixed', 
                                            weekday_data=weekday_events, weekend_data=weekend_events)
            day_type = "Weekday" if data_source == 'weekday' else "Weekend"
            if ep % 50 == 0:
                print(f"  Episode {ep}: Using {day_type} data for training")
            if ep == 0:
                env.validate_reward_scale()
            s_t = env.reset(day_events, data_source=data_source)
        else:
            day_events, data_source = select_training_data(events_by_day, ep, strategy=data_mode)
            day_type = data_source
            if ep == 0:
                env.validate_reward_scale()
            s_t = env.reset(day_events, data_source=data_source)
    ep_r, ep_loss_a, ep_loss_c = 0, [], []
    peak_violations = 0
    episode_infos = []
    peak_hours_count = 0

    # Cross-site statistics
    total_redirects = 0
    successful_redirects = 0
    # Initialize redirect_monitor (if cross-site is enabled)
    if env.enable_cross_site:
        redirect_monitor = RedirectMonitor(env.N)

    # Dynamically adjust exploration noise
    if ep < args.warmup_episodes:
        agent.noise_sd = 0.1  # High exploration during warm-up
    else:
        agent.noise_sd = max(0.1, 0.2 * (1 - ep / args.episodes))  # Gradually decrease

    # Learning rate decay
    if ep % 500 == 0 and ep > 0:
        for param_group in agent.opt_a.param_groups:
            param_group['lr'] *= 0.9
        for param_group in agent.opt_c.param_groups:
            param_group['lr'] *= 0.9

    # 2. Modify action processing in training loop
    for t in range(env.T):
        obs_np = env.get_obs_all()
        obs = torch.from_numpy(obs_np).float().to(mini.device)

        # Record pre-split state
        pre_loads = np.array([env._calculate_site_load(s) for s in range(env.N)])
        pre_queues = np.array([len(q) for q in env.queue])

        # Get action
        p = agent.add_noise(agent.act(obs))

        # Data validation
        if torch.isnan(p).any() or torch.isinf(p).any():
            print(f"Warning: Invalid actions at ep {ep}, step {t}")
            p = torch.where(torch.isnan(p) | torch.isinf(p), torch.full_like(p, 0.5), p)

        # Training mode
        agent.actor.train()
        agent.critic.train()

        # Process action dimensions
        if isinstance(p, torch.Tensor):
            if env.enable_cross_site and p.dim() == 2 and p.shape[1] > 1:
                # CrossSiteActor outputs two values: [acceptance probability, redirection willingness]
                # The environment only needs the acceptance probability
                p_env = p[:, 0].cpu().numpy()  # Take the first column and convert to numpy
            else:
                # Standard Actor or already 1D
                p_env = p.cpu().numpy()
                if p_env.ndim == 2 and p_env.shape[1] == 1:
                    p_env = p_env.squeeze(-1)
        else:
            p_env = p
            if p_env.ndim == 2:
                if p_env.shape[1] > 1 and env.enable_cross_site:
                    p_env = p_env[:, 0]
                elif p_env.shape[1] == 1:
                    p_env = p_env.squeeze(-1)


        if p_env.ndim > 1:
            p_env = p_env.flatten()

        assert len(p_env) == env.N, f"p_env length {len(p_env)} != env.N {env.N}"

        # Forced peak control
        hour_of_day = (t // 4) % 24
        is_peak_hour = hour_of_day in [17, 18, 19, 20]
        is_pre_peak = hour_of_day in [15, 16]
        is_post_peak = hour_of_day in [21, 22]  # New: post-peak period

        # Calculate current total queue length
        total_queue = sum(len(q) for q in env.queue)
        queue_pressure = total_queue / (env.N * env.max_queue)

        # Dynamic control based on system load 
        current_loads = []
        for sid in range(env.N):
            load = env._calculate_site_load(sid)
            current_loads.append(load)
        
        avg_system_load = np.mean(current_loads)

        if is_peak_hour:
            # Zero tolerance policy during peak hours 
            for sid in range(env.N):
                current_load = env._calculate_site_load(sid)
                if current_load > env.peak_thresh:
                    p_env[sid] = 0.0  # Strictly prohibited
                elif current_load > 0.65:
                    p_env[sid] = min(p_env[sid], 0.01)
                elif current_load > 0.6:
                    p_env[sid] = min(p_env[sid], 0.05)
                elif current_load > 0.5:
                    p_env[sid] = min(p_env[sid], 0.1)
                else:
                    p_env[sid] = min(p_env[sid], 0.3)

            # System-level restriction
            if avg_system_load > 0.6:
                p_env = p_env * 0.5  # Further reduce by 50%

        elif is_pre_peak:
            # ★ Strict control during pre-peak period ★
            for sid in range(env.N):
                current_load = env._calculate_site_load(sid)
                
                if current_load > 0.7:
                    force_prob = 0.02  # Preventive extremely low probability
                elif current_load > 0.6:
                    force_prob = 0.1
                elif current_load > 0.5:
                    force_prob = 0.2
                else:
                    force_prob = 0.5
                
                p_env[sid] = min(p_env[sid], force_prob)
        
        elif is_post_peak:
            # Gradual release strategy after peak 
            for sid in range(env.N):
                current_load = env._calculate_site_load(sid)
                queue_len = len(env.queue[sid])

                # Release rate based on load and queue length
                if current_load > 0.7:
                    release_rate = 0.1  # Slow release under high load
                elif current_load > 0.5:
                    release_rate = 0.3
                else:
                    release_rate = 0.5

                # Consider queue pressure
                if queue_len > 5:
                    release_rate = min(release_rate * 1.2, 0.6)
                
                p_env[sid] = min(p_env[sid], release_rate)

        # Global queue pressure control
        if queue_pressure > 0.6:  # Excessive queue pressure
            # Limit total acceptance
            total_accept_prob = np.sum(p_env)
            if total_accept_prob > env.N * 0.3:
                # Scale down proportionally
                scale_factor = (env.N * 0.3) / total_accept_prob
                p_env = p_env * scale_factor

        # Early training adopts more aggressive strategies
        if ep < 100:
            if is_peak_hour:
                p_env = p_env * 0.1  # 10% probability
            elif is_pre_peak:
                # Strict control during pre-peak period
                p_env = p_env * 0.3  # 30% probability

        # Mid-term training gradually relaxes
        elif ep < 200:
            if is_peak_hour:
                p_env = p_env * 0.2  # 20% probability

        # Calculate binary actions (for storage)
        if env.enable_cross_site and p.dim() == 2 and p.shape[1] > 1:
            a_bin = torch.bernoulli(p)
        else:
            # Create binary version for standard actions
            p_env_tensor = torch.from_numpy(p_env).float()
            a_bin = torch.bernoulli(p_env_tensor)
            if a_bin.dim() == 1:
                a_bin = a_bin.unsqueeze(-1)

        # Conduct smart redirection
        if hasattr(env, 'execute_smart_redirect'):
            redirect_decisions = env.execute_smart_redirect(p_env)
            if redirect_decisions:
                print(f"Executed {len(redirect_decisions)} redirects at step {t}")

        # Execute environment step
        s2, r, done, info = env.step(p_env)

        if 'redirect_decisions' in locals() and redirect_decisions:
            info['redirect_decisions'] = redirect_decisions

        # Reward Checking and Debugging
        if torch.any(torch.abs(r) > 200):
            peak_violations = info.get('violations', 0)
            if peak_violations > 0 and is_peak_hour:
                print(f"EP {ep:03d} T={t:02d}: Peak violations={peak_violations}, Rewards={r.cpu().numpy()}")
        
        if t % 20 == 0:  
            env.print_reward_analysis(info)

        if torch.any(torch.abs(r) > 100):
            print(f"WARNING: Large reward detected at ep {ep}, step {t}")
            print(f"Reward: {r.cpu().numpy()}")
            print(f"Info: {info.get('debug_info', {})}")
        
        post_loads = np.array([env._calculate_site_load(s) for s in range(env.N)])
        post_queues = np.array([len(q) for q in env.queue])

        # if cross-site redirection is enabled, record redirects
        if env.enable_cross_site and 'redirect_decisions' in info:
            for from_site, to_site, t in info['redirect_decisions']:
                redirect_monitor.record_redirect(
                    from_site, to_site,
                    pre_loads, post_loads,
                    pre_queues, post_queues
                )

        # Record statistics
        if 'violations' in info:
            peak_violations += info['violations']
        if 'peak_hour' in info and info['peak_hour']:
            peak_hours_count += 1
        if 'redirects' in info:
            total_redirects = info['redirects']
            
        episode_infos.append(info) 
        ep_metrics_last = info["metrics"]

        # Reward Checking and Debugging
        if torch.any(torch.isnan(r)) or torch.any(torch.isinf(r)):
            print(f"Warning: Invalid rewards at ep {ep}, step {t}")
            r = torch.where(torch.isnan(r) | torch.isinf(r), 
                          torch.zeros_like(r), r)
        
        r = torch.clamp(r, -500.0, 200.0)

        # Store experiences (note that action dimensions may vary)
        if info.get("arrivals", 0) > 0 or t % 4 == 0:
            # Calculate binary actions (for storage)
            if env.enable_cross_site and p.shape[1] > 1:
                # Store full binary actions
                a_bin = torch.bernoulli(p)
            else:
                a_bin = torch.bernoulli(p)

            future_free = np.asarray(info["future_free"])
            future_dom = np.asarray(info["future_dom"])
            if future_free.ndim == 1:
                future_free = future_free[:, None]  # (N,) -> (N,1)
            if future_dom.ndim == 1:
                future_dom = future_dom[:, None]
            #print("future_free shape:", future_free.shape, "future_dom shape:", future_dom.shape)
            agent.memory.push(
                torch.tensor(obs_np),
                p.detach().cpu(),
                r.cpu(),
                s2.cpu(),
                torch.tensor([done], dtype=torch.float32),
                torch.tensor(future_free),
                torch.tensor(future_dom),
                a_bin.detach().cpu()
            )

        # Update agent
        if len(agent.memory) >= mini.batch_size and t % 2 == 0:
            la, lc = agent.update()

            # Check training stability
            if not (np.isnan(la) or np.isnan(lc)):
                ep_loss_a.append(la)
                ep_loss_c.append(lc)
            else:
                print(f"Warning: Invalid losses at ep {ep}, step {t}: la={la}, lc={lc}")
        
        ep_r += r.sum().item()
        if done:
            break

    # Record episode results
    mean_actor_loss = np.mean(ep_loss_a) if ep_loss_a else 0.
    mean_critic_loss = np.mean(ep_loss_c) if ep_loss_c else 0.
    all_rewards.append(ep_r)
    all_actor_loss.append(mean_actor_loss)
    all_critic_loss.append(mean_critic_loss)
    all_peak_violations.append(peak_violations)

    # After episode ends, get redirection statistics (if enabled)
    if env.enable_cross_site:
        redirect_summary = redirect_monitor.get_summary()
    else:
        redirect_summary = {'total_redirects': 0}
    
    if ep % 20 == 0:
        violation_rate = None
        if ep_metrics_last is not None:
            m = ep_metrics_last
            if m["steps"] > 0:
                avg_wait = (m["wait_slots"] * 15) / max(m["accepts"], 1)
                avg_q = m["queue_len_sum"] / (m["steps"] * env.N)
                acc_rate = m["accepts"] / max(m["arrivals"], 1)
                avg_fee = m["fee_sum"] / max(m["accepts"], 1)

                # Calculate peak hour violations
                peak_hour_violations = 0
                peak_hour_steps = 0
                
                for info in episode_infos:
                    if info.get('peak_hour', False):
                        peak_hour_steps += 1
                        if 'peak_hour_violations' in info:
                            peak_hour_violations += info['peak_hour_violations']
                
                if peak_hour_steps > 0:
                    violation_rate = peak_hour_violations / (peak_hour_steps * env.N)
                else:
                    violation_rate = 0.0
                
                log_msg = (f'EP {ep:03d} | reward={ep_r:.0f} | wait={avg_wait:.1f}m | Q={avg_q:.2f} | '
                        f'acc={acc_rate:.2f} | fee={avg_fee:.2f} | actor_loss={mean_actor_loss:.4f} | '
                        f'critic_loss={mean_critic_loss:.4f} | peak_hour_violation_rate={violation_rate:.2%}')
                print(log_msg)
                logging.info(log_msg)

        # Peak Control Analysis
        detail2 = (f'EP {ep:03d} Details:\n'
                f'  Peak violations: {peak_violations}\n'
                f'  Peak hour steps: {peak_hours_count}/96 ({peak_hours_count/96*100:.1f}%)')
        print(detail2)
        logging.info(detail2)

        # Analyze peak hour violation rate
        peak_hour_violations = sum(1 for info in episode_infos
                                 if info.get('peak_hour', False) and info.get('violations', 0) > 0)
        if peak_hours_count > 0:
            violation_rate = peak_hour_violations / peak_hours_count
            vmsg = f'  Peak hour violation rate: {violation_rate:.2%}'
            print(vmsg)
            logging.info(vmsg)
  
        if hasattr(env, 'reward_breakdown'):
            print(f"  Reward breakdown: {env.reward_breakdown}")
    
    if ep % 50 == 0 and ep > 0 and env.enable_cross_site:
        print("\n" + "="*60)
        print("Performing Redirect Analysis...")
        print("="*60)

        # Temporary evaluation
        with torch.no_grad():
            test_events = random.choice(events_by_day)
            env.reset(test_events)
            
            redirect_summary = {
                'total_redirects': 0,
                'effective_redirects': 0,
                'load_improvements': [],
                'queue_improvements': []
            }
            
            for t in range(min(20, env.T)):  
                obs = torch.from_numpy(env.get_obs_all()).float().to(mini.device)
                a = agent.act(obs)
                
                pre_loads = [env._calculate_site_load(s) for s in range(env.N)]
                pre_queue_std = np.std([len(q) for q in env.queue])

                if env.enable_cross_site and a.shape[1] > 1:
                    a_env = a[:, 0]
                else:
                    a_env = a.squeeze(-1) if a.ndim == 2 and a.shape[1] == 1 else a
                    
                _, _, done, info = env.step(a_env)
                
                post_loads = [env._calculate_site_load(s) for s in range(env.N)]
                post_queue_std = np.std([len(q) for q in env.queue])
                
                if 'redirect_decisions' in info and info['redirect_decisions']:
                    redirect_summary['total_redirects'] += len(info['redirect_decisions'])
                    
                    load_std_improvement = np.std(pre_loads) - np.std(post_loads)
                    queue_std_improvement = pre_queue_std - post_queue_std
                    
                    if load_std_improvement > 0 or queue_std_improvement > 0:
                        redirect_summary['effective_redirects'] += 1
                        redirect_summary['load_improvements'].append(load_std_improvement)
                        redirect_summary['queue_improvements'].append(queue_std_improvement)
                
                if done:
                    break
    

        if redirect_summary['total_redirects'] > 0:
            effectiveness_rate = redirect_summary['effective_redirects'] / redirect_summary['total_redirects']
            avg_load_imp = np.mean(redirect_summary['load_improvements']) if redirect_summary['load_improvements'] else 0
            avg_queue_imp = np.mean(redirect_summary['queue_improvements']) if redirect_summary['queue_improvements'] else 0
            
            print(f"Redirect Summary (Episode {ep}):")
            print(f"  Total Redirects: {redirect_summary['total_redirects']}")
            print(f"  Effective Redirects: {redirect_summary['effective_redirects']}")
            print(f"  Effectiveness Rate: {effectiveness_rate:.2%}")
            print(f"  Avg Load Std Improvement: {avg_load_imp:.4f}")
            print(f"  Avg Queue Std Improvement: {avg_queue_imp:.4f}")
        else:
            print(f"No redirects performed in test (Episode {ep})")
        
        print("="*60 + "\n")

    if ep % 50 == 0:
        diagnostics = env.get_training_diagnostics()
        if diagnostics:
            print(f"\n=== TRAINING DIAGNOSTICS EP {ep} ===")
            print(f"Avg violations per step: {diagnostics['avg_violations_per_step']:.2f}")
            print(f"Avg reward per step: {diagnostics['avg_reward_per_step']:.2f}")
            print(f"Max peak load: {diagnostics['max_peak_load']:.3f}")
            print(f"Violation rate: {diagnostics['violation_rate']:.2%}")
            print("="*40)

    if ep_metrics_last:
        m = ep_metrics_last

        if m["steps"] > 0:  # ensure valid data
            avg_wait = (m["wait_slots"] * 15) / max(m["accepts"], 1)
            avg_q = m["queue_len_sum"] / (m["steps"] * env.N)
            acc_rate = m["accepts"] / max(m["arrivals"], 1)
            avg_fee = m["fee_sum"] / max(m["accepts"], 1)
            if ep % 20 == 0:
                detail_msg = (f'EP {ep:03d}  reward {ep_r:.0f}  wait={avg_wait:.1f}m  '
                              f'Q={avg_q:.2f}  acc={acc_rate:.2f}  fee={avg_fee:.2f}')
                print(detail_msg)
                logging.info(detail_msg)
           
                detail2 = (f'EP {ep:03d} Details:\n'
                           f'  Peak violations: {peak_violations}\n'
                           f'  Peak hour steps: {peak_hours_count}/96 ({peak_hours_count/96*100:.1f}%)')
                print(detail2)
                logging.info(detail2)
                # Analyze peak hour violation rate
                peak_hour_violations = sum(1 for info in episode_infos
                                         if info.get('peak_hour', False) and info.get('violations', 0) > 0)
                if peak_hours_count > 0:
                    violation_rate = peak_hour_violations / peak_hours_count
                    vmsg = f'  Peak hour violation rate: {violation_rate:.2%}'
                    print(vmsg)
                    logging.info(vmsg)
        else:
            warn_msg = f'EP {ep:03d}  reward {ep_r:.0f}  [No valid metrics - steps=0]'
            print(warn_msg)
            logging.warning(warn_msg)

    # save the best model
    if ep_r > best_reward and ep > args.warmup_episodes:
        best_reward = ep_r
        best_model_path = os.path.join(args.save_dir, f'best_maddpg_{data_mode}_ep{ep}.pth')
        best_metrics = {
            'episode': ep,
            'reward': ep_r,
            'avg_wait': avg_wait if m["steps"] > 0 else None,
            'avg_q': avg_q if m["steps"] > 0 else None,
            'acc_rate': acc_rate if m["steps"] > 0 else None,
            'avg_fee': avg_fee if m["steps"] > 0 else None,
            'peak_violations': peak_violations,
            'peak_hours': peak_hours_count,
            'peak_hour_violation_rate': violation_rate if m["steps"] > 0 and peak_hours_count > 0 else None
        }
        best_losses = {
            'actor_loss': float(np.mean(ep_loss_a) if ep_loss_a else 0.),
            'critic_loss': float(np.mean(ep_loss_c) if ep_loss_c else 0.)
        }
        torch.save({
            'episode': ep,
            'actor': agent.actor.state_dict(),
            'critic': agent.critic.state_dict(),
            'critic_t': agent.critic_t.state_dict(),
            'encoder': agent.encoder.state_dict(),
            'actor_opt': agent.opt_a.state_dict(),
            'critic_opt': agent.opt_c.state_dict(),
            'reward': ep_r,
            'hyperparams': vars(args),
            'metrics': best_metrics,
            'losses': best_losses
        }, best_model_path)
        
        logging.info(f'Best model updated at EP {ep:03d}, reward={ep_r:.2f}, saved to {best_model_path}')
        logging.info(f'Best metrics: {best_metrics}')
        logging.info(f'Best losses: {best_losses}')

    # Save checkpoint
    if ep % 100 == 0 and ep > 0 and env.enable_cross_site:
        checkpoint_path = os.path.join(args.save_dir, f'checkpoint_ep{ep}.pth')
        torch.save({
            'episode': ep,
            'actor': agent.actor.state_dict(),
            'critic': agent.critic.state_dict(),
            'critic_t': agent.critic_t.state_dict(),
            'encoder': agent.encoder.state_dict(),
            'rewards': all_rewards,
            'losses': {'actor': all_actor_loss, 'critic': all_critic_loss},
            'violations': all_peak_violations
        }, checkpoint_path)
        
        print("\n" + "="*60)
        print(f"PEAK SHAVING VALIDATION (Episode {ep})")
        print("="*60)

        # Evaluate peak shaving effect
        agent.actor.eval()
        with torch.no_grad():
            test_events = random.choice(events_by_day)
            env.reset(test_events)
            
            peak_hour_loads = []
            off_peak_loads = []
            peak_violations = 0
            peak_accepts = 0
            peak_arrivals = 0
            
            for t in range(env.T):
                obs = torch.from_numpy(env.get_obs_all()).float().to(mini.device)
                a = agent.act(obs)
                
                if env.enable_cross_site and a.shape[1] > 1:
                    a_env = a[:, 0]
                else:
                    a_env = a.squeeze(-1) if a.ndim == 2 and a.shape[1] == 1 else a
                
                _, _, done, info = env.step(a_env)

                # collect data
                hour = (t // 4) % 24
                current_loads = info.get('current_loads', [])
                
                if hour in [17, 18, 19, 20]:
                    peak_hour_loads.extend(current_loads)
                    peak_violations += sum(1 for load in current_loads if load > env.peak_thresh)
                    peak_accepts += sum(info['accept'])
                    peak_arrivals += info['arrivals']
                else:
                    off_peak_loads.extend(current_loads)
                
                if done:
                    break

        # Analyze results
        if peak_hour_loads and off_peak_loads:
            avg_peak = np.mean(peak_hour_loads)
            avg_off_peak = np.mean(off_peak_loads)
            peak_violation_rate = sum(1 for l in peak_hour_loads if l > env.peak_thresh) / len(peak_hour_loads)
            peak_accept_rate = peak_accepts / max(peak_arrivals, 1)
            
            print(f"Peak hour average load: {avg_peak:.3f}")
            print(f"Off-peak average load: {avg_off_peak:.3f}")
            print(f"Load shifting effect: {avg_off_peak - avg_peak:.3f}")
            print(f"Peak violation rate: {peak_violation_rate:.2%}")
            print(f"Peak hour accept rate: {peak_accept_rate:.2%}")
            print(f"Peak violations: {peak_violations}")

            # Define success criteria
            success_criteria = (
                avg_peak < avg_off_peak and  # Peak load < Off-peak load
                peak_violation_rate < 0.05 and  # Violation rate < 5%
                peak_accept_rate < 0.5  # Peak accept rate < 50%
            )
            
            if success_criteria:
                print("✓ Excellent peak shaving performance!")
            elif avg_peak < avg_off_peak:
                print("○ Good load shifting, but needs improvement")
            else:
                print("✗ Peak shaving failed - agent still accepts too much in peak hours")
        
        print("="*60 + "\n")
        
        agent.actor.train()

print(f"\nTraining completed! Best reward: {best_reward:.2f}")
print(f"Best model saved at: {best_model_path}")
if best_metrics:
    print(f"Best metrics: {best_metrics}")
    print(f"Best losses: {best_losses}")
    logging.info(f"Training completed! Best reward: {best_reward:.2f}")
    logging.info(f"Best model saved at: {best_model_path}")
    logging.info(f"Best metrics: {best_metrics}")
    logging.info(f"Best losses: {best_losses}")

# Visualize results

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
# Reward curve
axes[0].plot(all_rewards, alpha=0.7, label='Agent')
axes[0].plot(np.convolve(all_rewards, np.ones(20)/20, mode='valid'), 'r-', linewidth=2, label='Agent Smoothed')
axes[0].set_title('Training Rewards')
axes[0].set_xlabel('Episode')
axes[0].set_ylabel('Reward')
axes[0].legend()

# Loss curve
axes[1].plot(all_actor_loss, label='Actor Loss', alpha=0.7)
axes[1].plot(all_critic_loss, label='Critic Loss', alpha=0.7)
axes[1].set_title('Training Losses')
axes[1].set_xlabel('Episode')
axes[1].set_ylabel('Loss')
axes[1].legend()

# Learning progress (smoothed rewards)
window = 50
if len(all_rewards) > window:
    smoothed_rewards = np.convolve(all_rewards, np.ones(window)/window, mode='valid')
    axes[2].plot(smoothed_rewards)
    axes[2].set_title(f'Smoothed Rewards (window={window})')
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Smoothed Reward')

plt.tight_layout()
plt.savefig('training_results.png', dpi=300)
plt.show()

results = {
    "rewards": all_rewards,
    "actor_loss": all_actor_loss,
    "critic_loss": all_critic_loss,
    "peak_violations": all_peak_violations,
    "best_reward": best_reward,
    "best_model_path": best_model_path,
    "hyperparams": vars(args),
    "best_metrics": best_metrics,
    "best_losses": best_losses
}

with open("saved_results/improved_maddpg_results.pkl", "wb") as f:
    pickle.dump(results, f)

logging.info("Training results saved to saved_results/improved_maddpg_results.pkl")
print("Training results saved to saved_results/improved_maddpg_results.pkl")

# -----------------evaluation-----------------
# evaluate the best model
if best_model_path and os.path.exists(best_model_path):
    print("\nEvaluating best model...")
    checkpoint = torch.load(best_model_path, map_location=mini.device)
    agent.actor.load_state_dict(checkpoint['actor'])
    agent.critic.load_state_dict(checkpoint['critic'])
    agent.encoder.load_state_dict(checkpoint['encoder'])
    
    agent.actor.eval()
    agent.critic.eval()
    
    eval_rewards = []
    eval_violations = []
    
    for i in range(5):
        eval_events = random.choice(events_by_day)
        # identify day type automatically
        if args.mixed_training and weekday_events is not None and weekend_events is not None:
            if any(np.array_equal(np.array(eval_events), np.array(d)) for d in weekday_events):
                data_source = 'weekday'
            elif any(np.array_equal(np.array(eval_events), np.array(d)) for d in weekend_events):
                data_source = 'weekend'
            else:
                data_source = 'mixed'
            env.reset(eval_events, data_source=data_source)
        else:
            env.reset(eval_events)
        ep_r = 0
        violations = 0
        agent_total_loads = []
        baseline_total_loads = []
        with torch.no_grad():
            for step in range(env.T):
                obs = torch.from_numpy(env.get_obs_all()).float().to(mini.device)
                a = agent.act(obs)  # no noise
                # only pass acceptance probabilities to the environment
                if env.enable_cross_site and a.shape[1] > 1:
                    a_env = a[:, 0]
                else:
                    a_env = a.squeeze(-1) if a.ndim == 2 and a.shape[1] == 1 else a
                _, r, done, info = env.step(a_env)
                ep_r += r.sum().item()
                
                if 'violations' in info:
                    violations += info['violations']
                
                if 'current_loads' in info:
                    agent_total_loads.append(np.sum(info['current_loads']))
                    # Baseline: Assume full acceptance
                    baseline_load = 0
                    for sid in range(env.N):
                        # Calculate load under full acceptance
                        t_now = step % env.T
                        pub_load = env.cap[sid] * env.charger_kw
                        dom_load = env.dom_load[t_now][sid]
                        total_load = pub_load + dom_load
                        baseline_load += total_load / env.grid_cap[sid]
                    baseline_total_loads.append(baseline_load)
                if done:
                    break
        eval_rewards.append(ep_r)
        eval_violations.append(violations)

    # === Load Statistics ===
    if agent_total_loads and baseline_total_loads:
        print("\n=== Load Statistics ===")
        print(f"Baseline Total Load: Mean={np.mean(baseline_total_loads):.3f}, Max={np.max(baseline_total_loads):.3f}, Min={np.min(baseline_total_loads):.3f}")
        print(f"Agent Total Load:    Mean={np.mean(agent_total_loads):.3f}, Max={np.max(agent_total_loads):.3f}, Min={np.min(agent_total_loads):.3f}")
        print(f"Agent/Baseline Total Load Ratio: {np.mean(agent_total_loads)/np.mean(baseline_total_loads):.3f}")