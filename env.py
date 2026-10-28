# env.py
# Function: Defines the recommended environment for electric vehicle charging, including status, events, rewards, status transitions, etc.
import numpy as np
import torch
from utils.home import gen_home_curve
import time
import pickle

TorchFloat = None
TorchLong = None

class RunningStat(object):
    """
    Numerically‑stable online mean / variance (Welford, 1962)
    --------------------------------------------------------
    push(x)  Allows multiple sample lines to be pushed at once; shape = (batch, feat)
    """
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape, dtype=np.float64)   # mean
        self._S = np.zeros(shape, dtype=np.float64)   # Σ (x‑μ_old)(x‑μ_new)

    def push(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]            # (1,F)

        for row in x:               
            self._n += 1
            delta  = row - self._M
            self._M += delta / self._n
            self._S += delta * (row - self._M)

    # ---------- properties ----------
    @property
    def n(self):   return self._n

    @property
    def mean(self): return self._M

    @property
    def var(self):
        # When n<2, the variance is defined as 0 to avoid division by zero.
        return np.zeros_like(self._M) if self._n < 2 else self._S / (self._n - 1)

    @property
    def std(self):  
        return np.sqrt(np.maximum(self.var, 1e-12))

    # The above are implementations that return the mean, variance, and standard deviation, respectively. In MARL, they are often used for state or reward normalization to improve training stability.

class Identity:  # Identity transformation class, implements __call__ method, outputs whatever is input. Purpose: as a placeholder in the normalization process (if no normalization is needed).
    '''
    A convenience class which simply implements __call__
    as the identity function
    '''
    def __call__(self, x, *args, **kwargs):
        return x

    def reset(self):
        pass

class ZFilter:  # z-score normalization: (x-mean)/std
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """
    def __init__(self, shape, center=True, scale=True, clip=None):
        assert shape is not None
        self.center = center
        self.scale = scale
        self.clip = clip
        self.rs = RunningStat(shape)
        # self.prev_filter = prev_filter

    def __call__(self, x, **kwargs):  # Input data x, return normalized data
        # x = self.prev_filter(x, **kwargs)
        x = np.asarray(x)
        if x.ndim == 1:
            x = x[:, None]
        self.rs.push(x)
        # print(self.rs.mean,self.rs.std)
        if self.center:
            x = x - self.rs.mean
        if self.scale:
            if self.center:
                x = x / (self.rs.std + 1e-8)
            else:
                diff = x - self.rs.mean
                diff = diff/(self.rs.std + 1e-8)
                x = diff + self.rs.mean
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x

    def reset(self):   # Reset statistics
        # self.prev_filter.reset()
        self.rs.reset()

class RewardFilter:     # Reward normalization, mainly used for training reward scale normalization
    """
    1. update return
    2. divide reward by std(return) *without* subtracting and adding back mean
    """
    def __init__(self, shape, gamma, clip=None):  
        assert shape is not None  
        self.gamma = gamma  
        # self.prev_filter = prev_filter
        self.rs = RunningStat(shape)
        self.ret = np.zeros(shape)
        self.clip = clip

    def __call__(self, x, **kwargs):   # Input reward x, return normalized reward
        # x = self.prev_filter(x, **kwargs)
        x = np.asarray(x)
        if x.ndim == 1:
            x = x[:, None]
        self.ret = self.ret * self.gamma + x
        self.rs.push(self.ret)
        x = x / (self.rs.std + 1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x
    
    def reset(self):
        self.ret = np.zeros_like(self.ret)
        # self.prev_filter.reset()
        self.rs.reset()
        
# ====================================================================================
# Key CsvChargingEnv class for generating the charging environment based on events.pkl
# ====================================================================================
class CsvChargingEnv:
    def __init__(self, args, pkl_path="./data/events_augmented.pkl", max_queue=5, weekday_indices=None, weekend_indices=None):
        if args is None or not hasattr(args, "device"):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            if isinstance(args.device, str):
                self.device = torch.device(args.device)
            else:
                self.device = args.device
        global TorchFloat, TorchLong
        TorchFloat = torch.cuda.FloatTensor if self.device.type == 'cuda' else torch.FloatTensor
        TorchLong  = torch.cuda.LongTensor  if self.device.type == 'cuda' else torch.LongTensor

        # check data type (weekday/weekend/mixed)
        if 'weekday' in pkl_path:
            self._data_type = 'weekday'
        elif 'weekend' in pkl_path:
            self._data_type = 'weekend' 
        else:
            self._data_type = 'mixed'
        print(f"[ENV] Data type detected: {self._data_type}")

        # keep track of weekday/weekend indices
        self.weekday_indices = weekday_indices if weekday_indices is not None else []
        self.weekend_indices = weekend_indices if weekend_indices is not None else []

        blob = pickle.load(open(pkl_path, "rb"))
        self.all_events_by_day = blob["events"]  
        self.site_names = blob["site_names"]
        self.site_stats = blob["site_stats"]
        self.cap = np.array(self.site_stats["num_chargers"])  # (site number,)
        self.N = len(self.site_names)
        self.knn_idx = blob["knn_idx"]     # (site number, K)  # K=5
        # --------------------------------
        self.max_queue = max_queue
        self.site_slot_kwh_by_day = blob.get("site_slot_kwh_by_day", None)  

        # static features
        self.connector_types = np.array(self.site_stats["num_connector_types"], dtype=np.float32).reshape(self.N, 1)
        self.model_count = np.array(self.site_stats["model_count"], dtype=np.float32).reshape(self.N, 1)
        if "Total_Dwellings" in self.site_stats:
            self.total_dwellings = np.array(self.site_stats["Total_Dwellings"], dtype=np.float32).reshape(self.N, 1)
            # Check dwelling data
            print(f"Total dwellings range: {self.total_dwellings.min():.0f} - {self.total_dwellings.max():.0f}")
            print(f"Sample dwellings: {self.total_dwellings[:5, 0]}")
        else:
            # If Total_Dwellings field is missing, use estimated values based on the number of chargers, each serving 50 households
            estimated_dwellings = np.array(self.site_stats["num_chargers"], dtype=np.float32) * 50  
            self.total_dwellings = estimated_dwellings.reshape(self.N, 1)
            print(f"[WARNING] Total_Dwellings field is missing, using estimated values (number of chargers * 50)")
            print(f"Estimated dwellings range: {self.total_dwellings.min():.0f} - {self.total_dwellings.max():.0f}")

        # initialize the following fields
        self.avg_price_per_kwh = np.array(self.site_stats["avg_price_per_kwh"], dtype=np.float32).reshape(self.N, 1)
        self.avg_charge_cost = np.array(self.site_stats["avg_charge_cost"], dtype=np.float32).reshape(self.N, 1)
        self.latitude = np.array(self.site_stats["latitude"], dtype=np.float32).reshape(self.N, 1)
        self.longitude = np.array(self.site_stats["longitude"], dtype=np.float32).reshape(self.N, 1)

        self.state_filter = None
        #self.reward_filter = ZFilter(shape=[1], center=True, clip=None)
        self.reward_filter_time = Identity()                       # Peak shaving reward
        self.reward_filter_fee  = ZFilter(shape=[1], center=True, clip=10)  # Cost reward

        # Runtime variables
        self.t = 0
        self.free = None
        self.queue = None
        self.T = None
        self.arrivals = None
        self.leavings = None
        self.day_events = None  
        self.d_future = 16  
        self.enable_long_term_planning = True
        self.planning_horizon = 32  # 8h prediction
        # ============ Domestic-EV background load ============
        self.dom_load = None                   # shape (96, N)
        if args is None:
            args = type('args', (), {})()  # create empty object
        # ---------- Public charger power ----------
        # self.charger_kw = min(getattr(args, 'charger_kw', 35.0), 15.0)
        self.charger_kw   = getattr(args, "charger_kw", 22.0)   # kW / charger usually 22-50 kW
        self.pub_cap_kw   = self.cap * self.charger_kw         # (N,)

        self.dom_cap  = np.zeros_like(self.cap, dtype=np.float32)
        self.grid_cap = self.cap * self.charger_kw * 1.05     # 20% higher than total charger power
        self._grid_cap_init = False            # flag: only initialize once
        print('[GridCap init]  min(dom_cap)=', float(self.dom_cap.min()))

        self.beta_peak   = getattr(args, "beta_peak", 20.0)      # Peak penalty coefficient
        self.peak_thresh = getattr(args, "peak_thresh", 0.7)    # >70 % considered "over-peak"

        self.max_queue = max_queue
        # Cross-site diversion related parameters
        self.enable_cross_site = getattr(args, "enable_cross_site", True)
        self.redirect_radius = getattr(args, "redirect_radius", 5)  # Max redirect to 5 nearest sites
        self.redirect_penalty = getattr(args, "redirect_penalty", 0.2)  # Redirect distance penalty

        # Compute distance matrix between sites (based on latitude and longitude)
        self._compute_distance_matrix()
        # Record redirect statistics
        self.redirect_stats = {}

        # -------- Training period business indicators --------
        self._metrics_tpl = dict(
           steps=0, queue_len_sum=0, arrivals=0, accepts=0,
           wait_slots=0, fee_sum=0.0
        )
        self._m = self._metrics_tpl.copy()

        self.reward_breakdown = {
            'base_service': 0,
            'peak_penalty': 0, 
            'redirect_bonus': 0,
            'efficiency_bonus': 0
        }

    def _compute_distance_matrix(self):
        """Compute distance matrix between sites"""
        from scipy.spatial.distance import cdist
        coords = np.column_stack([self.latitude, self.longitude])
        # Use Euclidean distance approximation (Haversine should be used in practice)
        self.dist_matrix = cdist(coords, coords, metric='euclidean')
        # Normalize to [0,1]
        self.dist_matrix = self.dist_matrix / self.dist_matrix.max()
    # ---------- Core Interface ----------
    def reset(self, day_events=None, data_source=None):
        if day_events is None:
            day_events = self.all_events_by_day[0]
        self.day_events = day_events
        self.episode_peak_violations = 0
        self.redirect_debug_records = []
        
        self.t = 0
        self.T = 96
        self.free = self.cap.copy().astype(int)
        # Reset redirect statistics
        self.redirect_stats = {
            'total_redirects': 0,
            'successful_redirects': 0,
            'redirect_distances': []
        }
        # Slot-level charging量 for the day
        # Create enhanced event queue, record original site
        self.queue = [[] for _ in range(self.N)]  # Each element will be (original site id, waiting time)
        self.arrivals = [[] for _ in range(self.T+1)]
        self.leavings = [[] for _ in range(self.T+1)]
        
        for sid, ts, te, *_ in day_events:
            self.arrivals[int(ts)].append(int(sid))
            self.leavings[int(te)].append(int(sid))

        # Identify data source
        def _infer_data_source(day_events, weekday_indices, weekend_indices, all_events_by_day):
            day_idx = next((i for i, ref_day in enumerate(all_events_by_day) if np.array_equal(np.array(day_events), np.array(ref_day))), None)
            if day_idx is not None:
                if day_idx in weekday_indices:
                    return 'weekday', day_idx
                elif day_idx in weekend_indices:
                    return 'weekend', day_idx
                else:
                    return 'unknown', day_idx
            return 'unknown', None

        if data_source is not None:
            if data_source == 'weekday':
                is_weekday_pattern = True
                print(f"[DOMESTIC LOAD] Identify: Weekday home charging mode (data_source=weekday)")
            elif data_source == 'weekend':
                is_weekday_pattern = False
                print(f"[DOMESTIC LOAD] Identify: Weekend home charging mode (data_source=weekend)")
            else:
                inferred, day_idx = _infer_data_source(day_events, self.weekday_indices, self.weekend_indices, self.all_events_by_day)
                if inferred == 'weekday':
                    is_weekday_pattern = True
                    print(f"[DOMESTIC LOAD] Identify: Weekday home charging mode (day_idx={day_idx})")
                elif inferred == 'weekend':
                    is_weekday_pattern = False
                    print(f"[DOMESTIC LOAD] Identify: Weekend home charging mode (day_idx={day_idx})")
                else:
                    is_weekday_pattern = True
                    print(f"[DOMESTIC LOAD] Identify Failed: Defaulting to weekday home charging mode (day_idx={day_idx})")
        else:
            inferred, day_idx = _infer_data_source(day_events, self.weekday_indices, self.weekend_indices, self.all_events_by_day)
            if inferred == 'weekday':
                is_weekday_pattern = True
                print(f"[DOMESTIC LOAD] Identify: Weekday home charging mode (day_idx={day_idx})")
            elif inferred == 'weekend':
                is_weekday_pattern = False
                print(f"[DOMESTIC LOAD] Identify: Weekend home charging mode (day_idx={day_idx})")
            else:
                is_weekday_pattern = True
                print(f"[DOMESTIC LOAD] Identify Failed: Defaulting to weekday home charging mode (day_idx={day_idx})")

        if self.site_slot_kwh_by_day is not None:
            self.public_load = self.site_slot_kwh_by_day[day_idx].T 
        else:
            self.public_load = np.zeros((self.T, self.N), dtype=np.float32)  

        # generate domestic load curves
        curves = []
        seed_base = int(time.time()) % 10000
        
        for sid in range(self.N):
            # generate domestic load curve for each site
            curves.append(
                gen_home_curve(
                    n_house = int(max(self.total_dwellings[sid, 0], 1)),
                    weekday = is_weekday_pattern,  
                    seed = seed_base + sid
                )
            )
        raw_dom_load = np.stack(curves, axis=1).astype(np.float32)

        print(f"Raw domestic load range: {raw_dom_load.min():.2f} - {raw_dom_load.max():.2f} kW")
        print(f"Per-site domestic load sample: {raw_dom_load[0,:3]}")

        # Directly use the high-quality load curve generated by home.py and only perform simple capacity matching.
        self.dom_load = np.zeros((self.T, self.N), dtype=np.float32)
        
        for sid in range(self.N):
            site_raw = raw_dom_load[:, sid]
            n_chargers = self.cap[sid]
            
            # Adjust the domestic load scale according to the site capacity.
            public_capacity = n_chargers * self.charger_kw
            # Allow domestic load to occupy a reasonable proportion of public network capacity, leaving control space for intelligent entities.
            target_scale = public_capacity * np.random.uniform(0.3, 0.5)  # 30-50% grid capacity

            # simple linear scaling
            if site_raw.max() > 0:
                scale_factor = target_scale / site_raw.max()
                self.dom_load[:, sid] = site_raw * scale_factor
            else:
                self.dom_load[:, sid] = target_scale * 0.1
        

        if np.any(np.isnan(self.dom_load)) or np.any(np.isinf(self.dom_load)):
            print("ERROR: NaN or Inf in domestic load!")
            self.dom_load = self._generate_safe_domestic_load()
        
        if np.any(self.dom_load < 0):
            print("ERROR: Negative domestic load!")
            self.dom_load = np.maximum(self.dom_load, 0.1)
            
        if np.any(self.dom_load > 1000):
            print(f"WARNING: Very large domestic load: max={self.dom_load.max():.1f}kW")
            self.dom_load = np.clip(self.dom_load, 0, 500)
        
        print(f"Final domestic load range: {self.dom_load.min():.1f} - {self.dom_load.max():.1f} kW")
        print(f"Sample final loads: {self.dom_load[0,:3]}")

        # calculate grid capacity
        if not self._grid_cap_init:
            peak95 = np.percentile(self.dom_load, 95, axis=0)
            self.dom_cap[:] = peak95
            
            eps_cap = 1e-3
            self.dom_cap[self.dom_cap < eps_cap] = eps_cap
            
            safety = 0.1 * (self.pub_cap_kw + self.dom_cap)  # 0.3
            self.grid_cap[:] = self.pub_cap_kw + self.dom_cap + safety
            
            print('[GridCap] Domestic 95th percentile:', np.round(peak95[:5], 1))
            print('[GridCap] Public capacity:', np.round(self.pub_cap_kw[:5], 1))
            print('[GridCap] Grid capacity range:', 
                f'{self.grid_cap.min():.1f} - {self.grid_cap.max():.1f} kW')
            
            self._grid_cap_init = True

        print(f'[Reset] t=0  dom_load sample kW: {self.dom_load[0,:3]}  '
            f'grid_cap sample kW: {self.grid_cap[:3]}')

        # total load = dom_load + public_load
        self.total_load = self.dom_load + self.public_load  # shape: [96, N]

        state = self._get_state_tensor(init_filter=True)
        
        if torch.isnan(state).any():
            nan_cnt = torch.isnan(state).sum().item()
            print(f'[reset]  WARN: state contains {nan_cnt} NaNs')
        
        self._m = self._metrics_tpl.copy()
        print(f"DEBUG: Charger power: {self.charger_kw} kW")
        print(f"DEBUG: Grid capacity range: {self.grid_cap.min():.1f} - {self.grid_cap.max():.1f} kW")
        print(f"DEBUG: Load increase per vehicle: {self.charger_kw / self.grid_cap.min():.3f} - {self.charger_kw / self.grid_cap.max():.3f}")

        # if max_load_increase > 0.5, the configuration is unreasonable
        max_load_increase = self.charger_kw / self.grid_cap.min()
        if max_load_increase > 0.5:
            print(f"WARNING: Single vehicle increases load by {max_load_increase:.1%}, this is too much!")
            print("Consider reducing charger_kw or increasing grid_cap")
            
        return state

    def _generate_safe_domestic_load(self):
        """safer domestic load generation strategy"""
        dom_load = np.zeros((96, self.N), dtype=np.float32)
        
        for sid in range(self.N):
            # basic load: 30% of public charger capacity
            base_load = self.cap[sid] * self.charger_kw * 0.3

            # simple 24-hour pattern, repeated 4 times to get 96 15-minute slots
            hours = np.arange(24)

            # peak hours: 8-10 and 18-20
            daily_pattern = np.ones(24) * 0.3  # basic 30%
            daily_pattern[8:10] = 0.8   # morning peak 80%
            daily_pattern[18:20] = 1.0  # evening peak 100%

            # expand to 96 15-minute slots
            hourly_pattern = np.repeat(daily_pattern, 4)
            
            dom_load[:, sid] = base_load * hourly_pattern

            # add random noise (5%)
            noise = np.random.normal(1.0, 0.05, 96)
            dom_load[:, sid] *= noise

            dom_load[:, sid] = np.maximum(dom_load[:, sid], base_load * 0.1)
        
        return dom_load
    

    # key function for step
    def step(self, actions):
        """Improved step function - balancing peak control and waiting time"""
        self._m["steps"] += 1

        # 1. deal with arrivals, waiting times, and leavings
        arrivals_count = len(self.arrivals[self.t])
        for sid in self.arrivals[self.t]:
            if len(self.queue[sid]) < self.max_queue:
                self.queue[sid].append((sid, 0))
        self._m["arrivals"] += arrivals_count

        # update waiting times
        for sid in range(self.N):
            for i in range(len(self.queue[sid])):
                orig_site, wait_time = self.queue[sid][i]
                self.queue[sid][i] = (orig_site, wait_time + 1)

        # deal with leavings
        for sid in self.leavings[self.t]:
            self.free[sid] = min(self.free[sid] + 1, self.cap[sid])

        # 2. Analyzing actions
        if isinstance(actions, torch.Tensor):
            act_tensor = actions
        else:
            act_tensor = torch.tensor(actions)

        if act_tensor.dim() == 2:
            if act_tensor.shape[1] > 1 and self.enable_cross_site:
                accept_prob = act_tensor[:, 0].cpu().numpy().clip(0.0, 1.0)
                redirect_willingness = act_tensor[:, 1].cpu().numpy().clip(0.0, 1.0)
            else:
                accept_prob = act_tensor.squeeze(-1).cpu().numpy().clip(0.0, 1.0)
                redirect_willingness = np.zeros(self.N)
        else:
            accept_prob = act_tensor.cpu().numpy().clip(0.0, 1.0)
            redirect_willingness = np.zeros(self.N)
        
        if accept_prob.ndim > 1:
            accept_prob = accept_prob.flatten()
        assert len(accept_prob) == self.N, f"Accept prob length {len(accept_prob)} != N sites {self.N}"

        reward = np.zeros((self.N,), dtype=np.float32)
        accepts = [0] * self.N
        
        hour_of_day = (self.t // 4) % 24
        is_peak_hour = hour_of_day in [17, 18, 19, 20]
        is_post_peak = hour_of_day in [21, 22]
        total_violations = 0
        peak_hour_violations = 0

        # calculate current loads
        current_loads = []
        for sid in range(self.N):
            pub_load = self.public_load[self.t % self.T][sid] if self.public_load is not None else (self.cap[sid] - self.free[sid]) * self.charger_kw
            dom_load = self.dom_load[self.t % self.T][sid]
            total_load = pub_load + dom_load
            load_ratio = total_load / self.grid_cap[sid]
            current_loads.append(load_ratio)
            if load_ratio > self.peak_thresh:
                total_violations += 1
                if is_peak_hour:
                    peak_hour_violations += 1

        # ========== Global acceptance limits ==========
        # Prevent peak caused by backlog release
        max_accepts_per_step = max(1, int(self.N * 0.3))
        total_queue = sum(len(q) for q in self.queue)
        queue_pressure = total_queue / (self.N * self.max_queue)
        
        if queue_pressure > 0.6:
            max_accepts_per_step = max(1, int(self.N * 0.2))
        elif is_post_peak and queue_pressure > 0.4:
            max_accepts_per_step = max(1, int(self.N * 0.25))

        # calculate acceptance priorities
        accept_priorities = []
        for sid in range(self.N):
            if len(self.queue[sid]) > 0 and self.free[sid] > 0:
                orig_site, wait_time = self.queue[sid][0]
                
                priority_score = 0.0
                # Waiting time weighting (most important)
                if wait_time > 8:
                    priority_score += 20.0
                elif wait_time > 4:  
                    priority_score += 10.0
                elif wait_time > 2: 
                    priority_score += 5.0
                else:
                    priority_score += wait_time * 0.5

                # Load weighting (secondary)
                priority_score += (1.0 - current_loads[sid]) * 5.0

                # Peak hour adjustment
                if is_peak_hour and current_loads[sid] > 0.65:
                    priority_score -= 5.0
                    
                accept_priorities.append((priority_score, sid))

        # Sort by priority
        accept_priorities.sort(reverse=True)
        allowed_accepts = set([sid for _, sid in accept_priorities[:max_accepts_per_step]])

        # ========== Intelligent diversion decision ==========
        redirect_decisions = []
        just_redirected_in = set()
        
        if self.enable_cross_site:
            future_loads_1h = np.array([self._predict_future_load(s, 4) for s in range(self.N)])
            
            for sid in range(self.N):
                if sid in just_redirected_in or len(self.queue[sid]) == 0:
                    continue

                # get info of first vehicle
                _, front_wait_time = self.queue[sid][0]
                
                # Traffic diversion decisions based on waiting time and load
                need_redirect = False
                redirect_urgency = 0.0

                # Condition 1: Waiting time too long (highest priority)
                if front_wait_time > 6:  
                    need_redirect = True
                    redirect_urgency += (front_wait_time - 6) * 2.0
                elif front_wait_time > 3:
                    need_redirect = True
                    redirect_urgency += (front_wait_time - 4) * 1.0

                # Condition 2: Current load too high
                if current_loads[sid] > 0.7:
                    need_redirect = True
                    redirect_urgency += (current_loads[sid] - 0.7) * 10
                elif current_loads[sid] > 0.6 and front_wait_time > 4:
                    need_redirect = True
                    redirect_urgency += 5.0

                # Condition 3: Queue too long
                if len(self.queue[sid]) > 3:
                    need_redirect = True
                    redirect_urgency += (len(self.queue[sid]) - 3) * 2

                # Condition 4: Predicting future overload
                if future_loads_1h[sid] > 0.7:
                    need_redirect = True
                    redirect_urgency += (future_loads_1h[sid] - 0.7) * 5
                
                if not need_redirect:
                    continue

                # Find the best target
                best_target = None
                best_score = float('-inf')
                neighbors = self.knn_idx[sid][:self.redirect_radius]
                
                for nid in neighbors:
                    if nid >= self.N or nid == sid or nid in just_redirected_in:
                        continue
                    
                    target_load = current_loads[nid]
                    target_future = future_loads_1h[nid]
                    target_queue_len = len(self.queue[nid])

                    if target_load > 0.65:  # target load too high
                        continue
                    if target_future > 0.65:
                        continue
                    if target_queue_len >= self.max_queue - 1:
                        continue

                    # Consider distance-time trade-off
                    distance = self.dist_matrix[sid, nid]
                    travel_time_slots = int(distance * 4)

                    # If extra travel time is greater than saved waiting time, it's not worth it
                    expected_wait_at_target = target_queue_len * 2
                    time_saved = front_wait_time - (travel_time_slots + expected_wait_at_target)
                    
                    if time_saved < 0 and front_wait_time < 12:
                        continue  # not worth redirecting

                    # Comprehensive scoring
                    score = 0.0

                    # Capacity margin
                    capacity_margin = 0.65 - target_load
                    score += capacity_margin * 20

                    # Time saved
                    if time_saved > 0:
                        score += time_saved * 1.0

                    # Queue space
                    queue_space = (self.max_queue - target_queue_len) / self.max_queue
                    score += queue_space * 5

                    # Distance penalty (adjusted by waiting time)
                    distance_weight = max(0.5, 2.0 - front_wait_time / 8)
                    score -= distance * distance_weight

                    # Load difference
                    load_diff = current_loads[sid] - target_load
                    if load_diff > 0.2:
                        score += load_diff * 10
                    
                    if score > best_score:
                        best_score = score
                        best_target = nid
                
                # Dynamic threshold: The longer you wait, the easier it is to divert traffic.
                min_score_threshold = max(3.0, 8.0 - redirect_urgency * 0.2)
                
                if best_target is not None and best_score > min_score_threshold:
                    vehicle = self.queue[sid].pop(0)
                    self.queue[best_target].append(vehicle)
                    # record self.t when redirecting happens
                    redirect_decisions.append((sid, best_target, self.t))
                    just_redirected_in.add(best_target)
                    # Record redirection statistics
                    if not hasattr(self, 'redirect_wait_times'):
                        self.redirect_wait_times = []
                    self.redirect_wait_times.append(front_wait_time)
                    print(f"[DEBUG] Step {self.t}: Redirect from {sid} to {best_target}, "
                        f"wait_time={front_wait_time}, score={best_score:.2f}")

        # ========== Site-level rewards ==========
        accepts_this_step = 0
        
        if not hasattr(self, 'prev_loads'):
            self.prev_loads = [0.0] * self.N
        
        for sid in range(self.N):
            site_reward = 0.0
            queue_before = len(self.queue[sid])
            load_ratio = current_loads[sid]
            avg_wait = 0  

            # 1. Basic reward: Handling acceptance decisions
            if queue_before > 0 and self.free[sid] > 0:
                orig_site, wait_time = self.queue[sid][0]
                
                # ★ Dynamic safety threshold: Adjusting based on waiting time ★
                if wait_time > 16:  # Waited over 4 hours
                    safe_threshold = min(0.85, 0.7 + wait_time * 0.01)
                elif wait_time > 12:  # Waited over 3 hours
                    safe_threshold = 0.8
                elif is_peak_hour:
                    safe_threshold = 0.65
                else:
                    safe_threshold = 0.75

                # Check future safety
                if not self._can_accept_vehicle_future_safe(sid, lookahead=4, threshold=safe_threshold):
                    accepts[sid] = 0
                    # But if waited too long, apply penalty
                    if wait_time > 8:
                        site_reward -= min(wait_time - 8, 10) * 2.0
                    continue

                # Check if acceptance is allowed
                if sid not in allowed_accepts or accepts_this_step >= max_accepts_per_step:
                    if wait_time > 8:
                        site_reward -= min(wait_time - 8, 10) * 1.5
                    continue

                # Execute acceptance decision
                if np.random.rand() < accept_prob[sid]:
                    vehicle = self.queue[sid].pop(0)
                    self.free[sid] -= 1
                    accepts[sid] = 1
                    accepts_this_step += 1
                    self._m["accepts"] += 1
                    self._m["wait_slots"] += vehicle[1]
                    self._m["fee_sum"] += self.avg_charge_cost[sid, 0]

                    # ★ Improved acceptance reward: Considering waiting time ★
                    wait_penalty = min(wait_time * 0.5, 10.0)
                    
                    if load_ratio < 0.5:
                        site_reward += 8.0 - wait_penalty
                    elif load_ratio < 0.65:
                        site_reward += 5.0 - wait_penalty
                    elif load_ratio < 0.75:
                        site_reward += 2.0 - wait_penalty
                    else:
                        site_reward += 0.5 - wait_penalty

                    # Extra reward for redirecting vehicles
                    if orig_site != sid:
                        site_reward += 3.0
                else:
                    # Refusal case
                    if is_peak_hour and load_ratio > 0.6:
                        site_reward += 1.0
                    elif wait_time > 8:  # Waited over 2 hours
                        site_reward -= (wait_time - 8) * 0.5

            # 2. Peak control 
            if is_peak_hour and load_ratio > self.peak_thresh:
                # Second penalty
                site_reward -= 20.0 * ((load_ratio - self.peak_thresh) / (1 - self.peak_thresh)) ** 2

            # 3. Queue management (strengthened waiting time control)
            stats = self._get_queue_stats(sid)
            if stats['length'] > 0:
                avg_wait = stats['avg_wait']
                max_wait = stats['max_wait']

                # Average wait penalty
                if avg_wait > 8:  
                    site_reward -= (avg_wait - 8) * 0.3

                # Longest waiting penalty
                if max_wait > 12:  
                    site_reward -= (max_wait - 12) * 0.5
                elif max_wait > 8:  
                    site_reward -= (max_wait - 8) * 0.3

            # 4. Load smoothness
            load_change = abs(load_ratio - self.prev_loads[sid])
            if load_change < 0.1:
                site_reward += 1.0
            elif load_change > 0.2:
                site_reward -= 2.0

            # 5. Redirect reward
            for from_site, to_site, t in redirect_decisions:
                if from_site == sid:
                    # Source site: High load or long wait time for redirection reward
                    if current_loads[from_site] > 0.7 or avg_wait > 8:
                        site_reward += 5.0
                elif to_site == sid:
                    # Target site: Low load when receiving gets a reward
                    if current_loads[to_site] < 0.5:
                        site_reward += 3.0
                    elif current_loads[to_site] < 0.6:
                        site_reward += 1.0
                    else:
                        site_reward -= 2.0

            # 6. Neighbor load balancing
            neighbor_loads = []
            for nid in self.knn_idx[sid][:3]:
                if nid < self.N:
                    neighbor_loads.append(current_loads[nid])
            
            if neighbor_loads:
                load_diff = abs(load_ratio - np.mean(neighbor_loads))
                if load_diff < 0.1:
                    site_reward += 2.0
                else:
                    site_reward -= 1.0 * load_diff

            # 7. Prevent post-peak backlog release
            if is_post_peak:
                if load_ratio > 0.7 and accepts[sid] == 1:
                    site_reward -= 15.0
                elif queue_before > 5:
                    site_reward -= 5.0

            # Limit reward range
            reward[sid] = np.clip(site_reward, -30.0, 15.0)

        # Update load history
        self.prev_loads = current_loads.copy()

        # ========== Global reward ==========
        global_bonus = 0.0

        # 1. System peak control
        system_peak_ratio = np.max(current_loads)
        if system_peak_ratio < 0.7:
            global_bonus += 5.0
        elif system_peak_ratio > 0.75:
            global_bonus -= 10.0

        # 2. Waiting time control (important)
        total_wait_time = sum(wait for q in self.queue for _, wait in q)
        total_vehicles = sum(len(q) for q in self.queue)
        avg_system_wait = total_wait_time / max(total_vehicles, 1)

        if avg_system_wait > 8:  # Average wait exceeds 2 hours
            global_bonus -= (avg_system_wait - 8) * 2.0
        elif avg_system_wait < 4:  # Average wait less than 1 hour
            global_bonus += 3.0

        # 3. Load balancing
        load_std = np.std(current_loads)
        if load_std < 0.05:
            global_bonus += 5.0
        elif load_std > 0.25:
            global_bonus -= 5.0

        # 4. Peak violation penalty
        if total_violations > 0:
            global_bonus -= 8.0 * total_violations  # Reduce penalty intensity

        reward += global_bonus / self.N

        # ========== Time smoothness reward ==========
        if not hasattr(self, 'system_load_history'):
            self.system_load_history = []
        
        avg_load = np.mean(current_loads)
        self.system_load_history.append(avg_load)
        
        if len(self.system_load_history) > 8:
            self.system_load_history = self.system_load_history[-8:]
            load_changes = np.diff(self.system_load_history)
            if np.max(np.abs(load_changes)) < 0.05:
                reward += 3.0 / self.N

        # Update time step
        self.t += 1
        done = (self.t >= self.T)
        next_state = self._get_state_tensor()

        # Record peak tracking information
        if not hasattr(self, 'peak_tracking'):
            self.peak_tracking = {
                'peak_loads': [],
                'off_peak_loads': [],
                'peak_accepts': 0,
                'peak_arrivals': 0,
                'peak_violations': 0,
                'total_peak_steps': 0
            }
        
        if is_peak_hour:
            self.peak_tracking['peak_loads'].extend(current_loads)
            self.peak_tracking['peak_accepts'] += sum(accepts)
            self.peak_tracking['peak_arrivals'] += arrivals_count
            self.peak_tracking['peak_violations'] += peak_hour_violations
            self.peak_tracking['total_peak_steps'] += 1
        else:
            self.peak_tracking['off_peak_loads'].extend(current_loads)

        if done:
            sys_stats = self._get_system_queue_stats()
            
            print(f"\n=== EPISODE SUMMARY ===")

            # Peak analysis
            if self.peak_tracking['peak_loads']:
                avg_peak_load = np.mean(self.peak_tracking['peak_loads'])
                avg_off_peak = np.mean(self.peak_tracking['off_peak_loads']) if self.peak_tracking['off_peak_loads'] else 0
                print(f"Peak hour avg load: {avg_peak_load:.3f}")
                print(f"Off-peak avg load: {avg_off_peak:.3f}")
                print(f"Load shift effect: {avg_off_peak - avg_peak_load:.3f}")
                if self.peak_tracking['peak_arrivals'] > 0:
                    peak_accept_rate = self.peak_tracking['peak_accepts']/self.peak_tracking['peak_arrivals']
                    print(f"Peak hour accept rate: {peak_accept_rate:.2%}")

            # Waiting time analysis
            if sys_stats['total_vehicles'] > 0:
                print(f"\nWaiting Time Analysis:")
                print(f"  Vehicles in queue: {sys_stats['total_vehicles']}")
                print(f"  Average: {sys_stats['avg_wait']*15:.1f} minutes")
                print(f"  Maximum: {sys_stats['max_wait']*15:.1f} minutes")
                print(f"  Vehicles > 2h: {sys_stats['vehicles_over_2h']}")
                print(f"  Vehicles > 4h: {sys_stats['vehicles_over_4h']}")
            else:
                print(f"\nWaiting Time Analysis: No vehicles in queue")

            # Redirect analysis
            if hasattr(self, 'redirect_wait_times') and self.redirect_wait_times:
                print(f"\nRedirect Analysis:")
                print(f"  Total redirects: {len(self.redirect_wait_times)}")
                print(f"  Avg wait before redirect: {np.mean(self.redirect_wait_times)*15:.1f} min")
            
            print("="*25)

        # Return results
        reward_tensor = torch.tensor(reward, device=self.device).unsqueeze(-1)
        return next_state, reward_tensor, done, {
            "accept": accepts,
            "queue": sum(len(q) for q in self.queue),
            "arrivals": arrivals_count,
            "future_free": self._get_future_free(),
            "metrics": self._m.copy(),
            "future_dom": self._get_future_dom(),
            "violations": total_violations,
            "peak_hour": is_peak_hour,
            "peak_hour_violations": peak_hour_violations,
            "current_loads": current_loads,
            "redirect_decisions": redirect_decisions,
            "queue_pressure": queue_pressure,
            "accepts_this_step": accepts_this_step,
            "avg_wait_time": avg_system_wait  # 新增
        }
    
    
    def _calculate_site_load(self, sid):
        """Calculate the real-time load rate of the site"""
        pub_load = self.public_load[self.t % self.T][sid] if self.public_load is not None else (self.cap[sid] - self.free[sid]) * self.charger_kw
        dom_load = self.dom_load[self.t % self.T][sid]
        total_load = pub_load + dom_load
        return total_load / self.grid_cap[sid]

    def _predict_future_load(self, sid, horizon=4):
        """Enhanced future load prediction, supporting longer time horizons"""
        future_steps = min(horizon, self.T - self.t)
        if future_steps <= 0:
            return self._calculate_site_load(sid)
        
        future_loads = []

        # Predict future loads
        for i in range(1, future_steps + 1):
            t_future = (self.t + i) % self.T

            # Predict public charging load
            if self.public_load is not None:
                pub_load = self.public_load[t_future][sid]
            else:
                # based on current vehicles and queue
                current_vehicles = self.cap[sid] - self.free[sid]

                # consider vehicles in the queue
                queue_impact = min(len(self.queue[sid]), i // 4)  
                predicted_vehicles = min(current_vehicles + queue_impact, self.cap[sid])
                
                pub_load = predicted_vehicles * self.charger_kw

            # Predict domestic load
            dom_load = self.dom_load[t_future][sid]

            # Calculate total load ratio
            total_load = pub_load + dom_load
            load_ratio = total_load / self.grid_cap[sid]
            future_loads.append(load_ratio)

        # Return weighted average (recent steps have more weight)
        weights = np.exp(-np.arange(len(future_loads)) * 0.1)
        weights /= weights.sum()
        
        return np.average(future_loads, weights=weights)


    def _get_future_free(self):
        """
        Return the sequence of free pile numbers from t+1 to t+d, shape = (N, d_future)
        When out of bounds, wrap around to 0 (next day) - consistent with the paper "day-level wrap"
        """
        seq = []
        for i in range(1, self.d_future + 1):
            idx = (self.t + i) % self.T
            seq.append(self.free.copy())
        return np.stack(seq, axis=1).astype(np.float32)
    

    # ---------- Future domestic load ----------
    # Return the future 1…d steps domestic load (N, d_future)
    def _get_future_dom(self):
        """Return the future 1…d steps domestic load (N, d_future)"""
        idxs = [(self.t + i) % self.T for i in range(1, self.d_future + 1)]
        return self.dom_load[idxs].astype(np.float32).T

    def _can_accept_vehicle_future_safe(self, sid, lookahead=8, extra_vehicle=1, threshold=0.75):
        """
        Determine if the load is below the threshold within the future lookahead steps after accepting extra_vehicle vehicles.
        """
        t_now = self.t % self.T
        for i in range(lookahead):
            t_future = (self.t + i) % self.T
            dom_load = self.dom_load[t_future][sid]
            if self.public_load is not None:
                pub_load = self.public_load[t_future][sid]
                # Only add new vehicle load in the first time step
                if i == 0:
                    pub_load += extra_vehicle * self.charger_kw
            else:
                current_vehicles = self.cap[sid] - self.free[sid]
                future_vehicles = min(current_vehicles + (extra_vehicle if i == 0 else 0), self.cap[sid])
                pub_load = future_vehicles * self.charger_kw
            load_ratio = (pub_load + dom_load) / self.grid_cap[sid]
            if load_ratio >= threshold:
                return False
        return True   
    # ---------- Tools ----------
    # 3. Simplify state construction, remove potentially problematic normalization
    def _get_state_tensor(self, init_filter: bool = False):
        """Enhance state space, include longer future information"""
        idx_now = self.t % self.T

        # basic features
        free_ratio = (self.free / self.cap).reshape(self.N, 1).astype(np.float32)
        time_feat = np.full((self.N, 1), idx_now / self.T, dtype=np.float32)
        queue_feat = np.array([len(q) for q in self.queue], dtype=np.float32).reshape(self.N, 1) / self.max_queue

        # load features
        pub_load = ((self.cap - self.free) * self.charger_kw).reshape(self.N, 1).astype(np.float32)
        dom_load = self.dom_load[idx_now].reshape(self.N, 1).astype(np.float32)
        
        if np.any(np.isnan(dom_load)) or np.any(np.isinf(dom_load)):
            dom_load = np.where(np.isnan(dom_load) | np.isinf(dom_load), 
                            pub_load * 0.3, dom_load)
        
        grid_cap_safe = np.maximum(self.grid_cap.reshape(self.N, 1), 1.0)
        agg_ratio = (pub_load + dom_load) / grid_cap_safe

        # additional features
        hour_of_day = (self.t // 4) % 24
        is_peak = float(hour_of_day in [17, 18, 19, 20, 21])
        is_peak_feat = np.full((self.N, 1), is_peak, dtype=np.float32)

        # wait time features
        wait_time_feat = np.zeros((self.N, 1), dtype=np.float32)
        redirect_ratio_feat = np.zeros((self.N, 1), dtype=np.float32)
        
        for sid in range(self.N):
            if len(self.queue[sid]) > 0:
                avg_wait = np.mean([wait for _, wait in self.queue[sid]])
                wait_time_feat[sid, 0] = min(avg_wait / 10, 1.0)

                # Redirected vehicle ratio
                redirected = sum(1 for orig, _ in self.queue[sid] if orig != sid)
                redirect_ratio_feat[sid, 0] = redirected / len(self.queue[sid])

        # Neighbor features
        neighbor_load_feat = np.zeros((self.N, 3), dtype=np.float32)
        neighbor_queue_feat = np.zeros((self.N, 3), dtype=np.float32)
        neighbor_capacity_feat = np.zeros((self.N, 3), dtype=np.float32)

        for sid in range(self.N):
            neighbors = self.knn_idx[sid][:3]
            for i, nid in enumerate(neighbors):
                if nid < self.N:
                    n_load = self._calculate_site_load(nid)
                    neighbor_load_feat[sid, i] = n_load
                    neighbor_queue_feat[sid, i] = len(self.queue[nid]) / self.max_queue
                    neighbor_capacity_feat[sid, i] = self.free[nid] / self.cap[nid]

        # Action feasibility
        can_accept_feat = np.zeros((self.N, 1), dtype=np.float32)
        for sid in range(self.N):
            if len(self.queue[sid]) > 0 and self.free[sid] > 0:
                can_accept_feat[sid, 0] = 1.0

        # Station capacity
        capacity_norm = self.cap.reshape(self.N, 1) / np.max(self.cap)

        # Multi-time scale future information 
        # Short-term (1 hour)
        future_1h = np.array([self._predict_future_load(sid, 4) for sid in range(self.N)]).reshape(-1, 1)

        # Mid-term (4 hours)
        future_4h = np.array([self._predict_future_load(sid, 16) for sid in range(self.N)]).reshape(-1, 1)

        # Long-term (8 hours)
        future_8h = np.array([self._predict_future_load(sid, 32) for sid in range(self.N)]).reshape(-1, 1)

        # Predict queue evolution
        predicted_queue_growth = np.zeros((self.N, 1))
        for sid in range(self.N):
            if self.t > 0:
                recent_arrivals = 0
                for t_past in range(max(0, self.t-4), self.t):
                    recent_arrivals += len([s for s in self.arrivals[t_past] if s == sid])
                
                arrival_rate = recent_arrivals / 4.0
                predicted_queue_growth[sid, 0] = arrival_rate * 4  # 未来1小时的预期到达

        # concatenate all features
        state_np = np.concatenate([
            free_ratio,           # 0: vacancy ratio
            time_feat,            # 1: normalized time
            queue_feat,           # 2: queue ratio
            agg_ratio,            # 3: current total load ratio
            is_peak_feat,         # 4: is peak period
            wait_time_feat,       # 5: average wait time
            neighbor_load_feat,   # 6-8: neighbor load
            neighbor_queue_feat,  # 9-11: neighbor queue
            neighbor_capacity_feat, # 12-14: neighbor capacity
            can_accept_feat,      # 15: can accept
            redirect_ratio_feat,  # 16: redirected vehicle ratio
            capacity_norm,        # 17: station capacity
            future_1h,            # 18: 1 hour later load
            future_4h,            # 19: 4 hours later load
            future_8h,            # 20: 8 hours later load
            predicted_queue_growth # 21: predicted queue growth
        ], axis=1)
        
        if self.state_filter is None or init_filter:
            self.state_filters = {}
            continuous_features = [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 18, 19, 20, 21]
            
            for feat_idx in continuous_features:
                self.state_filters[feat_idx] = ZFilter(
                    shape=[1],
                    center=True, 
                    scale=True,
                    clip=5.0
                )
        
        state_normalized = state_np.copy()
        if hasattr(self, 'state_filters'):
            for feat_idx, filter in self.state_filters.items():
                if feat_idx < state_np.shape[1]:
                    state_normalized[:, feat_idx] = np.squeeze(filter(state_np[:, feat_idx]))
        
        if np.any(np.isnan(state_normalized)) or np.any(np.isinf(state_normalized)):
            state_normalized = np.where(
                np.isnan(state_normalized) | np.isinf(state_normalized), 
                0.0, state_normalized
            )
        
        return torch.from_numpy(state_normalized).type(TorchFloat).to(self.device)

    
    def print_reward_analysis(self, info):
        """print reward analysis information"""
        if not hasattr(self, '_reward_history'):
            self._reward_history = []
        
        if 'debug_info' in info:
            self._reward_history.append(info['debug_info'])
        
        if len(self._reward_history) >= 50:
            recent_rewards = [h['total_reward'] for h in self._reward_history[-50:]]
            recent_violations = [h['violations'] for h in self._reward_history[-50:]]
            
            print(f"\n=== REWARD ANALYSIS (Last 50 steps) ===")
            print(f"Average Total Reward: {np.mean(recent_rewards):.2f}")
            print(f"Average Violations: {np.mean(recent_violations):.2f}")
            print(f"Reward Range: {np.min(recent_rewards):.1f} to {np.max(recent_rewards):.1f}")
            print("="*40)

    def validate_reward_scale(self):
        """validate reward scale"""
        print("\n=== REWARD SCALE VALIDATION ===")
        print(f"Charger Power: {self.charger_kw} kW")
        print(f"Grid Capacity Range: {self.grid_cap.min():.1f} - {self.grid_cap.max():.1f} kW")
        print(f"Domestic Load Range: {self.dom_load.min():.1f} - {self.dom_load.max():.1f} kW")
        print(f"Expected reward range per site: -30 to +10")
        print(f"Expected episode reward range: {-30 * self.N * self.T} to {10 * self.N * self.T}")
        print("="*35)

    def get_training_diagnostics(self):
        """get training diagnostics information"""
        if hasattr(self, '_reward_history') and len(self._reward_history) > 0:
            recent = self._reward_history[-10:] if len(self._reward_history) >= 10 else self._reward_history
            
            violations = [h['violations'] for h in recent]
            rewards = [h['total_reward'] for h in recent]
            peak_loads = [h.get('peak_load', 0) for h in recent]
            
            return {
                'avg_violations_per_step': np.mean(violations),
                'avg_reward_per_step': np.mean(rewards),
                'max_peak_load': np.max(peak_loads),
                'violation_rate': sum(1 for v in violations if v > 0) / len(violations)
            }
        return None
    
    def execute_smart_redirect(self, actions, redirect_probs=None):
        """execute smart redirect"""
        redirects_executed = []
        
        for sid in range(self.N):
            if len(self.queue[sid]) == 0:
                continue
                
            current_load = self._calculate_site_load(sid)

            # if site load is high and acceptance probability is low
            if current_load > 0.7 and actions[sid] < 0.3:  # low acceptance probability
                # find best redirect target
                neighbors = self.knn_idx[sid][:self.redirect_radius]
                best_target = None
                min_load = current_load
                
                for i, nid in enumerate(neighbors):
                    if nid >= self.N:
                        continue
                        
                    neighbor_load = self._calculate_site_load(nid)
                    # only redirect to sites with significantly lower load
                    if neighbor_load < current_load - 0.2 and neighbor_load < 0.6:
                        if len(self.queue[nid]) < self.max_queue - 2:  # reserve capacity
                            if neighbor_load < min_load:
                                min_load = neighbor_load
                                best_target = nid

                # execute redirect
                if best_target is not None:
                    vehicle = self.queue[sid].pop(0)
                    self.queue[best_target].append(vehicle)
                    redirects_executed.append((sid, best_target))
                    print(f"[DEBUG] Redirect: {sid} -> {best_target}")

        return redirects_executed

    # Add these helper methods to the CsvChargingEnv class in env.py
    def _get_queue_stats(self, sid):
        """get queue statistics safely"""
        if not self.queue[sid]:
            return {
                'length': 0,
                'avg_wait': 0,
                'max_wait': 0,
                'min_wait': 0,
                'waits': []
            }
        
        waits = [wait for _, wait in self.queue[sid]]
        return {
            'length': len(self.queue[sid]),
            'avg_wait': np.mean(waits) if waits else 0,
            'max_wait': max(waits) if waits else 0,
            'min_wait': min(waits) if waits else 0,
            'waits': waits
        }

    def _get_system_queue_stats(self):
        """get system-level queue statistics"""
        all_waits = []
        total_vehicles = 0
        
        for sid in range(self.N):
            stats = self._get_queue_stats(sid)
            all_waits.extend(stats['waits'])
            total_vehicles += stats['length']
        
        if not all_waits:
            return {
                'total_vehicles': 0,
                'avg_wait': 0,
                'max_wait': 0,
                'vehicles_over_2h': 0,
                'vehicles_over_4h': 0
            }
        
        return {
            'total_vehicles': total_vehicles,
            'avg_wait': np.mean(all_waits),
            'max_wait': max(all_waits),
            'vehicles_over_2h': sum(1 for w in all_waits if w > 8),
            'vehicles_over_4h': sum(1 for w in all_waits if w > 16)
        }
    
    def get_state(self, t):
        """Keep interface consistent; return numpy (N,1)"""
        return self._get_state_tensor().cpu().numpy()
    
    # Batch observations for MADDPG/RLCHARGE
    def get_obs_all(self):
        """
        Return (N, F) numpy array for each agent's observation at its own site
        """
        return self._get_state_tensor().cpu().numpy()

    def get_knn(self):
        return self.knn_idx  