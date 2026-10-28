"""
utils/home.py  (v2.3-fix - keep ALL charts & stats)
---------------------------------------------------------
• Weekday / Weekend patterns
• Power distribution: 3 kW≈70 %, 7 kW≈28 %, 22 kW≈2 %
• Energy distribution: 70 % <10 kWh; peak 8-9 kWh; long-tail→70 kWh
• Duration distribution: 35 % <5 h; 43 % 9-15 h; 7 % >24 h
• Time period scaling to replicate report curves (0-2 h & 13-15 h peaks, 16 h spike)
• Return 15-min aggregated power curves or session details
---------------------------------------------------------
from utils.home import gen_home_curve, gen_home_sessions
curve = gen_home_curve(5000, weekday=True)  # ndarray(96,)
"""

import numpy as np
import pandas as pd

# ---------- constants ----------
DAY_MIN, RES = 24 * 60, 15
SLOTS = DAY_MIN // RES

# ---------- Start time weights ----------
# 0-2 h first peak; 7-8 h valley; 13-15 h second peak; 16 h small spike
weekday_p = np.array([
    5 ,4 ,3 , 0.6,0.7,0.9,
    1 ,0.9,1.0, 2.2,3.0,3.8,
    6 ,7 ,7 ,6 ,
    7 ,5 ,4 ,3 ,2 ,1.5,1.2,1.0
], dtype=float)
weekday_p /= weekday_p.sum()

weekend_p = np.array([
    4.5,3.5,2.5, 1.0,1.1,1.3,
    1.4,1.6,1.9, 3.5,4.5,5.0,
    5.8,6.0,5.8,5.5,
    5.0,4.0,3.8,3.5,3.0,2.5,2.0,1.6
], dtype=float)
weekend_p /= weekend_p.sum()


def _start_min(n, weekday: bool):
    hrs  = np.random.choice(np.arange(24), p=weekday_p if weekday else weekend_p, size=n)
    mins = np.random.randint(0, 60, n)
    return hrs*60 + mins, hrs

# ---------- duration ----------
def _duration_h(n, start_hr, weekday=True):
    """
    Plug-in duration generator (2017 domestic-charging style)

    • 37 %  <5 h    - log-normal (clipped at 4.8h, broader for 0-1h coverage) 
    •  2 %  5-7 h   - sparse gap (uniform) - decrease
    •  1 %  7-9 h   - valley (uniform) - lowest frequency
    • 45 %  9-15 h  - truncated N(12.5, 1.8) (increase second peak)
    •  7 %  15-24 h - shoulder (uniform) (increase medium duration)
    •  6.5%  >24 h   - long tail (uniform 26-42 h) (reduce long tail again)
    """
    r   = np.random.rand(n)
    dur = np.empty(n)

    idx_short   = r < .36      # 36% 
    idx_gap     = (r >= .36) & (r < .38)   # 2% gap 5-7h
    idx_valley  = (r >= .38) & (r < .39)   # 1% valley 7-9h
    idx_cluster = (r >= .39) & (r < .87)   # 48% second peak 9-15h
    idx_midlong = (r >= .87) & (r < .93)   # 6% shoulder 15-24h
    idx_overn   = r >= .94                 # 6% long tail >24h

    # ---------- First peak: strictly limited to 0-5h ----------
    dur[idx_short] = np.clip(np.random.lognormal(np.log(1.55), 0.8, idx_short.sum()), 0.5, 4.8)

    # ---------- 5-7h gap (sparse distribution) ----------
    dur[idx_gap] = np.random.uniform(5, 7, idx_gap.sum())

    # ---------- 7-9 h valley ----------
    dur[idx_valley] = np.random.uniform(7, 9, idx_valley.sum())

    # ---------- 9-15 h second peak ----------
    if weekday:
        # Weekday: slightly biased towards longer durations
        d2 = np.random.normal(12.8, 1.8, idx_cluster.sum())
    else:
        # Weekend: slightly biased towards shorter durations
        d2 = np.random.normal(12.2, 1.8, idx_cluster.sum())
    dur[idx_cluster] = np.clip(d2, 9, 15)

    # ---------- 15-24 h shoulder ----------
    dur[idx_midlong] = np.random.uniform(15, 24, idx_midlong.sum())

    # ---------- >24 h long tail ----------
    dur[idx_overn] = np.random.uniform(26, 42, idx_overn.sum())

    # ---------- time scaling (24 h) ----------
    hour_scale = np.array([
        1.00, 0.95, 0.90,           # 00-02 ↓
        0.90, 0.80, 0.75,           # 03-05
        0.70, 0.70, 0.72,           # 06-08
        0.78, 0.85, 0.88,           # 09-11
        1.05, 1.10, 1.12, 1.15,     # 12-15
        1.45, 1.30, 1.25, 1.20,     # 16-19
        1.15, 1.12, 1.08, 1.05      # 20-23
    ])
    dur *= hour_scale[start_hr]

    # ---------- Prevent "48 h stacking" ----------
    mask = dur > 46
    if mask.any():
        dur[mask] = np.random.uniform(26, 46, mask.sum())

    return np.clip(dur, 0.5, 48)


# ---------- energy ----------
def _energy_kwh(n, start_hr):
    """
    target:
      • 70 % <10 kWh, and <8-9 kWh> is the highest
      • The curve of the starting period retains the shape of "night high → morning low → evening rise"
    weights (sum = 0.70):
        0-2  kWh : 7 %
        2-4  kWh : 9 %
        4-6  kWh : 11 %
        6-7  kWh : 8 %
        7-8  kWh : 5 %
        8-9  kWh : 20 %
        9-10 kWh: 10 %
    """

    r   = np.random.rand(n)
    kwh = np.empty(n)

    # ----------- 0-10 kWh (64.5%) -----------
    cut = np.cumsum([.077, .077, .077, .077, .077, .001, .216])  # cumulative edges

    idx_0_2  = r < cut[0]
    idx_2_4  = (r >= cut[0]) & (r < cut[1])
    idx_4_6  = (r >= cut[1]) & (r < cut[2])
    idx_6_7  = (r >= cut[2]) & (r < cut[3])
    idx_7_8  = (r >= cut[3]) & (r < cut[4])
    idx_8_9  = (r >= cut[4]) & (r < cut[5])
    idx_9_10 = (r >= cut[5]) & (r < cut[6])

    kwh[idx_0_2]  = np.random.uniform(0.2, 2.0, idx_0_2.sum())
    kwh[idx_2_4]  = np.random.uniform(2.0, 4.0, idx_2_4.sum())
    kwh[idx_4_6]  = np.random.uniform(4.0, 6.0, idx_4_6.sum())
    kwh[idx_6_7]  = np.random.uniform(6.0, 7.0, idx_6_7.sum())
    kwh[idx_7_8]  = np.random.uniform(7.0, 8.0, idx_7_8.sum())

    # 8-9 kWh triangular distribution, mode = 8.6
    kwh[idx_8_9]  = np.random.triangular(8.0, 8.6, 9.0, idx_8_9.sum())

    # 9-10 kWh also uses triangular distribution, slightly biased to the left, will flow back to 8-9 after multiplying by 0.8-0.95
    kwh[idx_9_10] = np.random.triangular(9.0, 9.3, 10.0, idx_9_10.sum())

    # ----------- 10-70 kWh（35.5 %）-----------
    idx_hi = r >= 0.68   # Increase >10kWh proportion to 32%
    kwh[idx_hi] = np.random.lognormal(2.55, 0.60, idx_hi.sum())
    over = kwh > 70
    while over.any():                       # Avoid 70 kWh stacking
        kwh[over] = np.random.lognormal(2.55, 0.60, over.sum())
        over = kwh > 70

    # ----------- time scaling (reduce pressure on 10kWh+ interval) -----------
    hour_scale = np.array([
        1.25,1.18,1.05,1.00,0.95,0.90,   # 00-05
        0.85,0.8,0.85,0.8,0.87,0.87,   # 06-11 (increase scaling factor, reduce pressure)
        0.7,0.8,0.85,0.9,             # 12-15
        0.95,1.05,1.10,1.15,1.18,1.2,   # 16-21 (moderate increase)
        1.28,1.35                        # 22-23
    ])
    kwh *= hour_scale[start_hr]

    return np.clip(kwh, 0.5, 70)

# ---------- power ----------
def _power_kw(n):
    r = np.random.rand(n)
    return np.where(r < .70, 3.0, np.where(r < .98, 7.0, 22.0))

# ---------- core interface ----------
def gen_home_sessions(n_house:int, *, weekday:bool=True, seed:int|None=None):
    if seed is not None:
        np.random.seed(seed)
    st, hr = _start_min(n_house, weekday)
    kwh    = _energy_kwh(n_house, hr)
    dur_h  = _duration_h(n_house, hr, weekday)

    # check for safety
    if np.any(~np.isfinite(dur_h)):
        print(f"[WARNING] Found non-finite values in dur_h: {np.sum(~np.isfinite(dur_h))} values")
        dur_h = np.where(np.isfinite(dur_h), dur_h, 8.0) 
    
    if np.any(~np.isfinite(kwh)):
        print(f"[WARNING] Found non-finite values in kwh: {np.sum(~np.isfinite(kwh))} values")
        kwh = np.where(np.isfinite(kwh), kwh, 30.0)

    with np.errstate(divide='ignore', invalid='ignore'):
        power_ratio = kwh / dur_h
        power_ratio = np.where(np.isfinite(power_ratio), power_ratio, 3.75)  
    
    pw_kw = np.minimum(power_ratio, _power_kw(n_house))
    
    duration_min = dur_h * 60
    duration_min = np.where(np.isfinite(duration_min), duration_min, 480)  
    ed = (st + duration_min.astype(int)).clip(max=DAY_MIN-1)
    return pd.DataFrame(dict(start_min=st,end_min=ed,
                             duration_h=dur_h,power_kw=pw_kw,energy_kwh=kwh))

def gen_home_curve(n_house:int, *, weekday:bool=True, seed:int|None=None):
    """
    Generate a more realistic home charging load curve, correctly handling overnight charging.
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate 1.5 times the number of sessions to ensure coverage of various charging patterns
    total_sessions = int(n_house * 1.5)

    # Generate all sessions
    df = gen_home_sessions(total_sessions, weekday=weekday, seed=None)

    # Select a subset of sessions to adjust
    n_prev = int(len(df) * 0.3)
    prev_indices = np.random.choice(len(df), n_prev, replace=False)

    # These sessions mainly start in the evening/overnight of the previous day (16:00-24:00)
    for idx in prev_indices:
        # Generate start times between 16:00-24:00 for the previous day
        prev_start_hour = np.random.choice(range(16, 24), p=[0.1, 0.15, 0.2, 0.25, 0.15, 0.1, 0.05, 0.0])
        prev_start_min = prev_start_hour * 60 + np.random.randint(0, 60)

        # Adjust the start time to the previous day
        df.loc[idx, 'start_min'] = prev_start_min - DAY_MIN

        # For overnight charging, tend to have longer charging durations
        if df.loc[idx, 'duration_h'] < 8:
            df.loc[idx, 'duration_h'] *= np.random.uniform(1.5, 2.0)

    # Recalculate end times, handling non-finite values safely
    duration_min = df['duration_h'] * 60
    # Check and clean non-finite values
    if duration_min.isna().any() or ~np.isfinite(duration_min).any():
        print(f"[WARNING] Found non-finite values in duration calculation")
        duration_min = duration_min.fillna(480)  # Fill NaN with 8 hours (480 minutes)
        duration_min = np.where(np.isfinite(duration_min), duration_min, 480)
    
    df['end_min'] = df['start_min'] + duration_min.astype(int)
    
    curve = np.zeros(DAY_MIN)
    
    for r in df.itertuples():
        # Calculate the intersection with the current day
        s = max(r.start_min, 0)
        e = min(r.end_min, DAY_MIN)
        
        if e > s:
            # If it's overnight charging (starting from the previous day), the power may be reduced
            if r.start_min < 0:
                power = r.power_kw * 0.9
            else:
                power = r.power_kw
                
            curve[s:e] += power

    # Convert to 15-minute resolution
    return curve.reshape(-1, RES).mean(axis=1)

# ---------- Demo, charts & stats ----------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    homes = 4000
    c_wd, c_we = (gen_home_curve(homes, weekday=wd, seed=i)
                  for wd,i in [(True,0),(False,1)])

    # -- Aggregate Power --
    plt.figure(figsize=(8,4))
    t = np.arange(SLOTS)*0.25
    plt.plot(t,c_wd,label='Weekday')
    plt.plot(t,c_we,'--',label='Weekend')
    plt.xlabel('Hour'); plt.ylabel(f'kW aggregated ({homes//1000}×10³ homes)')
    plt.title('Domestic EV Charging – 15-min Resolution'); plt.legend()
    plt.tight_layout(); plt.show()

    # -- Basic Histograms --
    df_wd = gen_home_sessions(homes, weekday=True , seed=42)
    df_we = gen_home_sessions(homes, weekday=False, seed=43)

    fig,ax = plt.subplots(1,2,figsize=(10,4))
    ax[0].hist(df_wd.duration_h,bins=np.arange(0,49,1),alpha=.6,label='Weekday')
    ax[0].hist(df_we.duration_h,bins=np.arange(0,49,1),alpha=.6,label='Weekend')
    ax[0].set_xlabel('Duration (h)'); ax[0].set_ylabel('Frequency')
    ax[0].set_title('Plug-in Duration Histogram'); ax[0].legend()

    ax[1].hist(df_wd.energy_kwh,bins=np.arange(0,71,1),alpha=.6,label='Weekday')
    ax[1].hist(df_we.energy_kwh,bins=np.arange(0,71,1),alpha=.6,label='Weekend')
    ax[1].set_xlabel('Energy (kWh)'); ax[1].set_ylabel('Frequency')
    ax[1].set_title('Energy Supplied Histogram'); ax[1].legend()
    plt.tight_layout(); plt.show()

    # -------- Four Plots --------
    plt.figure(figsize=(12,8))

    plt.subplot(2,2,1)
    plt.hist(df_wd.energy_kwh,bins=np.arange(0,71,1),
             color='#D2691E',alpha=.8,edgecolor='k',lw=.4)
    plt.xlabel('Energy supplied, kWh'); plt.ylabel('Frequency')
    plt.title('Distribution of energy drawn (2017)')
    plt.xlim(0,70); plt.grid(alpha=.3)

    plt.subplot(2,2,2)
    plt.hist(df_wd.duration_h,bins=np.arange(0,73,1),
             color='#2E8B57',alpha=.8,edgecolor='k',lw=.4)
    plt.xlabel('Plug-in time, h'); plt.ylabel('Frequency')
    plt.title('Plug-in duration (2017)')
    plt.xlim(0,72); plt.grid(alpha=.3)

    plt.subplot(2,2,3)
    df_wd['start_hour'] = (df_wd.start_min//60).astype(int)
    hr_e = df_wd.groupby('start_hour').energy_kwh.mean()
    hrs  = np.arange(24)
    plt.plot(hrs,hr_e.reindex(hrs,fill_value=hr_e.mean()),
             'o-',color='#D2691E',lw=2,ms=4)
    plt.xlabel('Hour of start'); plt.ylabel('Energy supplied, kWh')
    plt.title('Median avg energy by start-hour')
    plt.xlim(0,23); plt.ylim(0,14); plt.grid(alpha=.3); plt.xticks(range(0,24,1))

    plt.subplot(2,2,4)
    hr_d = df_wd.groupby('start_hour').duration_h.mean()
    plt.plot(hrs,hr_d.reindex(hrs,fill_value=hr_d.mean()),
             'o-',color='#2E8B57',lw=2,ms=4)
    plt.xlabel('Hour of start'); plt.ylabel('Plug-in duration, h')
    plt.title('Median avg duration by start-hour')
    plt.xlim(0,23); plt.ylim(0,20); plt.grid(alpha=.3); plt.xticks(range(0,24,1))

    plt.tight_layout(); plt.show()

    # ---------- Statistics Output & Target Comparison ----------
    def stats(tag, s):
        print(f"{tag:<8} median={np.median(s):5.2f}  mean={s.mean():5.2f}  "
              f"std={s.std():5.2f}  min={s.min():4.2f}  max={s.max():5.2f}")

    print("\n" + "="*80)
    print("Charging Duration Distribution Statistics (hours):"); stats("Weekday", df_wd.duration_h); stats("Weekend", df_we.duration_h)
    print("\nEnergy Distribution Statistics (kWh):");       stats("Weekday", df_wd.energy_kwh); stats("Weekend", df_we.energy_kwh)
    print("\nPower Distribution Statistics (kW):");        stats("Weekday", df_wd.power_kw);   stats("Weekend", df_we.power_kw)

    target = {'<5h':35, '9-15h':31, '>24h':7, '<10kWh':70,
              'mean_dur_wd':9.1, 'mean_dur_we':9.5}
    actual = {
        '<5h':       (df_wd.duration_h<5).mean()*100,
        '9-15h':     ((df_wd.duration_h>=9)&(df_wd.duration_h<=15)).mean()*100,
        '>24h':      (df_wd.duration_h>24).mean()*100,
        '<10kWh':    (df_wd.energy_kwh<10).mean()*100,
        'mean_dur_wd': df_wd.duration_h.mean(),
        'mean_dur_we': df_we.duration_h.mean()
    }

    print("\n" + "-"*60)
    print(f"{'Target':<14}{'Target':>8}{'Actual':>8}{'Difference':>8}")
    print("-"*60)
    for k in ['<5h','9-15h','>24h','<10kWh','mean_dur_wd','mean_dur_we']:
        print(f"{k:<14}{target[k]:8.1f}{actual[k]:8.1f}{actual[k]-target[k]:8.1f}")
    print("="*80)
