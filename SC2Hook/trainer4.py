import os
import sys
import time
import struct
import mmap
import signal
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Terminal UI
# ─────────────────────────────────────────────────────────────────────────────
_TTY = sys.stdout.isatty()
def _c(code, t): return f"\033[{code}m{t}\033[0m" if _TTY else t
def cyan(t):    return _c("96", t)
def yellow(t):  return _c("93", t)
def green(t):   return _c("92", t)
def red(t):     return _c("91", t)
def dim(t):     return _c("2",  t)
def bold(t):    return _c("1",  t)
def magenta(t): return _c("95", t)

_SPARKS = " ▁▂▃▄▅▆▇█"
def sparkline(vals, width=30):
    if not vals: return " " * width
    lo, hi = min(vals), max(vals)
    span = hi - lo or 1e-9
    out = []
    for i in range(width):
        idx = int(i / width * len(vals))
        v = vals[min(idx, len(vals)-1)]
        out.append(_SPARKS[int((v-lo)/span*(len(_SPARKS)-1))])
    return "".join(out)

def fmt_time(s):
    s = int(s)
    if s < 60: return f"{s}s"
    if s < 3600: return f"{s//60}m{s%60:02d}s"
    return f"{s//3600}h{(s%3600)//60:02d}m"

def clear_lines(n):
    if _TTY:
        for _ in range(n): sys.stdout.write("\033[1A\033[2K")

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
SEQ_LEN         = 64
STATE_DIM       = 26                    # Updated with sin/cos phase channels
HIDDEN_SIZE     = 512
FLAT_DIM        = SEQ_LEN * STATE_DIM   # 1664

N_BUTTONS       = 7    # A B X Y L Z Shoulder
N_MEM_BITS      = 3    # 3-bit binary → mem op 0-7

BUFFER_CAPACITY = 8192
UPDATE_INTERVAL = 512
BATCH_SIZE      = 256
PPO_EPOCHS      = 4
GAE_LAMBDA      = 0.95
GAMMA           = 0.99
CLIP_EPS        = 0.2
ENTROPY_COEF    = 0.05      # Increased slightly to fight button saturation
VALUE_COEF      = 0.5
SELF_PRED_COEF  = 0.3
BTN_ACTIVITY_COEF = 0.25    # Slightly reduced

LR              = 3e-5
MAX_GRAD_NORM   = 0.5
RELOAD_INTERVAL = 30

MODEL_PATH      = "neura_cell.pt"

# ─────────────────────────────────────────────────────────────────────────────
# Attack Randomizer Config
# ─────────────────────────────────────────────────────────────────────────────
EXPLORE_FRAMES_TOTAL   = 200_000   # decay window: bonus fades to 0 by this frame
EXPLORE_COMBO_WINDOW   = 512       # how many recent combos to track for novelty
EXPLORE_STICK_BINS     = 16        # angular bins for stick coverage tracking
EXPLORE_COMBO_BONUS    = 1.2       # reward for a never-before-seen button combo
EXPLORE_REPEAT_PENALTY = 0.15      # penalty per repeat of a recently seen combo
EXPLORE_STICK_BONUS    = 0.5       # reward per new stick angle bin visited
EXPLORE_WILDCARD_PROB  = 0.08      # chance per batch frame to inject synthetic wild exp
WEIGHTS_PATH    = "neura_cell_best.pth"
BUFFER_PATH     = "replay_buffer.bin"
CHECKPOINT_PATH = "neura_cell_ppo.pth"

# ─────────────────────────────────────────────────────────────────────────────
# Experience struct layout (v3 - Independent buttons + 3-bit mem)
# state(26) + btn_actions(7) + mem_op(1) + params(2) + 
# btn_log_probs(7) + mem_log_prob(1) + done(1) + p1_prev(1) + p2_prev(1)
# Total = 47 floats
# ─────────────────────────────────────────────────────────────────────────────
EXP_FLOATS = STATE_DIM + N_BUTTONS + 1 + 2 + N_BUTTONS + 1 + 1 + 2   # 47
EXP_BYTES  = EXP_FLOATS * 4
HEADER_BYTES = 8

# ─────────────────────────────────────────────────────────────────────────────
# Offsets (STATE_DIM = 26)
# Total experience = 47 floats
# ─────────────────────────────────────────────────────────────────────────────
OFF_STATE        = 0
OFF_BTN_ACTIONS  = STATE_DIM                      # 26 → buttons [26:32]
OFF_MEM_OP       = STATE_DIM + N_BUTTONS          # 33
OFF_PARAM0       = STATE_DIM + N_BUTTONS + 1      # 34
OFF_PARAM1       = STATE_DIM + N_BUTTONS + 2      # 35
OFF_BTN_LOGPROBS = STATE_DIM + N_BUTTONS + 3      # 36 → button log probs [36:42]
OFF_MEM_LOGPROB  = STATE_DIM + N_BUTTONS + 3 + N_BUTTONS   # 43
OFF_DONE         = STATE_DIM + N_BUTTONS + 3 + N_BUTTONS + 1  # 44
OFF_P1_PREV      = OFF_DONE + 1                   # 45
OFF_P2_PREV      = OFF_DONE + 2                   # 46

MEM_OP_NAMES = [
    "NONE", "MEM_WRITE", "MEM_READ", "MEM_QUERY",
    "MEM_CONSOLIDATE", "MEM_BWRITE", "MEM_BQUERY", "RESET_SELF"
]
BTN_NAMES = ["A", "B", "X", "Y", "L", "Z", "Sh"]

# ─────────────────────────────────────────────────────────────────────────────
# NeuraCellNet v3 — Independent Bernoulli heads for buttons + mem bits
# ─────────────────────────────────────────────────────────────────────────────
class NeuraCellNet(nn.Module):
    """
    v3 architecture:
      - Old categorical(16) control head REMOVED
      - 7 independent button heads (Bernoulli) ADDED
      - 3-bit mem_head (Bernoulli) ADDED
      - All other heads unchanged
    This allows true multi-button actions without competition.
    """

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(STATE_DIM, HIDDEN_SIZE, num_layers=2,
                            batch_first=True, dropout=0.1)

        # Button head: 7 independent logits → Bernoulli
        self.button_head = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 128), nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(128, N_BUTTONS)
        )

        # Mem head: 3 independent logits → 3-bit mem op
        self.mem_head = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 64), nn.ReLU(),
            nn.Linear(64, N_MEM_BITS)
        )

        self.param_head = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )
        self.patch_head = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 128), nn.ReLU(),
            nn.Linear(128, 32)
        )
        self.self_pred_head = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 64), nn.ReLU(),
            nn.Linear(64, 8)
        )
        self.mem_pred_head = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )
        self.task_pred_head = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )
        self.value_head = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.view(x.size(0), SEQ_LEN, STATE_DIM)
        lstm_out, _ = self.lstm(x)
        last = lstm_out[:, -1, :]

        btn_logits = self.button_head(last)                   # (B, 7)
        mem_logits = self.mem_head(last)                      # (B, 3)
        params     = torch.tanh(self.param_head(last))        # (B, 2)
        patch      = torch.tanh(self.patch_head(last)) * 0.1  # (B, 32)
        self_pred  = self.self_pred_head(last)                # (B, 8)
        mem_pred   = self.mem_pred_head(last)                 # (B, 1)
        task_pred  = self.task_pred_head(last)                # (B, 1)
        value      = self.value_head(last)                    # (B, 1)

        return btn_logits, mem_logits, params, patch, self_pred, mem_pred, task_pred, value

    def get_action_logprob_value(self, x, btn_actions=None, mem_actions=None):
        """
        Returns actions, log probs, entropy, value, etc. for PPO.
        Uses independent Bernoulli for both buttons and mem bits.
        """
        btn_logits, mem_logits, params, patch, self_pred, mem_pred, task_pred, value = self(x)

        # Button distributions (7 independent Bernoulli)
        btn_dist = torch.distributions.Bernoulli(logits=btn_logits)
        if btn_actions is None:
            btn_actions = btn_dist.sample()
        btn_log_probs = btn_dist.log_prob(btn_actions)
        btn_log_prob  = btn_log_probs.sum(dim=-1)
        btn_entropy   = btn_dist.entropy().sum(dim=-1)

        # Mem op distribution (3 independent bits)
        mem_dist = torch.distributions.Bernoulli(logits=mem_logits)
        if mem_actions is None:
            mem_actions = mem_dist.sample()
        mem_log_prob = mem_dist.log_prob(mem_actions).sum(dim=-1)
        mem_entropy  = mem_dist.entropy().sum(dim=-1)

        total_log_prob = btn_log_prob + mem_log_prob
        total_entropy  = btn_entropy + mem_entropy

        return (btn_actions, mem_actions, total_log_prob, total_entropy,
                value.squeeze(-1), params, patch, self_pred, btn_log_probs)


# ─────────────────────────────────────────────────────────────────────────────
# ReplayBuffer, InProcessBuffer, compute_reward, compute_gae, ppo_update
# (kept mostly unchanged except minor adjustments for new action space)
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# ExplorationRandomizer
# ─────────────────────────────────────────────────────────────────────────────
# WHY reward-side instead of runner-side:
#   Injecting random actions in the runner writes them with the model's current
#   (near-zero) log probs, which causes exploding PPO ratios.  Rewarding the
#   trainer for novel behavior keeps log probs consistent and lets PPO clip
#   safely.  The bonus decays to zero so it never fights real combat rewards.
#
# Stick coverage uses sin/cos of the action angle so the model is nudged to
# sweep the full circle, not just learn "go right" and "go up" independently.
#
# Button coverage uses a bitmask histogram over all 128 possible 7-button
# combos.  Every new combo seen gets a novelty bonus; repeats get a small
# penalty.  Both decay linearly with total training frames.
# ─────────────────────────────────────────────────────────────────────────────
class ExplorationRandomizer:
    """
    Tracks novelty of (button combo, stick angle) pairs and returns a shaped
    bonus/penalty that decays linearly over EXPLORE_FRAMES_TOTAL frames.

    Usage
    -----
        explorer = ExplorationRandomizer()
        ...
        # inside compute_reward (or the training loop):
        bonus = explorer.step(btn_actions, param0, param1, total_frames)
        reward += bonus
    """

    ALL_COMBOS = 128   # 2^7 possible button masks

    def __init__(self):
        # Bitmask visit counts for all 128 button combos
        self.combo_counts  = np.zeros(self.ALL_COMBOS, dtype=np.int32)
        # Rolling window of recent combo masks (for repeat detection)
        self.recent_combos = deque(maxlen=EXPLORE_COMBO_WINDOW)
        # Angular bin visit flags  (sin/cos of stick angle → bin index)
        self.angle_bins_seen = np.zeros(EXPLORE_STICK_BINS, dtype=bool)
        # Counters
        self.total_novel_combos = 0
        self.total_novel_angles = 0

    # ── internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _btn_mask(btn_actions: np.ndarray) -> int:
        """Convert 7-element float array → integer bitmask 0-127."""
        mask = 0
        for i, v in enumerate(btn_actions[:7]):
            if v > 0.5:
                mask |= (1 << i)
        return mask

    @staticmethod
    def _stick_bin(param0: float, param1: float) -> int:
        """
        Map (sX, sY) to one of EXPLORE_STICK_BINS angular sectors.
        Uses atan2 so the full circle is covered.
        Dead-center (tiny magnitude) maps to bin -1 (ignored).
        """
        mag = (param0 ** 2 + param1 ** 2) ** 0.5
        if mag < 0.05:
            return -1   # dead zone — no coverage credit
        angle = np.arctan2(param1, param0)              # [-π, π]
        norm  = (angle + np.pi) / (2 * np.pi)           # [0, 1)
        return int(norm * EXPLORE_STICK_BINS) % EXPLORE_STICK_BINS

    @staticmethod
    def _decay(total_frames: int) -> float:
        """Linear decay: 1.0 at frame 0, 0.0 at EXPLORE_FRAMES_TOTAL."""
        return max(0.0, 1.0 - total_frames / EXPLORE_FRAMES_TOTAL)

    # ── public API ────────────────────────────────────────────────────────────

    def step(self, btn_actions: np.ndarray,
             param0: float, param1: float,
             total_frames: int) -> float:
        """
        Call once per experience.  Returns a shaped exploration bonus
        (can be negative when the agent is stuck repeating the same combo).
        """
        scale = self._decay(total_frames)
        if scale == 0.0:
            return 0.0   # exploration phase is over

        bonus = 0.0
        mask  = self._btn_mask(btn_actions)

        # Penalise mashing all (or nearly all) buttons — not a useful combo
        bit_count = bin(mask).count('1')
        if bit_count >= 5:
            return -0.4 * scale   # hard discourage, skip novelty logic entirely

        # ── 1. Button combo novelty ───────────────────────────────────────
        if self.combo_counts[mask] == 0:
            # Brand-new combo — big bonus
            bonus += EXPLORE_COMBO_BONUS
            self.total_novel_combos += 1
        else:
            # Count how many times it appeared in the recent window
            recent_count = sum(1 for c in self.recent_combos if c == mask)
            if recent_count > 0:
                bonus -= EXPLORE_REPEAT_PENALTY * min(recent_count, 4)

        self.combo_counts[mask] += 1
        self.recent_combos.append(mask)

        # ── 2. Stick angle coverage (sin/cos aware) ───────────────────────
        abin = self._stick_bin(param0, param1)
        if abin >= 0 and not self.angle_bins_seen[abin]:
            bonus += EXPLORE_STICK_BONUS
            self.angle_bins_seen[abin] = True
            self.total_novel_angles += 1

        return float(bonus * scale)

    def generate_wildcard_exp(self) -> np.ndarray:
        """
        Synthesise a random experience with a uniformly random button combo
        and a stick direction sampled on the unit circle (sin/cos uniform).
        Used to seed the buffer with hard-to-reach combos early in training.
        """
        exp = np.zeros(EXP_FLOATS, dtype=np.float32)

        # Random game-like state
        exp[:STATE_DIM] = np.random.uniform(0, 1, STATE_DIM).astype(np.float32)
        # sin/cos phase channels should be in [-1, 1]
        phase = np.random.uniform(0, 2 * np.pi)
        exp[24] = np.sin(phase)
        exp[25] = np.cos(phase)

        # Uniform random button combo (all 128 equally likely)
        combo_mask = np.random.randint(0, 128)
        for i in range(N_BUTTONS):
            exp[OFF_BTN_ACTIONS + i] = float((combo_mask >> i) & 1)

        # Stick: uniform direction on unit circle, random magnitude [0.3, 1.0]
        angle = np.random.uniform(0, 2 * np.pi)
        mag   = np.random.uniform(0.3, 1.0)
        exp[OFF_PARAM0] = float(np.cos(angle) * mag)
        exp[OFF_PARAM1] = float(np.sin(angle) * mag)

        exp[OFF_MEM_OP] = float(np.random.randint(0, 8))

        # Log probs for a fair coin (Bernoulli p=0.5) → log(0.5) per bit
        exp[OFF_BTN_LOGPROBS:OFF_BTN_LOGPROBS + N_BUTTONS] = np.log(0.5)
        exp[OFF_MEM_LOGPROB] = np.log(0.5) * N_MEM_BITS

        exp[OFF_DONE]    = 0.0
        exp[OFF_P1_PREV] = np.random.uniform(0, 0.5)
        exp[OFF_P2_PREV] = np.random.uniform(0, 0.5)
        return exp

    def coverage_summary(self) -> str:
        n_combos = int((self.combo_counts > 0).sum())
        n_angles = int(self.angle_bins_seen.sum())
        return (f"combos {n_combos}/{self.ALL_COMBOS} "
                f"| angles {n_angles}/{EXPLORE_STICK_BINS} "
                f"| novel_combos {self.total_novel_combos} "
                f"| novel_angles {self.total_novel_angles}")


# ReplayBuffer class remains the same as in your last version
class ReplayBuffer:
    def __init__(self, path: str, capacity: int):
        self.path = path
        self.capacity = capacity
        self.mm = None
        self.file = None
        self._last_read_total = 0
        self._local_buffer: deque = deque(maxlen=capacity)

    def wait_for_runner(self, timeout: int = 300):
        print(f" Waiting for runner to create {self.path}...")
        waited = 0
        while not os.path.exists(self.path):
            time.sleep(1); waited += 1
            if waited % 10 == 0: print(f" Still waiting... ({waited}s)")
            if waited > timeout:
                raise TimeoutError(f"Runner did not create {self.path} within {timeout}s")
        print(f" {green('Buffer file found!')} Opening...\n")
        self._open()

    def _open(self):
        self.file = open(self.path, 'r+b')
        self.mm = mmap.mmap(self.file.fileno(), 0)

    def _read_header(self):
        self.mm.seek(0)
        data = self.mm.read(HEADER_BYTES)
        write_head, total_written = struct.unpack('<II', data)
        return write_head, total_written

    def _read_exp(self, slot: int) -> Optional[np.ndarray]:
        offset = HEADER_BYTES + slot * EXP_BYTES
        if offset + EXP_BYTES > len(self.mm): return None
        self.mm.seek(offset)
        raw = self.mm.read(EXP_BYTES)
        return np.frombuffer(raw, dtype=np.float32).copy()

    def sync(self) -> int:
        if self.mm is None: return 0
        write_head, total_written = self._read_header()
        new_frames = total_written - self._last_read_total
        if new_frames <= 0: return 0
        new_frames = min(new_frames, self.capacity)
        for i in range(new_frames):
            slot = (write_head - new_frames + i) % self.capacity
            exp = self._read_exp(slot)
            if exp is not None and len(exp) == EXP_FLOATS:
                self._local_buffer.append(exp)
        self._last_read_total = total_written
        return new_frames

    def __len__(self): return len(self._local_buffer)

    def sample(self, n: int):
        indices = np.random.choice(len(self._local_buffer), size=n, replace=False)
        return np.stack([self._local_buffer[i] for i in indices], axis=0)

    def close(self):
        if self.mm: self.mm.close()
        if self.file: self.file.close()


# ─────────────────────────────────────────────────────────────────────────────
#  In-process fallback buffer
# ─────────────────────────────────────────────────────────────────────────────
class InProcessBuffer:
    def __init__(self, csv_path: str = "rollout_log.csv", capacity: int = 8192):
        self.csv_path    = csv_path
        self.capacity    = capacity
        self._buffer: deque = deque(maxlen=capacity)
        self._last_mtime = 0

    def sync(self) -> int:
        if not os.path.exists(self.csv_path):
            return self._generate_synthetic(64)
        mtime = os.path.getmtime(self.csv_path)
        if mtime == self._last_mtime: return 0
        self._last_mtime = mtime
        try:
            import pandas as pd
            df = pd.read_csv(self.csv_path)
            new = 0
            for _, row in df.tail(512).iterrows():
                exp = np.zeros(EXP_FLOATS, dtype=np.float32)
                for i in range(STATE_DIM):
                    col = f'state_{i}'
                    exp[i] = float(row[col]) if col in row else 0.0
                # btn_actions
                for b in range(N_BUTTONS):
                    exp[OFF_BTN_ACTIONS + b] = float(row.get(f'btn_{b}', 0))
                exp[OFF_MEM_OP]  = float(row.get('mem_op', 0))
                exp[OFF_PARAM0]  = float(row.get('param0', 0))
                exp[OFF_PARAM1]  = float(row.get('param1', 0))
                for b in range(N_BUTTONS):
                    exp[OFF_BTN_LOGPROBS + b] = float(row.get(f'btn_logp_{b}', -0.693))
                exp[OFF_MEM_LOGPROB] = float(row.get('mem_logp', -2.079))
                exp[OFF_DONE]    = float(row.get('done', 0))
                exp[OFF_P1_PREV] = float(row.get('p1_prev', 0))
                exp[OFF_P2_PREV] = float(row.get('p2_prev', 0))
                self._buffer.append(exp)
                new += 1
            return new
        except Exception as e:
            print(f"  [Buffer] CSV read error: {e}")
            return 0

    def _generate_synthetic(self, n: int) -> int:
        for _ in range(n):
            exp = np.zeros(EXP_FLOATS, dtype=np.float32)
            exp[:STATE_DIM] = np.random.uniform(0, 1, STATE_DIM).astype(np.float32)
            # Random button actions (sparse — most buttons off most of the time)
            exp[OFF_BTN_ACTIONS:OFF_BTN_ACTIONS+N_BUTTONS] = \
                (np.random.random(N_BUTTONS) < 0.15).astype(np.float32)
            exp[OFF_MEM_OP]  = float(np.random.randint(0, 8))
            exp[OFF_PARAM0]  = np.random.uniform(-1, 1)
            exp[OFF_PARAM1]  = np.random.uniform(-1, 1)
            # Log probs for independent Bernoulli at p=0.5: log(0.5) = -0.693 each
            exp[OFF_BTN_LOGPROBS:OFF_BTN_LOGPROBS+N_BUTTONS] = np.log(0.5)
            exp[OFF_MEM_LOGPROB] = np.log(0.5) * N_MEM_BITS
            exp[OFF_DONE]    = float(np.random.random() < 0.01)
            exp[OFF_P1_PREV] = np.random.uniform(0, 0.3)
            exp[OFF_P2_PREV] = np.random.uniform(0, 0.3)
            self._buffer.append(exp)
        return n

    def __len__(self): return len(self._buffer)

    def sample(self, n: int):
        indices = np.random.choice(len(self._buffer), size=n, replace=False)
        return np.stack([self._buffer[i] for i in indices], axis=0)



# Reward function - strongly boosted stick + harsher button penalties
def compute_reward(exp: np.ndarray, prev_self_err: float,
                   explorer: 'ExplorationRandomizer' = None,
                   total_frames: int = 0) -> float:
    state = exp[OFF_STATE:OFF_STATE + STATE_DIM]
    btn_actions = exp[OFF_BTN_ACTIONS:OFF_BTN_ACTIONS + N_BUTTONS]
    mem_op = int(exp[OFF_MEM_OP])
    param0 = exp[OFF_PARAM0]
    param1 = exp[OFF_PARAM1]
    done = bool(exp[OFF_DONE] > 0.5)
    p1_pct_prev = exp[OFF_P1_PREV]
    p2_pct_prev = exp[OFF_P2_PREV]
    p1_pct = state[8]
    p2_pct = state[9]
    mem_hit = state[1]
    self_err = state[21]
    stock_delta = state[6]

    reward = 0.0

    # ── Stick movement reward ─────────────────────────────────────────────────
    stick_mag = abs(param0) + abs(param1)
    if stick_mag < 0.18:
        reward -= 0.3          # penalise dead-stick hard
    elif stick_mag > 0.65:
        reward += 0.25         # reward committed movement
    else:
        reward += 0.05         # small reward for any movement

    # Directional variety bonus: reward non-axis-aligned angles
    if stick_mag > 0.1:
        angle = np.arctan2(param1, param0)
        # sin(2θ) peaks at 45° diagonals — encourages non-trivial directions
        reward += abs(np.sin(2 * angle)) * 0.08

    # ── Combat rewards ────────────────────────────────────────────────────────
    p2_delta = p2_pct - p2_pct_prev
    p1_delta = p1_pct - p1_pct_prev
    if p2_delta > 0.001:
        reward += 3.0 * p2_delta   # dealing damage
    if p1_delta > 0.001:
        reward -= 2.5 * p1_delta   # taking damage

    if done:
        reward += 2.0 if stock_delta > 0 else -5.0

    # ── Self-modeling bonus ───────────────────────────────────────────────────
    err_delta = prev_self_err - self_err
    if err_delta > 0:
        reward += err_delta * 2.0  # improving self-model
    reward -= self_err * 0.1       # small pressure to keep error low

    # ── Memory bonus ─────────────────────────────────────────────────────────
    if mem_hit > 0.7:
        reward += 0.15 * mem_hit

    # ── Button discipline ─────────────────────────────────────────────────────
    btn_sum = int(btn_actions.sum())

    # Hard penalty for mashing all buttons at once — the main collapse symptom
    if btn_sum >= 5:
        reward -= 0.6 + (btn_sum - 5) * 0.3   # −0.6 for 5, −0.9 for 6, −1.2 for all 7

    # Mild penalty for pressing more than 2 buttons with no damage result
    if btn_sum > 2 and abs(p2_delta) < 0.001 and abs(p1_delta) < 0.001:
        reward -= (btn_sum - 2) * 0.08

    # Penalty for holding shield/dodge buttons with no effect
    btn_L  = btn_actions[4] > 0.5
    btn_Sh = btn_actions[6] > 0.5
    btn_Y  = btn_actions[3] > 0.5
    if (btn_L or btn_Sh) and abs(p1_delta) < 0.001:
        reward -= 0.1
    if btn_Y and abs(p1_delta) < 0.001:
        reward -= 0.05

    # Penalty for doing absolutely nothing
    if btn_sum == 0 and mem_op == 0:
        reward -= 0.15

    # ── Exploration bonus (decays to 0 over EXPLORE_FRAMES_TOTAL frames) ──
    if explorer is not None:
        reward += explorer.step(btn_actions, param0, param1, total_frames)

    return float(reward)


# GAE and ppo_update remain the same as your last version
# (I kept them unchanged except for using the new get_action_logprob_value)

def compute_gae(rewards, values, dones, gamma=GAMMA, lam=GAE_LAMBDA):
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t+1] * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        advantages[t] = gae
    returns = advantages + values[:T]
    return advantages, returns


def ppo_update(model, optimizer, batch_exp, device,
               old_log_probs, advantages, returns, self_next_targets):
    B = len(batch_exp)

    states_single = torch.tensor(batch_exp[:, OFF_STATE:OFF_STATE+STATE_DIM],
                                 dtype=torch.float32, device=device)
    states_seq = states_single.unsqueeze(1).expand(B, SEQ_LEN, STATE_DIM).clone()
    for t in range(SEQ_LEN - 1):
        noise_scale = 0.05 * (1.0 - t / SEQ_LEN)
        states_seq[:, t, :] += torch.randn_like(states_seq[:, t, :]) * noise_scale
    states_flat = states_seq.view(B, SEQ_LEN * STATE_DIM)

    btn_acts_np = batch_exp[:, OFF_BTN_ACTIONS:OFF_BTN_ACTIONS+N_BUTTONS]
    mem_bits_np = np.zeros((B, N_MEM_BITS), dtype=np.float32)
    for i in range(B):
        op = int(batch_exp[i, OFF_MEM_OP])
        for bit in range(N_MEM_BITS):
            mem_bits_np[i, bit] = float((op >> bit) & 1)

    btn_acts_t = torch.tensor(btn_acts_np, dtype=torch.float32, device=device)
    mem_bits_t = torch.tensor(mem_bits_np, dtype=torch.float32, device=device)

    advantages_t = (advantages.to(device) - advantages.mean()) / (advantages.std() + 1e-8)
    returns_t = returns.to(device)
    old_lp = old_log_probs.to(device)

    total_loss_sum = policy_loss_sum = value_loss_sum = 0.0
    entropy_sum = aux_loss_sum = btn_act_sum = 0.0

    for epoch in range(PPO_EPOCHS):
        perm = torch.randperm(B, device=device)
        for start in range(0, B, BATCH_SIZE):
            idx = perm[start:start+BATCH_SIZE]
            if len(idx) < 2: continue

            x_batch = states_flat[idx]
            btn_a_batch = btn_acts_t[idx]
            mem_a_batch = mem_bits_t[idx]
            adv_batch = advantages_t[idx]
            ret_batch = returns_t[idx]
            old_lp_batch = old_lp[idx]

            (_, _, log_prob, entropy, value, params, patch, self_pred, _) = \
                model.get_action_logprob_value(x_batch, btn_a_batch, mem_a_batch)

            ratio = torch.exp(log_prob - old_lp_batch)
            surr1 = ratio * adv_batch
            surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv_batch
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = nn.functional.mse_loss(value, ret_batch)
            entropy_loss = -entropy.mean()

            if self_next_targets is not None:
                snt = self_next_targets[idx].to(device)
                aux_loss = nn.functional.mse_loss(self_pred, snt)
            else:
                aux_loss = torch.tensor(0.0, device=device)

            # Button diversity regularization:
            #   - Penalise the mean prob being too HIGH (all-buttons collapse)
            #   - Penalise the mean prob being too LOW  (dead buttons)
            #   - Reward variance across buttons (they should fire selectively)
            btn_probs = torch.sigmoid(model.button_head(
                model.lstm(x_batch.view(len(idx), SEQ_LEN, STATE_DIM))[0][:, -1, :]))
            btn_mean = btn_probs.mean(dim=-1)          # per-sample mean across 7 buttons
            btn_var  = btn_probs.var(dim=-1)           # per-sample variance across 7 buttons

            # Push mean toward 0.2 (roughly 1-2 buttons active on average)
            btn_act_loss  = ((btn_mean - 0.20) ** 2).mean()
            # Reward variance — high variance = selective; low = all same
            btn_act_loss -= btn_var.mean() * 0.5
            btn_act_loss  = torch.clamp(btn_act_loss, min=0.0)

            loss = (policy_loss +
                    VALUE_COEF * value_loss +
                    ENTROPY_COEF * entropy_loss +
                    SELF_PRED_COEF * aux_loss +
                    BTN_ACTIVITY_COEF * btn_act_loss)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            total_loss_sum += loss.item()
            policy_loss_sum += policy_loss.item()
            value_loss_sum += value_loss.item()
            entropy_sum += entropy.mean().item()
            aux_loss_sum += aux_loss.item()
            btn_act_sum += btn_act_loss.item()

    n_steps = PPO_EPOCHS * max(1, B // BATCH_SIZE)
    return {
        'total': total_loss_sum / n_steps,
        'policy': policy_loss_sum / n_steps,
        'value': value_loss_sum / n_steps,
        'entropy': entropy_sum / n_steps,
        'aux': aux_loss_sum / n_steps,
        'btn_act': btn_act_sum / n_steps,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Pretraining Phase - Systematic Action Space Bootstrapping
# 30,000 steps covering button combinations + directional stick exploration
# ─────────────────────────────────────────────────────────────────────────────
def run_pretraining(model, optimizer, device, num_steps=30000):
    print(bold("\n" + "="*60))
    print(bold("     Starting Improved Pretraining Phase (30,000 steps)"))
    print(bold("     Teaching button combos + full stick range"))
    print(bold("="*60 + "\n"))

    import math
    from tqdm import tqdm

    model.train()
    step = 0
    pretrain_losses = deque(maxlen=500)
    best_loss = float('inf')

    pbar = tqdm(total=num_steps, desc="Pretraining", ncols=100)

    while step < num_steps:
        combo_idx     = step % 128
        direction_idx = (step // 128) % 16
        magnitude_idx = (step // (128 * 16)) % 5

        # Target buttons
        target_btn = torch.zeros(1, N_BUTTONS, device=device)
        for i in range(N_BUTTONS):
            if (combo_idx & (1 << i)):
                target_btn[0, i] = 1.0

        # Target stick (16 directions, 5 magnitudes)
        angle = direction_idx * (2 * math.pi / 16.0)
        magnitudes = [0.30, 0.50, 0.70, 0.85, 1.00]
        mag = magnitudes[magnitude_idx]

        target_stick = torch.tensor([[mag * math.sin(angle),
                                      mag * math.cos(angle)]], device=device)

        # Better fake state: add some basic structure instead of pure zeros
        fake_state = torch.zeros(1, FLAT_DIM, device=device)
        # Add some fake "game state" signal so LSTM has something to work with
        fake_state[0, 8] = 0.5   # fake p1 health
        fake_state[0, 9] = 0.5   # fake p2 health
        fake_state[0, 22] = 0.4  # fake distance
        fake_state[0, 24] = math.sin(step * 0.05)  # phase
        fake_state[0, 25] = math.cos(step * 0.05)

        # Forward
        btn_logits, mem_logits, params, _, self_pred, _, _, value = model(fake_state)

        # Losses
        btn_loss = nn.functional.binary_cross_entropy_with_logits(btn_logits, target_btn)
        stick_loss = nn.functional.mse_loss(torch.tanh(params), target_stick)

        # Balanced total loss
        loss = btn_loss + 1.5 * stick_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()

        pretrain_losses.append(loss.item())
        if loss.item() < best_loss:
            best_loss = loss.item()

        if step % 200 == 0 or step == num_steps - 1:
            avg_loss = np.mean(pretrain_losses) if pretrain_losses else 0.0
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg': f'{avg_loss:.4f}',
                'best': f'{best_loss:.4f}',
                'btn_loss': f'{btn_loss.item():.4f}',
                'stick_loss': f'{stick_loss.item():.4f}'
            })

        pbar.update(1)
        step += 1

    pbar.close()

    print(bold(f"\n Pretraining completed!"))
    print(f"   Final avg loss : {np.mean(pretrain_losses):.4f}")
    print(f"   Best loss      : {best_loss:.4f}")
    print(f"   Steps completed: {num_steps:,}\n")

    return model
# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print(bold("\n ╔══════════════════════════════════════════════╗"))
    print(bold(" ║") + cyan(" NeuraCell v3 — Independent Buttons + Mem Bits ") + bold("║"))
    print(bold(" ╚══════════════════════════════════════════════╝\n"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" {dim('device')} {cyan(str(device))}")
    print(f" {dim('action space')} {cyan(f'{N_BUTTONS} independent buttons + {N_MEM_BITS} mem bits')}")
    print(f" {dim('exp floats')} {cyan(str(EXP_FLOATS))}")
    print(f" {dim('update every')} {cyan(str(UPDATE_INTERVAL))} frames\n")

    # ─────────────────────────────────────────────────────────────────────────────
    # Model Creation
    # ─────────────────────────────────────────────────────────────────────────────
    model = NeuraCellNet().to(device)

    # Calculate number of parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f" {dim('model params')} {cyan(f'{n_params:,}')}\n")

    # ── Create Optimizer FIRST ─────────────────────────────────────────────────
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-6)

    # ── Pretraining Phase ─────────────────────────────────────────────────────
    # This runs only the first time (when no checkpoint exists)
    if not os.path.exists(CHECKPOINT_PATH):
        print(yellow(" No checkpoint found → Starting structured pretraining (30,000 steps)..."))
        print(yellow(" This will take approximately 12-18 minutes on your G750JW.\n"))

        model = run_pretraining(model, optimizer, device, num_steps=30000)
        
        # Save the pretrained model
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        print(green(f" Pretraining completed and saved to {CHECKPOINT_PATH}"))
        print(green(" Proceeding with normal PPO training.\n"))
    else:
        print(green(f" Found existing checkpoint → Loading {CHECKPOINT_PATH}"))
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device), strict=False)
        print(green(" Model loaded successfully.\n"))

    # ── Replay buffer setup ───────────────────────────────────────────────────
    try:
        buf_mmap = ReplayBuffer(BUFFER_PATH, BUFFER_CAPACITY)
        if os.path.exists(BUFFER_PATH):
            buf_mmap._open()
            buffer = buf_mmap
            print(f" {green('Connected to live runner buffer')} ({BUFFER_PATH})\n")
        else:
            print(f" {yellow('Runner buffer not found — using in-process fallback')}\n")
            buffer = InProcessBuffer(capacity=BUFFER_CAPACITY)
    except Exception as e:
        print(f" {yellow(f'Buffer open failed ({e}) — using fallback')}\n")
        buffer = InProcessBuffer(capacity=BUFFER_CAPACITY)

    total_updates    = 0
    total_frames     = 0
    reward_history   = deque(maxlen=200)
    policy_history   = deque(maxlen=100)
    value_history    = deque(maxlen=100)
    entropy_history  = deque(maxlen=100)
    prev_self_err    = 0.5
    last_save_time   = time.time()
    start_time       = time.time()
    DISPLAY_LINES    = 0
    running          = True

    # Attack randomizer — seeds exploration via shaped rewards
    explorer = ExplorationRandomizer()
    print(f" {dim('explore window')} {cyan(f'{EXPLORE_FRAMES_TOTAL:,}')} frames  "
          f"{dim('combo bonus')} {cyan(str(EXPLORE_COMBO_BONUS))}  "
          f"{dim('stick bins')} {cyan(str(EXPLORE_STICK_BINS))}\n")

    def handle_exit(sig, frame):
        nonlocal running
        print(f"\n\n  {yellow('Stopping trainer...')}")
        running = False
    signal.signal(signal.SIGINT,  handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    min_frames_to_start = UPDATE_INTERVAL
    print(f"  Waiting for {min_frames_to_start} frames before first update...")
    while running and len(buffer) < min_frames_to_start:
        new = buffer.sync(); total_frames += new
        if new > 0:
            print(f"\r  Buffer: {len(buffer)}/{min_frames_to_start} frames", end="")
        time.sleep(0.5)
    print(f"\n  {green('Buffer ready!')} Starting PPO updates.\n")

    frames_since_update = 0

    while running:
        new_frames = buffer.sync()
        total_frames        += new_frames
        frames_since_update += new_frames

        if frames_since_update >= UPDATE_INTERVAL and len(buffer) >= BATCH_SIZE * 2:
            frames_since_update = 0

            n_sample = min(len(buffer), BATCH_SIZE * PPO_EPOCHS * 2)
            batch    = buffer.sample(n_sample)

            # ── Rewards ───────────────────────────────────────────────────────
            rewards = np.zeros(n_sample, dtype=np.float32)
            dones   = batch[:, OFF_DONE].astype(np.float32)

            # Inject wildcard experiences to seed unseen button combos early
            decay = max(0.0, 1.0 - total_frames / EXPLORE_FRAMES_TOTAL)
            n_wildcards = int(n_sample * EXPLORE_WILDCARD_PROB * decay)
            if n_wildcards > 0:
                wildcards = np.stack([explorer.generate_wildcard_exp()
                                      for _ in range(n_wildcards)], axis=0)
                batch = np.concatenate([batch, wildcards], axis=0)
                rewards = np.concatenate([rewards,
                                          np.zeros(n_wildcards, dtype=np.float32)])
                dones = np.concatenate([dones,
                                        np.zeros(n_wildcards, dtype=np.float32)])
                n_sample = len(batch)

            for i in range(n_sample):
                r = compute_reward(batch[i], prev_self_err, explorer, total_frames)
                rewards[i] = r
                reward_history.append(r)
                prev_self_err = prev_self_err * 0.99 + float(batch[i][STATE_DIM - 3]) * 0.01

            # ── Values for GAE ────────────────────────────────────────────────
            model.eval()
            with torch.no_grad():
                states_single = torch.tensor(
                    batch[:, OFF_STATE:OFF_STATE+STATE_DIM],
                    dtype=torch.float32, device=device)
                states_seq   = states_single.unsqueeze(1).expand(
                    n_sample, SEQ_LEN, STATE_DIM).clone()
                states_flat  = states_seq.reshape(n_sample, SEQ_LEN * STATE_DIM)
                *_, values, _, _, _ = model.get_action_logprob_value(states_flat)
                values_np = values.cpu().numpy()
                bootstrap  = np.append(values_np, values_np[-1])

                # Old log probs = sum of per-button + mem log probs stored at collection
                btn_lp_stored = torch.tensor(
                    batch[:, OFF_BTN_LOGPROBS:OFF_BTN_LOGPROBS+N_BUTTONS], device=device)
                mem_lp_stored = torch.tensor(
                    batch[:, OFF_MEM_LOGPROB:OFF_MEM_LOGPROB+1], device=device)
                old_log_probs = (btn_lp_stored.sum(dim=-1) + mem_lp_stored.squeeze(-1))

            # ── GAE ───────────────────────────────────────────────────────────
            advantages, returns_ = compute_gae(rewards, bootstrap, dones)
            advantages_t = torch.tensor(advantages)
            returns_t    = torch.tensor(returns_)

            # Self-next targets from state[12:20]
            self_next_t = torch.tensor(
                batch[:, OFF_STATE+12:OFF_STATE+20], dtype=torch.float32)

            # ── PPO ───────────────────────────────────────────────────────────
            model.train()
            losses = ppo_update(model, optimizer, batch, device,
                                 old_log_probs, advantages_t, returns_t, self_next_t)
            total_updates += 1

            policy_history.append(losses['policy'])
            value_history.append(losses['value'])
            entropy_history.append(losses['entropy'])

            # ── Display ───────────────────────────────────────────────────────
            if _TTY: clear_lines(DISPLAY_LINES)

            avg_r = float(np.mean(reward_history)) if reward_history else 0.0
            max_r = float(np.max(reward_history))  if reward_history else 0.0
            min_r = float(np.min(reward_history))  if reward_history else 0.0
            r_spark = sparkline(list(reward_history), width=28)
            p_spark = sparkline(list(policy_history), width=14)
            v_spark = sparkline(list(value_history),  width=14)
            reward_color = green if avg_r > 0 else red

            lines = [
                (f"  {bold(f'Update {total_updates:>5}')}"
                 f"  {dim('frames')} {cyan(f'{total_frames:,}')}"
                 f"  {dim('buffer')} {cyan(f'{len(buffer):,}')}"
                 f"  {dim('elapsed')} {fmt_time(time.time()-start_time)}"),

                (f"  {dim('reward')}  avg {reward_color(f'{avg_r:+.3f}')}"
                 f"  min {red(f'{min_r:+.3f}')}"
                 f"  max {green(f'{max_r:+.3f}')}"),

                (f"  {dim('reward curve')}  {magenta(r_spark)}"),

                (f"  {dim('policy')} {yellow(f'{losses['policy']:+.4f}')}"
                 f"  {dim('value')} {cyan(f'{losses['value']:.4f}')}"
                 f"  {dim('entropy')} {green(f'{losses['entropy']:.4f}')}"
                 f"  {dim('aux')} {cyan(f'{losses['aux']:.4f}')}"
                 f"  {dim('btn_act')} {yellow(f'{losses['btn_act']:.4f}')}"),
                (f"  {dim('policy curve')}  {yellow(p_spark)}"
                 f"   {dim('value curve')}  {cyan(v_spark)}"),

                (f"  {dim('self_err')} {yellow(f'{prev_self_err:.4f}')}"
                 f"  {dim('adv mean')} {cyan(f'{advantages.mean():+.4f}')}"
                 f"  {dim('adv std')}  {cyan(f'{advantages.std():.4f}')}"),

                (f"  {dim('explore')}  {magenta(explorer.coverage_summary())}"
                 f"  {dim('decay')} {cyan(f'{max(0.0, 1.0-total_frames/EXPLORE_FRAMES_TOTAL):.3f}')}"),
            ]
            for ln in lines: print(ln)
            sys.stdout.flush()
            DISPLAY_LINES = len(lines)

            # ── Hot-reload ────────────────────────────────────────────────────
            now = time.time()
            if now - last_save_time > RELOAD_INTERVAL:
                last_save_time = now
                model.eval()
                torch.save(model.state_dict(), CHECKPOINT_PATH)
                try:
                    example = torch.zeros(1, SEQ_LEN * STATE_DIM)
                    traced  = torch.jit.trace(model, example)
                    traced.save(MODEL_PATH)
                    if _TTY:
                        clear_lines(DISPLAY_LINES); DISPLAY_LINES = 0
                    print(f"  {green('✓')} Weights saved → {MODEL_PATH}"
                          f"  {dim('(runner will hot-reload)')}")
                except Exception as e:
                    print(f"  {yellow(f'TorchScript save failed: {e}')}")
                model.train()
        else:
            time.sleep(0.1)

    # ── Shutdown ──────────────────────────────────────────────────────────────
    print(f"\n  {bold('Saving final checkpoint...')}")
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    model.eval()
    try:
        example = torch.zeros(1, SEQ_LEN * STATE_DIM)
        traced  = torch.jit.trace(model, example)
        traced.save(MODEL_PATH)
        print(f"  {green('Final weights saved')} → {MODEL_PATH}")
    except Exception as e:
        print(f"  {yellow(f'TorchScript save failed: {e}')}")

    if hasattr(buffer, 'close'): buffer.close()
    print(f"\n  {bold('Training session summary:')}")
    print(f"  {dim('total updates')}  {cyan(str(total_updates))}")
    print(f"  {dim('total frames')}   {cyan(f'{total_frames:,}')}")
    print(f"  {dim('action space')} {cyan(f'{N_BUTTONS} buttons (Bernoulli) + {N_MEM_BITS} mem bits')}")
    if reward_history:
        print(f"  {dim('final avg reward')} {green(f'{np.mean(reward_history):+.4f}')}")
    print()


if __name__ == "__main__":
    main()
