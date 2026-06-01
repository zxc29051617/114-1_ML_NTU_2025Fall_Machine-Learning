import os
import re
import time
import math
import random
import unicodedata
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

try:
    from sklearn.cluster import KMeans
    _SKLEARN_OK = True
except Exception:
    _SKLEARN_OK = False


def load_flat_cfg(path):
    cfg = {}
    if not os.path.exists(path):
        return cfg
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            k, v = line.split('=', 1)
            cfg[k.strip()] = v.strip()
    return cfg

cfg = load_flat_cfg('conf.cfg')

def get_str(key, default=""):
    return cfg.get(key, str(default))

def get_int(key, default=0):
    try:
        return int(cfg.get(key, default))
    except:
        return int(default)

def get_float(key, default=0.0):
    try:
        return float(cfg.get(key, default))
    except:
        return float(default)

def get_bool(key, default=False):
    v = cfg.get(key, None)
    if v is None:
        return default
    return str(v).lower() in ("1", "true", "yes", "y")

# Global Variables

AUTO_SEED = get_bool('program_auto_seed', False)
SEED = (int.from_bytes(os.urandom(4), 'little') if AUTO_SEED else get_int('program_seed', 0))
QUIET = get_bool('program_quiet', False)

torch.set_num_threads(max(1, get_int('learner_num_thread', get_int('zero_num_threads', 4))))

DATA_DIR = get_str('zero_training_directory', '')  # 存放 SGF 棋譜檔案的資料夾路徑
if DATA_DIR == '':
    DATA_DIR = './train_set/'
    
BOARD_SIZE = get_int('env_board_size', 19)              # 棋盤大小（預設 19）
N = get_int('players_per_batch', 20)                    # 每批次（episode）要取樣的玩家數量
M = get_int('games_per_player', 9)                      # 每個玩家要取樣的棋譜數量
N_FRAMES = get_int('n_frames', 10)                      # 棋譜序列的固定長度（時間步），會對棋譜進行截斷或補齊
MOVE_STRIDE = get_int('move_step_to_choose', 1)         # 步距，代表每隔幾手棋才取樣一手（例如 1 代表取樣所有手）

# Learner (use your keys)
LR = get_float('learner_learning_rate', 3e-4)
EPISODES_PER_EPOCH = get_int('learner_training_step', 500)   # episodes each epoch
LOG_EVERY = max(1, get_int('learner_training_display_step', 100))
MOMENTUM = get_float('learner_momentum', 0.9)                # reserved if you switch to SGD
WEIGHT_DECAY = get_float('learner_weight_decay', 0.0)
BATCH_SIZE_TARGET = max(1, get_int('learner_batch_size', N*M))  # for grad accumulation
EPOCHS = get_int('epochs', 1)  # not in your conf; default 1
GRAD_CLIP = get_float('grad_clip', 1.0)

# 模型維度（可視需求調整/擴充）
MOVE_DIM = get_int('move_dim', 320)
NUM_LAYERS = get_int('num_layers', get_int('nn_num_blocks', 6))
NUM_HEADS  = get_int('num_heads', 8)
FFN_DIM    = get_int('ffn_dim', 1024)
EMB_DIM    = get_int('emb_dim', get_int('nn_num_hidden_channels', 512))

DC_ENABLE         = get_bool('deepcluster_enable', False)
DC_K              = get_int('deepcluster_k', 10000)
DC_REFRESH_EVERY  = get_int('deepcluster_refresh_steps', 200)
DC_SAMPLES        = get_int('deepcluster_samples', 4000)
LOSS_ALPHA_DC     = get_float('loss_alpha_dc', 1.0)

OPENING_ENABLE    = get_bool('opening_aux_enable', False)
OPENING_K         = get_int('opening_K', 5)
LOSS_GAMMA_OPEN   = get_float('loss_gamma_open', 0.0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

MOVE_RE = re.compile(r";([BW])\[([a-s]{2})\]")
PB_RE = re.compile(r"PB\[(.*?)\]")
PW_RE = re.compile(r"PW\[(.*?)\]")

def safe_name(s: str) -> str:
    return unicodedata.normalize('NFKC', s).strip()

# 建立一個「玩家名稱」到「他的所有棋譜」的索引。
class PlayerIndex:
    """ player → list[(path, role)]，role ∈ {'B','W'} """
    def __init__(self, data_dir, min_games=1):
        self.index = defaultdict(list)
        self.players = []
        self.all_items = []
        self._build(data_dir, min_games)

    def _build(self, data_dir, min_games):
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        files = [f for f in os.listdir(data_dir) if f.endswith('.sgf')]
        if not files:
            raise RuntimeError(f"No .sgf files under {data_dir}")

        for fname in files:
            path = os.path.join(data_dir, fname)
            try:
                text = open(path, 'r', encoding='utf-8').read()
            except UnicodeDecodeError:
                text = open(path, 'r', encoding='latin-1').read()

            pb = PB_RE.search(text)
            pw = PW_RE.search(text)
            if pb:
                pbn = safe_name(pb.group(1))
                if pbn:
                    self.index[pbn].append((path, 'B'))
            if pw:
                pwn = safe_name(pw.group(1))
                if pwn:
                    self.index[pwn].append((path, 'W'))

        self.players = [p for p, lst in self.index.items() if len(lst) >= min_games]
        self.players.sort()
        for p in self.players:
            for item in self.index[p]:
                self.all_items.append(item)

    def sample_players(self, n):
        if len(self.players) < n:
            raise RuntimeError(f"Need {n} players, but only {len(self.players)} available.")
        return random.sample(self.players, n)

    def sample_games(self, player, m):
        lst = self.index[player]
        if len(lst) >= m:
            return random.sample(lst, m)
        return [random.choice(lst) for _ in range(m)]

    def sample_unlabeled_items(self, num):
        if not self.all_items:
            raise RuntimeError("No SGF items indexed.")
        if num <= len(self.all_items):
            return random.sample(self.all_items, num)
        return [random.choice(self.all_items) for _ in range(num)]
    
# 負責將 SGF 檔案轉換為模型所需的 [T, 3] 格式的 Tensor。
class SGFLoader:
    def __init__(self, n_frames=N_FRAMES, move_stride=MOVE_STRIDE, board_size=BOARD_SIZE):
        self.n_frames = n_frames
        self.move_stride = max(1, move_stride)
        self.board_size = board_size
        self.max_coord = ord('a') + board_size - 1

    @staticmethod
    def parse_moves(text):
        return MOVE_RE.findall(text)

    def _read_text(self, path):
        try:
            return open(path, 'r', encoding='utf-8').read()
        except UnicodeDecodeError:
            return open(path, 'r', encoding='latin-1').read()

    def load_player_view(self, path, role, rand_offset=True):
        """
        回傳 shape [T,3] 的 tensor:
        [ sign , x , y ]
        - sign = +1.0 代表這一手是這個玩家下的
        - sign = -1.0 代表這一手是對手下的
        - x,y 是落子座標 (0-based)
        我們不再把對手的手抹成0，因為那會丟掉太多資訊。
        我們也隨機決定從哪一手開始抽，避免永遠只看開局。
        """
        text = self._read_text(path)
        moves = self.parse_moves(text)  # list of (color, "xy")
        me_color = 'B' if role == 'B' else 'W'

        span = self.n_frames * self.move_stride
        max_start = max(0, len(moves) - span)
        if rand_offset and max_start > 0:
            start_idx = random.randint(0, max_start)
        else:
            start_idx = 0

        seq = []
        for idx in range(start_idx, len(moves), self.move_stride):
            color, coord = moves[idx]

            if ord(coord[0]) > self.max_coord or ord(coord[1]) > self.max_coord:
                continue

            x = float(ord(coord[0]) - ord('a'))
            y = float(ord(coord[1]) - ord('a'))

            if color == me_color:
                sign = 1.0
            else:
                sign = -1.0

            seq.append([sign, x, y])

            if len(seq) >= self.n_frames:
                break

        while len(seq) < self.n_frames:
            seq.append([0.0, 0.0, 0.0])

        return torch.tensor(seq, dtype=torch.float32)

    def opening_5hot(self, path, K=5):
        B = self.board_size
        dim = K * (B * B)
        vec = torch.zeros(dim, dtype=torch.float32)
        text = self._read_text(path)
        moves = self.parse_moves(text)
        t = 0
        for (color, coord) in moves:
            if ord(coord[0]) > self.max_coord or ord(coord[1]) > self.max_coord:
                continue
            if t >= K:
                break
            x = ord(coord[0]) - ord('a')
            y = ord(coord[1]) - ord('a')
            idx = t * (B * B) + (y * B + x)
            vec[idx] = 1.0
            t += 1
        return vec
    
def apply_sym_batch(seq_batch, B, kind):
    """
    seq_batch: [B,T,3] tensor (sign, x, y)
    B: board_size
    kind: 0..7  (8 symmetries of the square board)
    returns transformed seq_batch (new tensor)
    """
    s = seq_batch.clone()
    x = s[:,:,1]
    y = s[:,:,2]

    if   kind == 0: xn, yn = x, y
    elif kind == 1: xn, yn = (B-1)-x, y
    elif kind == 2: xn, yn = x, (B-1)-y
    elif kind == 3: xn, yn = (B-1)-x, (B-1)-y
    elif kind == 4: xn, yn = y, x
    elif kind == 5: xn, yn = (B-1)-y, x
    elif kind == 6: xn, yn = y, (B-1)-x
    else:           xn, yn = (B-1)-y, (B-1)-x

    s[:,:,1] = xn
    s[:,:,2] = yn
    return s

# MoveEncoder, TransformerGameModel, GameNet 將「棋譜序列」轉換為「向量」的神經網路結構

class MoveEncoder(nn.Module):
    def __init__(self, in_dim=3, out_dim=MOVE_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, out_dim), nn.ReLU()
        )
    def forward(self, x):  # [B,T,3]
        return self.net(x)
    
class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    shape in:  [B, T, D]
    shape out: [B, T, D] (add PE)
    """
    def __init__(self, d_model, max_len=512):
        super().__init__()
        import math
        pe = torch.zeros(max_len, d_model)  # [T, D]
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # [T,1]
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even
        pe[:, 1::2] = torch.cos(position * div_term)  # odd
        pe = pe.unsqueeze(0)  # [1,T,D]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [B, T, D]
        T = x.size(1)
        return x + self.pe[:, :T, :]

class TransformerGameModel(nn.Module):
    def __init__(self, move_dim=MOVE_DIM, emb_dim=EMB_DIM,
                 num_layers=NUM_LAYERS, num_heads=NUM_HEADS, ffn_dim=FFN_DIM,
                 max_len=512, dropout_p=0.1):
        super().__init__()

        enc_layer = nn.TransformerEncoderLayer(
            d_model=move_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout_p,
            batch_first=True,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # <NEW> positional encoding module
        self.posenc = PositionalEncoding(d_model=move_dim, max_len=max_len)

        # projection to final embedding space
        self.proj = nn.Sequential(
            nn.Linear(move_dim, emb_dim),
            nn.Tanh()
        )

    def forward(self, x):  # x: [B,T,move_dim]
        # add positional information BEFORE feeding encoder
        h = self.posenc(x)
        h = self.encoder(h)          # [B,T,move_dim]

        # mean-pool across time
        h = h.mean(dim=1)            # [B,move_dim]

        # final projection + L2 normalize
        z = self.proj(h)             # [B,emb_dim]
        z = nn.functional.normalize(z, p=2, dim=-1)
        return z


class GameNet(nn.Module):
    def __init__(self, board_size=BOARD_SIZE, dc_k=10000,
                 enable_dc=False, enable_open=False, opening_k=5):
        super().__init__()
        self.move_enc = MoveEncoder()
        self.seq_enc  = TransformerGameModel()
        self.enable_dc = enable_dc
        self.enable_open = enable_open
        self.dc_k = dc_k
        self.cluster_head = nn.Linear(EMB_DIM, dc_k) if enable_dc else None
        if enable_open:
            self.open_dim = opening_k * (board_size * board_size)
            self.open_head = nn.Linear(EMB_DIM, self.open_dim)
        else:
            self.open_head = None

    def encode(self, x):
        m = self.move_enc(x)
        z = self.seq_enc(m)
        return z

    def forward(self, x):
        return self.encode(x)


# GE2E (Generalized End-to-End) Loss

""" 讓同一個玩家 (N) 的 M 盤棋的嵌入向量
    彼此之間的相似度（Cosine Similarity）盡可能高，並與其他玩家的嵌入向量相似度盡可能低。
"""

class GE2ELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(10.0))
        self.b = nn.Parameter(torch.tensor(-5.0))
        self.softplus = nn.Softplus()

    def forward(self, embeds):  # [N,M,D]
        N, M, D = embeds.shape
        e = nn.functional.normalize(embeds, p=2, dim=-1)
        centroids = e.mean(dim=1)                     # [N,D]
        sum_j = e.sum(dim=1, keepdim=True)            # [N,1,D]
        c_loo = (sum_j - e) / max(1, M - 1)           # [N,M,D]

        e_flat = e.reshape(N*M, D)                    # [NM,D]
        cos_all = torch.matmul(e_flat, centroids.T)   # [NM,N]
        js = torch.arange(N, device=e.device).repeat_interleave(M)  # [NM]
        cos_self = (e_flat * c_loo.reshape(N*M, D)).sum(dim=-1)     # [NM]
        cos_all[torch.arange(N*M, device=e.device), js] = cos_self

        w = self.softplus(self.w) + 1e-6
        b = self.b
        S = w * cos_all + b                            # [NM,N]

        loss = nn.CrossEntropyLoss()(S, js)
        with torch.no_grad():
            acc = (S.argmax(dim=1) == js).float().mean().item()
        return loss, acc


# ===================== (Optional) DeepCluster utils =====================
class DeepClusterModule:
    def __init__(self, k=10000, use_cosine=True):
        self.k = k
        self.use_cosine = use_cosine
        self.centroids = None

    @torch.no_grad()
    def refresh(self, model, pidx, loader, samples=4000, batch_size=512, device='cpu'):
        items = pidx.sample_unlabeled_items(samples)
        seqs = [loader.load_player_view(path, role) for (path, role) in items]
        X = torch.stack(seqs, dim=0)  # [S,T,3]

        Z = []
        model.eval()
        for i in range(0, X.size(0), batch_size):
            xb = X[i:i+batch_size].to(device)
            zb = model.encode(xb)
            Z.append(zb.cpu())
        Z = torch.cat(Z, dim=0).numpy()

        k_eff = min(self.k, Z.shape[0])
        if _SKLEARN_OK and k_eff >= 2:
            km = KMeans(n_clusters=k_eff, n_init='auto', max_iter=100, random_state=0)
            km.fit(Z)
            C = torch.tensor(km.cluster_centers_, dtype=torch.float32)
        else:
            Zt = torch.tensor(Z, dtype=torch.float32)
            idx = torch.randperm(Zt.size(0))[:k_eff]
            C = Zt[idx].clone()
            for _ in range(20):
                dist = pairwise_cosine(-Zt, -C) if self.use_cosine else pairwise_euclidean2(Zt, C)
                a = dist.argmin(dim=1)
                for j in range(k_eff):
                    mask = (a == j)
                    if mask.any():
                        C[j] = Zt[mask].mean(dim=0)
        C = nn.functional.normalize(C, p=2, dim=-1)
        self.centroids = C.to(device)

    @torch.no_grad()
    def assign(self, z):
        assert self.centroids is not None, "Centroids not initialized."
        C = self.centroids
        z_n = nn.functional.normalize(z, p=2, dim=-1)
        sims = torch.matmul(z_n, C.T)
        y = sims.argmax(dim=1)
        return y

def pairwise_euclidean2(X, C):
    x2 = (X*X).sum(dim=1, keepdim=True)
    c2 = (C*C).sum(dim=1, keepdim=True).T
    return x2 + c2 - 2.0 * (X @ C.T)

def pairwise_cosine(Xn, Cn):
    Xn = nn.functional.normalize(Xn, p=2, dim=-1)
    Cn = nn.functional.normalize(Cn, p=2, dim=-1)
    return 1.0 - (Xn @ Cn.T)

# Training Loop

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train():
    set_seed(SEED)

    pidx = PlayerIndex(DATA_DIR, min_games=1)
    if len(pidx.players) < N:
        raise RuntimeError(f"玩家數不足：需要 N={N}，目前 {len(pidx.players)}。")

    loader = SGFLoader(n_frames=N_FRAMES, move_stride=MOVE_STRIDE, board_size=BOARD_SIZE)

    model = GameNet(board_size=BOARD_SIZE, dc_k=DC_K,
                    enable_dc=DC_ENABLE, enable_open=OPENING_ENABLE,
                    opening_k=OPENING_K).to(device)
    ge2e = GE2ELoss().to(device)

    main_params = list(model.parameters())
    ge2e_params = list(ge2e.parameters())

    optim_all = optim.AdamW([
        {"params": main_params, "lr": min(LR, 1e-3), "weight_decay": WEIGHT_DECAY},
        {"params": ge2e_params, "lr": min(LR, 5e-4), "weight_decay": 0.0},
    ])

    # grad accumulation: make effective batch ~= learner_batch_size
    per_episode_bs = N * M
    grad_acc_steps = max(1, math.ceil(BATCH_SIZE_TARGET / per_episode_bs))

    # optional DeepCluster
    dc = DeepClusterModule(k=DC_K, use_cosine=True) if DC_ENABLE else None
    if DC_ENABLE:
        print("[DeepCluster] Initializing centroids...")
        dc.refresh(model, pidx, loader, samples=DC_SAMPLES, device=device)
        print(f"[DeepCluster] Centroids ready: k = {dc.centroids.size(0)}")

    bce = nn.BCEWithLogitsLoss() if OPENING_ENABLE else None
    ce  = nn.CrossEntropyLoss()    if DC_ENABLE else None

    global_step = 0
    for epoch in range(1, EPOCHS + 1):
        running = defaultdict(float)
        seen = 0
        pbar = tqdm(range(EPISODES_PER_EPOCH), desc=f"Epoch {epoch}/{EPOCHS}", disable=QUIET)
        optim_all.zero_grad()

        for ep in pbar:
            # --- build one GE2E episode: N persons × M games ---
            players = pidx.sample_players(N)
            buckets = [[] for _ in range(N)]
            open_feats = []
            for j, p in enumerate(players):
                games = pidx.sample_games(p, M)
                for (path, role) in games:
                    seq = loader.load_player_view(path, role, rand_offset=True)
                    buckets[j].append(seq)
                    if OPENING_ENABLE:
                        open_feats.append(loader.opening_5hot(path, OPENING_K))

            for j in range(N):
                if len(buckets[j]) < M:
                    buckets[j] += [buckets[j][-1].clone()] * (M - len(buckets[j]))

            batch = torch.stack([torch.stack(buckets[j][:M], dim=0) for j in range(N)], dim=0)

            x = batch.view(N*M, N_FRAMES, 3)  # [N*M, T, 3]

            # === NEW: random board symmetry augmentation ===
            # 讓模型在訓練時就看到旋轉/翻轉後的座標，學到對稱不變性
            sym_kind = random.randint(0,7)
            x = apply_sym_batch(x, BOARD_SIZE, sym_kind)

            x = x.to(device)

            if OPENING_ENABLE:
                open_vec = torch.stack(open_feats[:N*M], dim=0).to(device)

            # --- forward ---
            z = model(x).view(N, M, -1)
            loss_ge2e, acc_ge2e = ge2e(z)

            # DeepCluster pseudo-label head
            loss_dc = torch.tensor(0.0, device=device)
            acc_dc  = torch.tensor(0.0, device=device)
            if DC_ENABLE and dc.centroids is not None and model.cluster_head is not None and LOSS_ALPHA_DC > 0:
                with torch.no_grad():
                    z_flat = z.reshape(N*M, -1)
                    y_pseudo = dc.assign(z_flat)
                logits_dc = model.cluster_head(z_flat)
                loss_dc = ce(logits_dc, y_pseudo)
                with torch.no_grad():
                    pred_dc = logits_dc.argmax(dim=1)
                    acc_dc = (pred_dc == y_pseudo).float().mean()

            # Opening aux head
            loss_open = torch.tensor(0.0, device=device)
            if OPENING_ENABLE and model.open_head is not None and LOSS_GAMMA_OPEN > 0:
                z_flat = z.reshape(N*M, -1)
                logits_open = model.open_head(z_flat)
                loss_open = bce(logits_open, open_vec)

            # total loss (scale by grad_acc_steps for proper averaging)
            loss = (loss_ge2e + LOSS_ALPHA_DC * loss_dc + LOSS_GAMMA_OPEN * loss_open) / grad_acc_steps
            loss.backward()

            # grad step every grad_acc_steps episodes
            if GRAD_CLIP and GRAD_CLIP > 0:
                nn.utils.clip_grad_norm_(main_params + ge2e_params, GRAD_CLIP)
            optim_all.step()
            optim_all.zero_grad()

            # logging
            running['loss'] += float(loss.item()) * grad_acc_steps
            running['ge2e_loss'] += float(loss_ge2e.item())
            running['ge2e_acc']  += float(acc_ge2e)
            if DC_ENABLE:
                running['dc_loss'] += float(loss_dc.item())
                running['dc_acc']  += float(acc_dc.item())
            if OPENING_ENABLE:
                running['open_loss'] += float(loss_open.item())

            seen += 1
            global_step += 1

            if not QUIET and (ep + 1) % LOG_EVERY == 0:
                msg = f"ep={ep+1} | loss={running['loss']/seen:.4f} | GE2E(L/Acc)={running['ge2e_loss']/seen:.4f}/{running['ge2e_acc']/seen:.3f}"
                if DC_ENABLE:
                    msg += f" | DC(L/Acc)={running['dc_loss']/seen:.4f}/{running['dc_acc']/seen:.3f}"
                if OPENING_ENABLE:
                    msg += f" | OPEN(L)={running['open_loss']/seen:.4f}"
                pbar.set_postfix_str(msg)

            # periodic DeepCluster refresh
            if DC_ENABLE and (global_step % DC_REFRESH_EVERY == 0):
                dc.refresh(model, pidx, loader, samples=DC_SAMPLES, device=device)

        # checkpoint
        os.makedirs('weights', exist_ok=True)
        torch.save({
            'model': model.state_dict(),
            'ge2e': ge2e.state_dict(),
            'centroids': (dc.centroids if DC_ENABLE else None),
            'cfg_snapshot': cfg
        }, os.path.join('weights', f'combined_epoch{epoch}.pt'))

        if not QUIET:
            msg = f"Epoch {epoch} | loss={running['loss']/seen:.4f} | GE2E(L/Acc)={running['ge2e_loss']/seen:.4f}/{running['ge2e_acc']/seen:.3f}"
            if DC_ENABLE:
                msg += f" | DC(L/Acc)={running['dc_loss']/seen:.4f}/{running['dc_acc']/seen:.3f}"
            if OPENING_ENABLE:
                msg += f" | OPEN(L)={running['open_loss']/seen:.4f}"
            print(msg)

    print("Training finished. Checkpoints -> ./weights/")

if __name__ == "__main__":
    train()
