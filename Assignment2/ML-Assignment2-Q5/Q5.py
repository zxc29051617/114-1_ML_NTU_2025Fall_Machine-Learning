# Q5.py — robust id parsing + candidate centroid (+ optional TTA)
import os, re, argparse, torch, pandas as pd, torch.nn.functional as F
import train as tr

PB_RE = re.compile(r"PB\[(.*?)\]")
PW_RE = re.compile(r"PW\[(.*?)\]")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--test_dir",   type=str, required=True)
    p.add_argument("--output_csv", type=str, default="submission.csv")
    p.add_argument("--n_frames",   type=int, default=10)
    p.add_argument("--move_stride",type=int, default=4)
    p.add_argument("--device",     type=str, default="cpu", choices=["cpu","cuda"])
    p.add_argument("--board_size", type=int, default=None)
    p.add_argument("--tta",        type=int, default=1, choices=[1,2,4,8])  # 可選
    return p.parse_args()

def ensure_dir(path, name):
    if not os.path.isdir(path):
        raise FileNotFoundError(f"{name} not found: {path}")

def list_sgfs(folder):
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(".sgf")])

# === 關鍵：正確抓 player 的數字 ===
def parse_player_id(stem: str) -> int:
    # 1) 優先抓 'player(\d+)' 這一段
    m = re.search(r"player\s*0*([0-9]+)", stem, re.IGNORECASE)
    if m: 
        return int(m.group(1))
    # 2) 退而求其次：抓所有數字串，挑「最長的」；長度相同取第一個
    nums = re.findall(r"[0-9]+", stem)
    if nums:
        nums.sort(key=lambda s: (-len(s), s))  # 最長優先
        return int(nums[0])
    raise ValueError(f"Cannot parse player id from stem='{stem}'")

# ---------- 8-way symmetry on (sign,x,y) ----------
def apply_sym(seq, B, kind):
    s = seq.clone()
    x, y = s[:,1], s[:,2]
    if   kind == 0: xn, yn = x, y
    elif kind == 1: xn, yn = B-1-x, y
    elif kind == 2: xn, yn = x, B-1-y
    elif kind == 3: xn, yn = B-1-x, B-1-y
    elif kind == 4: xn, yn = y, x
    elif kind == 5: xn, yn = B-1-y, x
    elif kind == 6: xn, yn = y, B-1-x
    else:            xn, yn = B-1-y, B-1-x
    s[:,1], s[:,2] = xn, yn
    return s

@torch.no_grad()
def embed_seq(model, seq_t, device, B, tta=1):
    if tta == 1:
        z = model(seq_t.unsqueeze(0).to(device))
        return F.normalize(z, p=2, dim=-1).cpu()
    kinds = {1:[0], 2:[0,3], 4:[0,1,2,3], 8:list(range(8))}[tta]
    zs = []
    for k in kinds:
        zs.append(model(apply_sym(seq_t, B, k).unsqueeze(0).to(device)))
    z = torch.stack(zs, dim=0).mean(0)
    return F.normalize(z, p=2, dim=-1).cpu()

def _read_text(path):
    try:
        return open(path, 'r', encoding='utf-8').read()
    except UnicodeDecodeError:
        return open(path, 'r', encoding='latin-1').read()

def _roles_for_player_in_file(fpath, player_token: str):
    """
    回傳該檔內 player_token 出現在哪些角色：集合 {'B','W'} 的子集。
    以精確等號比對（不做模糊包含），大小寫不敏感、去前後空白。
    """
    text = _read_text(fpath)
    want = player_token.strip().lower()
    roles = set()
    for m in PB_RE.findall(text):
        if m.strip().lower() == want:
            roles.add('B')
    for m in PW_RE.findall(text):
        if m.strip().lower() == want:
            roles.add('W')
    return roles


@torch.no_grad()
def embed_one_sgf(model, loader, fpath, device, board_size, tta, player_token=None):
    """
    只用玩家實際扮演的顏色做嵌入；若同檔同玩家兩色都有則平均。
    同時支援 TTA（棋盤 8 對稱）。
    推論時一律 rand_offset=False，避免每次抽不同片段導致預測不穩。
    """
    roles = set()
    if player_token is not None:
        roles = _roles_for_player_in_file(fpath, player_token)

    vecs = []

    # 如果我們能判斷他是黑或白，就只取那幾種角色的片段
    if 'B' in roles:
        xb = loader.load_player_view(fpath, "B", rand_offset=False)  # <- 固定起點
        vecs.append(embed_seq(model, xb, device, board_size, tta))
    if 'W' in roles:
        xw = loader.load_player_view(fpath, "W", rand_offset=False)  # <- 固定起點
        vecs.append(embed_seq(model, xw, device, board_size, tta))

    if vecs:
        z = torch.stack(vecs, dim=0).mean(0)
        return F.normalize(z, p=2, dim=-1).cpu()

    # 如果完全判斷不出顏色（很少但有可能），就 B/W 都嵌再平均
    xb = loader.load_player_view(fpath, "B", rand_offset=False)
    xw = loader.load_player_view(fpath, "W", rand_offset=False)
    zb = embed_seq(model, xb, device, board_size, tta)
    zw = embed_seq(model, xw, device, board_size, tta)
    return F.normalize((zb + zw) / 2.0, p=2, dim=-1).cpu()



def build_cand_centroids(model, loader, cand_dir, device, board_size, tta):
    bag = {}
    files = list_sgfs(cand_dir)
    for i, fname in enumerate(files, 1):
        fpath = os.path.join(cand_dir, fname)
        stem  = os.path.splitext(fname)[0]      # e.g., "player001"
        pid   = parse_player_id(stem)           # 整數 id
        try:
            z = embed_one_sgf(model, loader, fpath, device, board_size, tta, player_token=stem)  # <-- 關鍵
            bag.setdefault(pid, []).append(z)
        except Exception as e:
            print(f"[warn] skip {fname}: {e}")
        if i % 50 == 0 or i == len(files):
            print(f"[cand] {i}/{len(files)} done")
    if not bag:
        raise RuntimeError("No candidate embeddings.")
    ids, mats = [], []
    for pid, zs in sorted(bag.items()):
        Z = torch.cat(zs, dim=0).mean(0, keepdim=True)     # centroid [1,D]
        ids.append(pid); mats.append(Z)
    return ids, torch.cat(mats, dim=0)                     # ([C], [C,D])


def build_query_embeddings(model, loader, query_dir, device, board_size, tta):
    rows, mats = [], []
    files = list_sgfs(query_dir)
    for i, fname in enumerate(files, 1):
        fpath = os.path.join(query_dir, fname)
        stem  = os.path.splitext(fname)[0]
        qid   = parse_player_id(stem)
        try:
            z = embed_one_sgf(model, loader, fpath, device, board_size, tta, player_token=stem)  # <-- 關鍵
            rows.append(qid); mats.append(z)
        except Exception as e:
            print(f"[warn] skip {fname}: {e}")
        if i % 50 == 0 or i == len(files):
            print(f"[query] {i}/{len(files)} done")
    if not rows:
        raise RuntimeError("No query embeddings.")
    order = sorted(range(len(rows)), key=lambda i: rows[i])
    rows = [rows[i] for i in order]
    mats = torch.cat([mats[i] for i in order], dim=0)
    return rows, mats


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    cand_dir  = os.path.join(args.test_dir, "cand_set")
    query_dir = os.path.join(args.test_dir, "query_set")
    ensure_dir(args.test_dir, "test_dir"); ensure_dir(cand_dir, "cand_set"); ensure_dir(query_dir, "query_set")

    # load ckpt & align dims/heads
    ckpt = torch.load(args.model_path, map_location="cpu")
    cfgs = ckpt.get("cfg_snapshot", {})
    tr.MOVE_DIM   = int(cfgs.get("move_dim", tr.MOVE_DIM))
    tr.NUM_LAYERS = int(cfgs.get("num_layers", cfgs.get("nn_num_blocks", tr.NUM_LAYERS)))
    tr.NUM_HEADS  = int(cfgs.get("num_heads", tr.NUM_HEADS))
    tr.FFN_DIM    = int(cfgs.get("ffn_dim", tr.FFN_DIM))
    tr.EMB_DIM    = int(cfgs.get("emb_dim", cfgs.get("nn_num_hidden_channels", tr.EMB_DIM)))
    enable_dc   = str(cfgs.get("deepcluster_enable", "false")).lower() in ("1","true","yes","y")
    enable_open = str(cfgs.get("opening_aux_enable", "false")).lower() in ("1","true","yes","y")
    opening_k   = int(cfgs.get("opening_K", 5))
    dc_k        = int(cfgs.get("deepcluster_k", 10000))
    board_size  = args.board_size or int(cfgs.get("env_board_size", 19))

    model = tr.GameNet(board_size=board_size, dc_k=dc_k,
                       enable_dc=enable_dc, enable_open=enable_open,
                       opening_k=opening_k).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    loader = tr.SGFLoader(n_frames=args.n_frames, move_stride=args.move_stride, board_size=board_size)

    # --- 重點：候選人先做「玩家 centroid」 ---
    cand_ids, cand_mat = build_cand_centroids(model, loader, cand_dir, device, board_size, args.tta)  # [C], [C,D]
    query_ids, query_mat = build_query_embeddings(model, loader, query_dir, device, board_size, args.tta)  # [Q], [Q,D]

    print("[info] Computing cosine similarity ...")
    sims = torch.matmul(query_mat, cand_mat.T)  # [Q, C]
    best = sims.argmax(dim=1)

    rows = [{"id": int(qid), "label": int(cand_ids[best[i]])} for i, qid in enumerate(query_ids)]
    pd.DataFrame(rows, columns=["id","label"]).to_csv(args.output_csv, index=False, encoding="utf-8")
    print(f"[info] Saved to {args.output_csv}")

if __name__ == "__main__":
    main()
