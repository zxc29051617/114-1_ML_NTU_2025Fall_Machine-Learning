import numpy as np



def l2_distance_squared(vec1, vec2):
    return np.sum((vec1 - vec2)**2)


def trim_mean(arr, proportiontocut=0.1):
    arr_sorted = np.sort(arr)
    n = len(arr_sorted)
    k = int(n * proportiontocut)
    if n - 2*k <= 0:
        return arr.mean()
    return arr_sorted[k : n - k].mean()


# ===========================
# 參數設定
# ===========================

L = 50 # 層數
# 隨機每層維度，範圍 100~50000
d = np.random.randint(100, 50000, size=L).tolist()
N = 8       # task 數量
beta = 0.1  # residual scaling for layer-adaptive


# 模擬 pretrained weights
W_pre = [np.random.uniform(low=-5, high=5, size=di) for di in d]

# 模擬 task-specific weights
W_tasks = []
for i in range(N):
    W_tasks.append([W_pre[l] + np.random.randn(d[l])*0.5 for l in range(L)])

# 計算 task vector τ_i = W_i - W_pre
tau_tasks = []
for i in range(N):
    tau_tasks.append([W_tasks[i][l] - W_pre[l] for l in range(L)])

# ===========================
#  原始 Global EMR + Task-specific masks & rescaler
# ===========================
# ===========================
# Global EMR (全 network)
# ===========================







tau_uni_global = []       # 可以不用分層，先保留 list
M_global_tasks = []
lambda_global_tasks = []

# 1. flatten 每個任務的所有層
tau_tasks_flat = [np.concatenate(tau_tasks[i]) for i in range(N)]
layer_shapes = [tau_tasks[0][l].shape for l in range(L)]  # 保存原始層形狀

# 2. 建全 network 矩陣 D x N
T_global = np.column_stack(tau_tasks_flat)  # shape: (D, N)

# 3. Electing global unified vector
gamma_uni = np.sign(np.sum(T_global, axis=1))                            
epsilon_uni = np.max(np.abs(T_global) * (np.sign(T_global) == gamma_uni[:, None]), axis=1)  
tau_uni_global_flat = gamma_uni * epsilon_uni                             

# 4. Task-specific masks & rescalers
for i in range(N):
    mask = (tau_tasks_flat[i] * tau_uni_global_flat > 0).astype(float)  
    lam = np.sum(np.abs(tau_tasks_flat[i])) / (np.sum(np.abs(mask * tau_uni_global_flat)) + 1e-6)
    M_global_tasks.append(mask)    # 每個 task 一個全 network mask
    lambda_global_tasks.append(lam) # 每個 task 一個 scalar

# 5. 如果需要分層使用，可以切回原始 shape
tau_uni_global = []
start = 0
for shape in layer_shapes:
    end = start + shape[0]
    tau_uni_global.append(tau_uni_global_flat[start:end])
    start = end

# ===========================
# 2 Layer-adaptive EMR
# Layer-wise scalar + soft mask + residual
# ===========================
# ===========================
# Importance-aware soft masks (Fisher-weighted)
# ===========================
# I_l: Fisher information, 這裡先假設全 (每個layer都一樣重要)
I_layer = [np.ones_like(tau_tasks[0][l]) for l in range(L)]  # list of L arrays

alpha = 5
epsilon = 1e-6

M_layer_tasks = []
lambda_layer_tasks = []
tau_uni_layer = []




for l in range(L):
    # ---------- SVD elect ----------
    T_l = np.column_stack([tau_tasks[i][l] for i in range(N)])
    U, S, Vt = np.linalg.svd(T_l, full_matrices=False)
    r = 1
    U_r = U[:, :r]
    z_tasks = [U_r.T @ tau_tasks[i][l] for i in range(N)]
    z_uni = np.sign(sum(z_tasks)) * max([abs(z) for z in z_tasks])
    tau_uni = (U_r * z_uni).flatten()
    tau_uni_layer.append(tau_uni)
    
    # ---------- Soft mask & layer scalar ----------
    M_i, lambda_i = [], []
    I_l_vec = I_layer[l]  # Fisher information vector
    for i in range(N):
        tau_i = tau_tasks[i][l]  # 每層每個 task 的更新
        # Fisher-weighted similarity
        s = (I_l_vec * tau_i * tau_uni) / (np.sqrt( tau_i**2) * np.sqrt( tau_uni**2) + epsilon)
        mask = 1 / (1 + np.exp(-alpha * s))  # sigmoid
        M_i.append(mask)

        tau_uni = tau_uni_layer[l]
        
        lam = trim_mean(np.abs(tau_i), proportiontocut=0.1) / (trim_mean(np.abs(tau_uni), proportiontocut=0.1) + epsilon)
        lambda_i.append(lam)  
    M_layer_tasks.append(M_i)

    lambda_layer_tasks.append(lambda_i)



# 初始化累加
dist_global_to_uni = 0      
dist_layer_to_uni = 0
dist_global_to_recon = 0    
dist_layer_to_recon = 0



# tau_uni_global_flat: unified vector for global EMR
# tau_i_flat: flattened task-specific vectors
# tau_hat_flat: reconstructed task-specific vectors



# ---------- Global EMR ----------
for i in range(N):
    tau_i_flat = tau_tasks_flat[i]                # 全 network vector (original)
    mask = M_global_tasks[i]                      # 全 network mask
    lam = lambda_global_tasks[i]                  # scalar
    tau_hat_flat = lam * mask * tau_uni_global_flat

    dist_global_to_uni += l2_distance_squared(tau_i_flat, tau_uni_global_flat)
    dist_global_to_recon += l2_distance_squared(tau_i_flat, tau_hat_flat)


# tau_uni_layer : unified vector per layer
# tau_i : task-specific per layer
# tau_layer_hat : reconstructed per layer


# ---------- Layer-adaptive EMR ----------
for i in range(N):
    for l in range(L):
        tau_i = tau_tasks[i][l]
        tau_layer_hat = lambda_layer_tasks[l][i] * M_layer_tasks[l][i] * tau_uni_layer[l]
        residual = tau_i - tau_uni_layer[l]
        tau_layer_hat += beta * residual

        dist_layer_to_uni += l2_distance_squared(tau_i, tau_uni_layer[l])
        dist_layer_to_recon += l2_distance_squared(tau_i, tau_layer_hat)


# 平均距離平方
dist_global_to_uni /= N
dist_global_to_recon /= N
dist_layer_to_uni /= N
dist_layer_to_recon /= N

print("===== L2 Squared Distances Comparison =====")
print(f"Global EMR -> to tau_uni: {dist_global_to_uni:.4f}, to reconstructed tau_hat: {dist_global_to_recon:.4f}")
print(f"Layer-adaptive EMR -> to tau_uni: {dist_layer_to_uni:.4f}, to reconstructed tau_hat: {dist_layer_to_recon:.4f}")
