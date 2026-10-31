#Hierarchical + Layer-wise Trimmed + 8tasks
import os
import sys
import shutil
import random
import scipy.io
from google.colab import drive

print("初始化環境")
drive.mount('/content/drive')

PROJECT_PATH = '/content/drive/MyDrive/NTU_ML_FINALPROJECT/EMR_MERGING/vision'
LOCAL_DATA_ROOT = '/content/datasets'
LOCAL_CACHE = '/content/local_cache'
os.makedirs(LOCAL_DATA_ROOT, exist_ok=True)
os.makedirs(LOCAL_CACHE, exist_ok=True)

# 插件安裝
os.system("pip uninstall -y datasets > /dev/null 2>&1")
os.system("pip uninstall -y open_clip_torch > /dev/null 2>&1")
os.system("pip install open_clip_torch==2.20.0 -q")
os.system("pip install git+https://github.com/mlfoundations/task_vectors.git -q")
os.system("pip install scipy scikit-learn -q")

if not os.path.exists(PROJECT_PATH):
    print("找不到 PROJECT_PATH")
    sys.exit(1)

os.chdir(PROJECT_PATH)
if PROJECT_PATH not in sys.path:
    sys.path.insert(0, PROJECT_PATH)

DATASETS_DIR = os.path.join(PROJECT_PATH, "datasets")



# 輔助函數
def ensure_symlink(src, dst):
    if not os.path.exists(src):
        print(f"來源不存在: {src}")
        return False
    if os.path.islink(dst):
        os.unlink(dst)
    elif os.path.exists(dst):
        return True
    os.symlink(src, dst)
    print(f"Link: {os.path.basename(src)}")
    return True

def copy_to_local(src_dir, dst_dir, desc=""):
    if os.path.exists(dst_dir):
        file_count = sum(len(files) for _, _, files in os.walk(dst_dir))
        if file_count > 10:
            print(f"{desc} 本地快取已存在 ({file_count} 檔案)")
            return dst_dir
    if not os.path.exists(src_dir):
        print(f"來源不存在: {src_dir}")
        return None
    print(f"複製 {desc} 到本地快取...")
    try:
        if os.path.exists(dst_dir): shutil.rmtree(dst_dir)
        shutil.copytree(src_dir, dst_dir)
        print(f"複製完成")
        return dst_dir
    except Exception as e:
        print(f"複製失敗: {e}")
        return None


# GTSRB
print("\nGTSRB")
gtsrb_local = copy_to_local(os.path.join(DATASETS_DIR, "gtsrb"), os.path.join(LOCAL_CACHE, "gtsrb"), "GTSRB")
if gtsrb_local: ensure_symlink(gtsrb_local, os.path.join(LOCAL_DATA_ROOT, "gtsrb"))
else: ensure_symlink(os.path.join(DATASETS_DIR, "gtsrb"), os.path.join(LOCAL_DATA_ROOT, "gtsrb"))


# RESISC45
print("\nRESISC45")
local_resisc_root = os.path.join(LOCAL_DATA_ROOT, "RESISC45")
os.makedirs(local_resisc_root, exist_ok=True)

src_resisc = os.path.join(DATASETS_DIR, "RESISC45", "NWPU-RESISC45")
if not os.path.exists(src_resisc): src_resisc = os.path.join(DATASETS_DIR, "NWPU-RESISC45")

dst_resisc = os.path.join(LOCAL_CACHE, "NWPU-RESISC45")
resisc_local = copy_to_local(src_resisc, dst_resisc, "RESISC45")
if resisc_local: ensure_symlink(resisc_local, os.path.join(local_resisc_root, "NWPU-RESISC45"))
else: ensure_symlink(src_resisc, os.path.join(local_resisc_root, "NWPU-RESISC45"))

img_root = os.path.join(local_resisc_root, "NWPU-RESISC45")
if os.path.isdir(img_root) and not os.path.exists(os.path.join(local_resisc_root, "resisc45-train.txt")):
    print("生成RESISC45 splits...")
    random.seed(42)
    classes = sorted([d for d in os.listdir(img_root) if os.path.isdir(os.path.join(img_root, d))])
    train_l, val_l, test_l = [], [], []
    for cls in classes:
        files = [f for f in os.listdir(os.path.join(img_root, cls)) if f.lower().endswith(".jpg")]
        random.shuffle(files)
        n = len(files)
        n1, n2 = int(0.6 * n), int(0.8 * n)
        train_l.extend([f"{cls}/{x}" for x in files[:n1]])
        val_l.extend([f"{cls}/{x}" for x in files[n1:n2]])
        test_l.extend([f"{cls}/{x}" for x in files[n2:]])
    for n, l in [("train", train_l), ("val", val_l), ("test", test_l)]:
        with open(os.path.join(local_resisc_root, f"resisc45-{n}.txt"), "w") as f: f.write("\n".join(l))
    print(f"Splits: Train:{len(train_l)} Test:{len(test_l)}")

# Cars
print("\nCars")
local_cars_root = os.path.join(LOCAL_DATA_ROOT, "stanford_cars")
os.makedirs(local_cars_root, exist_ok=True)
drive_cars_base = os.path.join(DATASETS_DIR, "stanford_cars")
if not os.path.exists(drive_cars_base): drive_cars_base = DATASETS_DIR

for folder in ["cars_train", "cars_test", "car_devkit"]:
    src = os.path.join(drive_cars_base, folder)
    if folder == "car_devkit" and not os.path.exists(src): src = os.path.join(drive_cars_base, "devkit")
    target_name = "devkit" if "devkit" in folder else folder
    dst_cache = os.path.join(LOCAL_CACHE, target_name)
    if copy_to_local(src, dst_cache, f"Cars/{folder}"):
        ensure_symlink(dst_cache, os.path.join(local_cars_root, target_name))
    else:
        ensure_symlink(src, os.path.join(local_cars_root, target_name))

# 複製 cars_test_annos_withlabels
withlabels_candidates = [
    os.path.join(drive_cars_base, "cars_test_annos_withlabels.mat"),
    os.path.join(drive_cars_base, "cars_test_annos_withlabels"),
    os.path.join(DATASETS_DIR, "cars_test_annos_withlabels.mat"),
    os.path.join(DATASETS_DIR, "cars_test_annos_withlabels"),
]
withlabels_src = None
for candidate in withlabels_candidates:
    if os.path.exists(candidate):
        withlabels_src = candidate
        break

devkit_dst = os.path.join(local_cars_root, "devkit")
if withlabels_src:
    dst_path = os.path.join(devkit_dst, "cars_test_annos_withlabels.mat")
    if not os.path.exists(dst_path):
        shutil.copy2(withlabels_src, dst_path)
        print(f"複製 cars_test_annos_withlabels.mat")
else:
    print("找不到 cars_test_annos_withlabels")

def setup_cars_index_official():
    train_idx = os.path.join(local_cars_root, "train_index.txt")
    test_idx = os.path.join(local_cars_root, "test_index.txt")
    
    if os.path.exists(train_idx) and os.path.exists(test_idx):
        with open(test_idx) as f:
            test_count = len(f.readlines())
        if test_count > 8000:
            print(f"Cars index已存在 (測試: {test_count})")
            return

    print("建立 Cars index")
    meta_path = None
    for p in [os.path.join(local_cars_root, "devkit", "cars_meta.mat"),
              os.path.join(local_cars_root, "devkit", "devkit", "cars_meta.mat")]:
        if os.path.exists(p): meta_path = p; break

    train_annos_path = None
    for p in [os.path.join(local_cars_root, "devkit", "cars_train_annos.mat"),
              os.path.join(local_cars_root, "devkit", "devkit", "cars_train_annos.mat")]:
        if os.path.exists(p): train_annos_path = p; break

    test_annos_path = None
    for p in [os.path.join(local_cars_root, "devkit", "cars_test_annos_withlabels.mat"),
              os.path.join(local_cars_root, "devkit", "devkit", "cars_test_annos_withlabels.mat")]:
        if os.path.exists(p): test_annos_path = p; break

    if not meta_path: 
        print("找不到 cars_meta.mat")
        return

    meta = scipy.io.loadmat(meta_path)
    class_names = [str(c[0]) for c in meta['class_names'][0]]
    with open(os.path.join(local_cars_root, "classes.txt"), "w") as f:
        for i, name in enumerate(class_names): f.write(f"{i}\t{name}\n")
    print(f"類別數: {len(class_names)}")

    train_img_dir = os.path.join(local_cars_root, "cars_train")
    if train_annos_path:
        annos = scipy.io.loadmat(train_annos_path)['annotations'][0]
        tr_lines = []
        for a in annos:
            try:
                fname, cidx = str(a['fname'][0]), int(a['class'][0][0]) - 1
                img_path = os.path.join(train_img_dir, fname)
                if os.path.exists(img_path): tr_lines.append(f"{img_path}\t{cidx}")
            except: continue
        with open(train_idx, "w") as f: f.write("\n".join(tr_lines))
        print(f"訓練: {len(tr_lines)}")

    test_img_dir = os.path.join(local_cars_root, "cars_test")
    if test_annos_path:
        annos = scipy.io.loadmat(test_annos_path)['annotations'][0]
        te_lines = []
        for a in annos:
            try:
                fname, cidx = str(a['fname'][0]), int(a['class'][0][0]) - 1
                img_path = os.path.join(test_img_dir, fname)
                if os.path.exists(img_path): te_lines.append(f"{img_path}\t{cidx}")
            except: continue
        with open(test_idx, "w") as f: f.write("\n".join(te_lines))
        print(f"測試: {len(te_lines)}")
    else:
        print("使用train_set 80/20 split")
        if train_annos_path:
            annos = scipy.io.loadmat(train_annos_path)['annotations'][0]
            cls_samples = {}
            for a in annos:
                try:
                    fname, cidx = str(a['fname'][0]), int(a['class'][0][0]) - 1
                    img_path = os.path.join(train_img_dir, fname)
                    if os.path.exists(img_path): cls_samples.setdefault(cidx, []).append(img_path)
                except: continue
            random.seed(42)
            tr_lines, te_lines = [], []
            for c in sorted(cls_samples.keys()):
                s = cls_samples[c]; random.shuffle(s); idx = int(len(s) * 0.8)
                tr_lines.extend([f"{p}\t{c}" for p in s[:idx]])
                te_lines.extend([f"{p}\t{c}" for p in s[idx:]])
            with open(train_idx, "w") as f: f.write("\n".join(tr_lines))
            with open(test_idx, "w") as f: f.write("\n".join(te_lines))

setup_cars_index_official()

# SUN397 
print("\nSUN397 (symlink)")
src_sun397 = os.path.join(DATASETS_DIR, "sun397")
if not os.path.exists(src_sun397): src_sun397 = os.path.join(DATASETS_DIR, "SUN397")
ensure_symlink(src_sun397, os.path.join(LOCAL_DATA_ROOT, "sun397"))

# EuroSAT
print("\nEuroSAT")
drive_eurosat = os.path.join(DATASETS_DIR, "EuroSAT_splits")
local_eurosat = os.path.join(LOCAL_DATA_ROOT, "EuroSAT_splits")
eurosat_local = copy_to_local(drive_eurosat, os.path.join(LOCAL_CACHE, "EuroSAT_splits"), "EuroSAT")
if eurosat_local: ensure_symlink(eurosat_local, local_eurosat)
else: ensure_symlink(drive_eurosat, local_eurosat)


# DTD
print("\nDTD")
drive_dtd = os.path.join(DATASETS_DIR, "dtd")
local_dtd = os.path.join(LOCAL_DATA_ROOT, "dtd")
dtd_local = copy_to_local(drive_dtd, os.path.join(LOCAL_CACHE, "dtd"), "DTD")
if dtd_local: ensure_symlink(dtd_local, local_dtd)
else: ensure_symlink(drive_dtd, local_dtd)

if os.path.isdir(local_dtd) and os.path.exists(os.path.join(local_dtd, 'images')):
    if not os.path.exists(os.path.join(local_dtd, 'train')):
        try:
            os.symlink(os.path.join(local_dtd, 'images'), os.path.join(local_dtd, 'train'))
            os.symlink(os.path.join(local_dtd, 'images'), os.path.join(local_dtd, 'test'))
        except: pass


# 其他
print("\n其他資料集")
for name in ["svhn", "MNIST"]:
    ensure_symlink(os.path.join(DATASETS_DIR, name), os.path.join(LOCAL_DATA_ROOT, name))

print(f"\n資料集: {os.listdir(LOCAL_DATA_ROOT)}")




# Dataset Loaders
with open(os.path.join(DATASETS_DIR, "resisc45.py"), "w") as f:
    f.write(r'''
import os
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
class RESISC45Dataset(ImageFolder):
    def __init__(self, root, split="train", transforms=None):
        split_file = None
        for p in [os.path.join(root, "RESISC45", f"resisc45-{split}.txt"), os.path.join(root, f"resisc45-{split}.txt")]:
            if os.path.exists(p): split_file = p; break
        valid_fns = set()
        if split_file:
            with open(split_file) as f:
                for line in f:
                    parts = line.strip().split("/")
                    if len(parts) >= 2: valid_fns.add(f"{parts[-2]}/{parts[-1]}")
        data_root = None
        for d in [os.path.join(root, "RESISC45", "NWPU-RESISC45"), os.path.join(root, "NWPU-RESISC45")]:
            if os.path.isdir(d): data_root = d; break
        if not data_root: raise FileNotFoundError("RESISC45 not found")
        is_valid = (lambda x: f"{x.split(os.sep)[-2]}/{x.split(os.sep)[-1]}" in valid_fns) if valid_fns else None
        super().__init__(root=data_root, transform=transforms, is_valid_file=is_valid)
class RESISC45:
    def __init__(self, preprocess, location, batch_size=32, num_workers=0):
        self.train_dataset = RESISC45Dataset(location, 'train', preprocess)
        self.train_loader = DataLoader(self.train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)
        self.test_dataset = RESISC45Dataset(location, 'test', preprocess)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, num_workers=num_workers)
        self.classnames = list(self.train_dataset.class_to_idx.keys())
''')

with open(os.path.join(DATASETS_DIR, "cars.py"), "w") as f:
    f.write(r'''
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
class CarsIndexDataset(Dataset):
    def __init__(self, root, split="train", transform=None):
        self.transform, self.samples, self.classes = transform, [], []
        data_root = None
        for d in [os.path.join(root, "stanford_cars"), root]:
            if os.path.exists(os.path.join(d, f"{split}_index.txt")): data_root = d; break
        if not data_root: raise FileNotFoundError("Cars index not found")
        with open(os.path.join(data_root, "classes.txt")) as f:
            self.classes = [l.strip().split("\t")[1] for l in f if "\t" in l]
        with open(os.path.join(data_root, f"{split}_index.txt")) as f:
            for l in f:
                p = l.strip().split("\t")
                if len(p)==2 and os.path.exists(p[0]): self.samples.append((p[0], int(p[1])))
        print(f"[Cars] {split}: {len(self.samples)} imgs")
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        p, l = self.samples[i]
        img = Image.open(p).convert("RGB")
        return (self.transform(img) if self.transform else img), l
class Cars:
    def __init__(self, preprocess, location, batch_size=32, num_workers=0):
        self.train_loader = DataLoader(CarsIndexDataset(location, 'train', preprocess), batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.test_loader = DataLoader(CarsIndexDataset(location, 'test', preprocess), batch_size=batch_size, num_workers=num_workers)
        self.classnames = self.train_loader.dataset.classes
''')


code_content = f'''
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
import types
import logging


# Environment Setup
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import open_clip.model
import open_clip.transformer

# Compatibility patches for open_clip
if hasattr(open_clip.model, 'VisionTransformer') and not hasattr(open_clip.model, 'VisualTransformer'):
    open_clip.model.VisualTransformer = open_clip.model.VisionTransformer
if not hasattr(open_clip.model, 'Transformer') and hasattr(open_clip.transformer, 'Transformer'):
    open_clip.model.Transformer = open_clip.transformer.Transformer
if not hasattr(open_clip.model, 'ResidualAttentionBlock') and hasattr(open_clip.transformer, 'ResidualAttentionBlock'):
    open_clip.model.ResidualAttentionBlock = open_clip.transformer.ResidualAttentionBlock

try:
    import modeling
    src = types.ModuleType('src')
    sys.modules['src'] = src
    src.modeling = modeling
    sys.modules['src.modeling'] = modeling
except:
    pass

# Fix torch.load recursion issue
if not hasattr(torch, '_original_load'):
    torch._original_load = torch.load

def _safe_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return torch._original_load(*args, **kwargs)

torch.load = _safe_load

from task_vectors import TaskVector
from eval import eval_single_dataset



# Logging Setup
def create_log_dir(path, filename='log.txt'):
    if not os.path.exists(path):
        os.makedirs(path)
    logger = logging.getLogger(path)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(path + '/' + filename)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger



# Model Utils
def fix_model(model):
    """Fix compatibility issues with older model checkpoints."""
    for m in model.modules():
        t = str(type(m))
        if "Transformer" in t and not hasattr(m, 'batch_first'):
            m.batch_first = True
        if "VisionTransformer" in t or "VisualTransformer" in t:
            for attr, val in [('patch_dropout', nn.Identity()), ('attn_pool', None),
                              ('final_ln_after_pool', False), ('input_patchnorm', False),
                              ('global_average_pool', False), ('output_tokens', False)]:
                if not hasattr(m, attr):
                    setattr(m, attr, val)
        if "ResidualAttentionBlock" in t:
            for attr in ['ls_1', 'ls_2']:
                if not hasattr(m, attr):
                    setattr(m, attr, nn.Identity())
    if hasattr(model, 'visual'):
        if not hasattr(model.visual, 'patch_dropout'):
            model.visual.patch_dropout = nn.Identity()
        if not hasattr(model.visual, 'input_patchnorm'):
            model.visual.input_patchnorm = False
    return model

def apply_vector(vector, pretrained_checkpoint):
    """Apply a task vector to a pretrained model."""
    with torch.no_grad():
        pretrained_model = fix_model(torch.load(pretrained_checkpoint))
        pretrained_state_dict = pretrained_model.state_dict()
        new_state_dict = {{}}
        for key in pretrained_state_dict:
            if key not in vector:
                continue
            new_state_dict[key] = pretrained_state_dict[key] + vector[key]
    pretrained_model.load_state_dict(new_state_dict, strict=False)
    return pretrained_model



# Trimmed Mean Function
TRIM_RATIO = 0.1

def trimmed_mean(tensor, ratio=TRIM_RATIO):
    """Compute trimmed mean (remove top/bottom ratio before averaging)."""
    tensor = torch.abs(tensor).view(-1).float()
    if tensor.numel() == 0:
        return torch.tensor(0.0, device=tensor.device)
    sorted_tensor = torch.sort(tensor.cpu())[0] if tensor.device.type == 'cuda' else torch.sort(tensor)[0]
    lo = int(tensor.numel() * ratio)
    hi = int(tensor.numel() * (1 - ratio))
    if lo < hi:
        return torch.mean(sorted_tensor[lo:hi]).to(tensor.device)
    return torch.mean(tensor).to(tensor.device)



# ELECT Helper Function
def basic_elect(tensors_list):
    stacked = torch.stack(tensors_list, 0)
    sum_tensor = stacked.sum(0)
    flag = (sum_tensor > 0) * 2 - 1  # +1 or -1
    
    param_max = torch.zeros_like(tensors_list[0])
    for param in tensors_list:
        mask = (param * flag) > 0
        param_abs = torch.abs(mask * param)
        param_max = torch.where(param_abs > param_max, param_abs, param_max)
    
    return param_max * flag




# EMR Merge
def hierarchical_emr_merge(task_vectors, dataset_names):
    """
    Hierarchical EMR Merge with Layer-wise Trimmed Mean Rescaling.
    
    Step 1: Cluster similar tasks using cosine similarity
    Step 2: Hierarchical ELECT - merge similar tasks first, then merge with others
    Step 3: Layer-wise rescaling using trimmed mean
    """

    # Collect all parameters
    n2p = []
    sum_param = {{}}
    for m in range(len(task_vectors)):
        n2p_temp = task_vectors[m].vector
        n2p.append(n2p_temp)
        for n in n2p_temp:
            if n not in sum_param:
                sum_param[n] = []
            sum_param[n].append(n2p_temp[n])
    
    
    # Clustering - Find most similar task pair
    print("Clustering tasks by similarity")
    sample_layer = list(sum_param.keys())[0]
    vecs = [tv.vector[sample_layer].view(-1).float() for tv in task_vectors]
    vecs_stack = torch.stack(vecs)
    vecs_norm = F.normalize(vecs_stack, p=2, dim=1)
    sim_matrix = torch.mm(vecs_norm, vecs_norm.t())
    
    n_tasks = len(task_vectors)
    sim_matrix.fill_diagonal_(-1)  # Exclude self-similarity
    max_idx = torch.argmax(sim_matrix).item()
    row, col = max_idx // n_tasks, max_idx % n_tasks
    
    print(f"   Most similar pair: {{dataset_names[row]}} & {{dataset_names[col]}}")
    

    # Hierarchical ELECT
    print("Hierarchical ELECT")
    vector_unified = {{}}
    for n in sum_param:
        tensors = sum_param[n]
        # First merge the similar pair
        group1 = basic_elect([tensors[row], tensors[col]])
        # Then merge with remaining tasks
        remaining = [tensors[i] for i in range(n_tasks) if i != row and i != col]
        vector_unified[n] = basic_elect([group1] + remaining)
    

    # Layer-wise Trimmed Mean Rescaling
    print("Layer-wise Trimmed Mean Rescaling")
    masks = {{}}
    rescalers = {{}}
    
    for n in sum_param:
        masks[n] = []
        rescalers[n] = []
        uni_flag = torch.sign(vector_unified[n])
        
        # Compute masks for each task
        for m in range(len(task_vectors)):
            param = task_vectors[m].vector[n]
            mask = (param * uni_flag) > 0
            masks[n].append(mask)
        
        # Compute layer-wise rescalers using trimmed mean
        for m in range(len(task_vectors)):
            numerator = trimmed_mean(task_vectors[m].vector[n], TRIM_RATIO)
            denominator = trimmed_mean(vector_unified[n] * masks[n][m], TRIM_RATIO)
            if denominator.item() != 0:
                rescaler = numerator / denominator
            else:
                rescaler = torch.tensor(0.0, device=denominator.device)
            rescalers[n].append(rescaler)
    
    return vector_unified, masks, rescalers
  


# Main
if __name__ == "__main__":
    # Configuration
    exam_datasets = ['SUN397', 'Cars', 'RESISC45', 'EuroSAT', 'SVHN', 'GTSRB', 'MNIST', 'DTD']
    model = 'ViT-B-32'
    
    # Arguments setup
    class Args:
        pass
    
    args = Args()
    args.home = '{PROJECT_PATH}'
    args.data_location = '/content/datasets'
    args.model = model
    args.save = args.home + '/checkpoints/' + model
    args.logs_path = args.home + '/logs/' + model
    args.openclip_cachedir = args.home + '/openclip_cache'
    args.cache_dir = args.openclip_cachedir
    args.batch_size = 64
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.workers = 4
    args.print_freq = 50
    args.seed = 42
    
    os.makedirs(args.save, exist_ok=True)
    os.makedirs(args.logs_path, exist_ok=True)
    os.makedirs(args.openclip_cachedir, exist_ok=True)
    
    pretrained_checkpoint = args.home + '/checkpoints/' + model + '/zeroshot.pt'
    
    # Setup logging
    str_time_ = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
    log = create_log_dir(args.logs_path, 'log_{{}}__hierarchical_emr.txt'.format(str_time_))
    
    log.info("=" * 60)
    log.info("Hierarchical EMR Merging + Layer-wise Trimmed Mean Rescaling")
    log.info("=" * 60)
    
    # Load task vectors
    log.info("Loading task vectors...")
    task_vectors = []
    valid_datasets = []
    
    for dataset_name in exam_datasets:
        # Try multiple possible checkpoint paths
        possible_names = [dataset_name, dataset_name.lower()]
        if dataset_name == 'EuroSAT':
            possible_names.append('EuroSAT_splits')
        
        ckpt_path = None
        for name in possible_names:
            path = args.home + '/checkpoints/' + model + '/' + name + '/finetuned.pt'
            if os.path.exists(path):
                ckpt_path = path
                break
        
        if ckpt_path:
            log.info(f"{{dataset_name}}")
            task_vectors.append(TaskVector(pretrained_checkpoint, ckpt_path))
            valid_datasets.append(dataset_name)
        else:
            log.info(f"{{dataset_name}}: checkpoint not found")
    
    if not task_vectors:
        log.info("No valid task vectors found!")
        sys.exit(1)
    
    # Merge models using Hierarchical EMR
    log.info("")
    vector_unified, masks, rescalers = hierarchical_emr_merge(task_vectors, valid_datasets)
    
    # Evaluate each task
    log.info("")
    log.info("Evaluating merged model on each task...")
    log.info("-" * 40)
    
    accs = []
    for i, dataset in enumerate(valid_datasets):
        # Reconstruct task-specific vector
        task_vector_recon = {{}}
        for n in vector_unified:
            task_vector_recon[n] = vector_unified[n] * masks[n][i] * rescalers[n][i].to(vector_unified[n].device)
        
        # Apply to pretrained model
        image_encoder = apply_vector(task_vector_recon, pretrained_checkpoint)
        
        # Evaluate
        try:
            metrics = eval_single_dataset(image_encoder, dataset, args)
            acc = metrics.get('top1') * 100
            accs.append(acc)
            log.info(f"{{dataset}}: {{acc:.2f}}%")
        except Exception as e:
            log.info(f"{{dataset}}: Error - {{e}}")
            accs.append(0)
    
    # Summary
    log.info("-" * 40)
    valid_accs = [a for a in accs if a > 0]
    if valid_accs:
        log.info(f"Average Accuracy: {{np.mean(valid_accs):.2f}}%")
    log.info("=" * 60)
'''

py_path = os.path.join(PROJECT_PATH, "main_emr_merging.py")
with open(py_path, "w") as f:
    f.write(code_content)

print("\n" + "=" * 60)
print("=" * 60)
!python "{py_path}"
