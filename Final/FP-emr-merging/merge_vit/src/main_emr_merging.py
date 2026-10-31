import sys
import os
import numpy as np
sys.path.append(os.getcwd())
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import time
import sys
sys.path.append('/remote-home/yepeng2')
from task_vectors import TaskVector
from eval import eval_single_dataset
from args import parse_arguments
import torch.nn.functional as F


def compare_vectors(vec1, vec2):
    diffs = {}
    for key in vec1:
        a = vec1[key].flatten()
        b = vec2[key].flatten()
        diff = a - b
        diffs[key] = {
            'l1': diff.abs().mean().item(),       # L1 距離平均
            'l2': torch.norm(diff).item(),        # L2 範數
            'max_abs': diff.abs().max().item(),   # 最大絕對值差異
            'cos_sim': torch.nn.functional.cosine_similarity(a, b, dim=0).item()  # 余弦相似度
        }
    return diffs


def trimmed_mean(tensor, ratio=0.1):
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

def create_log_dir(path, filename='log.txt'):
    import logging
    if not os.path.exists(path):
        os.makedirs(path)
    logger = logging.getLogger(path)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(path+'/'+filename)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def apply_vector(vector, pretrained_checkpoint):#, scaling_coef=1.0):
    """Apply a task vector to a pretrained model."""
    with torch.no_grad():
        pretrained_model = torch.load(pretrained_checkpoint, map_location='cpu', weights_only=False)
        new_state_dict = {}
        pretrained_state_dict = pretrained_model.state_dict()
        for key in pretrained_state_dict:
            if key not in vector:
                print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
                continue
            new_state_dict[key] = pretrained_state_dict[key] + vector[key]
    pretrained_model.load_state_dict(new_state_dict, strict=False)
    return pretrained_model

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


def emr_merge(task_vectors):
    sum_param = {}
    n2p = []
    for m in range(len(task_vectors)):
        n2p_temp = task_vectors[m].vector
        n2p.append(n2p_temp)
        for n in n2p_temp:
            if n not in sum_param:
                sum_param[n] = []
            sum_param[n].append(n2p_temp[n])

    sum_param = {k: torch.stack(v, 0).mean(0) for k, v in sum_param.items()}
    vector_unified = {}
    scales = torch.zeros(len(task_vectors))
    masks = {}
    for n in sum_param:
        masks[n] = []
        flag = (sum_param[n]>0) * 2 - 1
        param_max = torch.zeros_like(n2p[0][n])

        for m in range(len(task_vectors)):
            param = task_vectors[m].vector[n]
            mask = (param * flag) > 0
            masks[n].append(mask)
            param_abs = torch.abs(mask*param)
            param_max = torch.where(param_abs>param_max, param_abs, param_max)
            scales[m] += torch.mean(torch.abs(param))

        vector_unified[n] =  param_max * flag
    new_scales = torch.zeros(len(task_vectors))
    for m in range(len(task_vectors)):
        for n in vector_unified:
            p = vector_unified[n] * masks[n][m]
            new_scales[m] += torch.mean(torch.abs(p))

    rescalers = scales / new_scales

    return vector_unified, masks, rescalers



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
    sum_param = {}
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
    
    print(f"   Most similar pair: {dataset_names[row]} & {dataset_names[col]}")

    # Hierarchical ELECT
    print("Hierarchical ELECT")
    vector_unified = {}
    for n in sum_param:
        tensors = sum_param[n]
        # First merge the similar pair
        group1 = basic_elect([tensors[row], tensors[col]])
        # Then merge with remaining tasks
        remaining = [tensors[i] for i in range(n_tasks) if i != row and i != col]
        vector_unified[n] = basic_elect([group1] + remaining)

    
    return vector_unified

def emr_merge_layerwise_scalers(task_vectors, log=None):
        # Collect all parameters
    n2p = []
    sum_param = {}
    for m in range(len(task_vectors)):
        n2p_temp = task_vectors[m].vector
        n2p.append(n2p_temp)
        for n in n2p_temp:
            if n not in sum_param:
                sum_param[n] = []
            sum_param[n].append(n2p_temp[n])

    masks = {}
    rescalers = {}
    
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
            numerator = trimmed_mean(task_vectors[m].vector[n], 0.1)
            denominator = trimmed_mean(vector_unified[n] * masks[n][m], 0.1)
            if denominator.item() != 0:
                rescaler = numerator / denominator
            else:
                rescaler = torch.tensor(0.0, device=denominator.device)
            rescalers[n].append(rescaler)
    
    return rescalers



def emr_merge_cosine_mask(task_vectors, alpha):
    sum_param = {}
    n2p = []
    for m in range(len(task_vectors)):
        n2p_temp = task_vectors[m].vector
        n2p.append(n2p_temp)
        for n in n2p_temp:
            if n not in sum_param:
                sum_param[n] = []
            sum_param[n].append(n2p_temp[n])
    sum_param = {k: torch.stack(v, 0).mean(0) for k, v in sum_param.items()}
    vector_unified = {}

    scales = torch.zeros(len(task_vectors))
    masks = {}
    for n in sum_param:
        masks[n] = []
        flag = (sum_param[n]>0) * 2 - 1
        param_max = torch.zeros_like(n2p[0][n])
        for m in range(len(task_vectors)):
            param = task_vectors[m].vector[n]
            mask = (param * flag) > 0
            masks[n].append(mask)
            param_abs = torch.abs(mask*param)
            param_max = torch.where(param_abs>param_max, param_abs, param_max)
            scales[m] += torch.mean(torch.abs(param))
        vector_unified[n] =  param_max * flag

    epsilon = 1e-6
    #alpha = 5
    cosine_masks = {}
    for n in sum_param:
        cosine_masks[n] = []
        for m in range(len(task_vectors)):
            param = task_vectors[m].vector[n]
            s = (param * vector_unified[n]) / (torch.sqrt(param**2) * torch.sqrt(vector_unified[n]**2) + epsilon)
            #mask = torch.sigmoid( torch.relu(alpha *s) )
            mask = torch.sigmoid(alpha * s)
            cosine_masks[n].append(mask)

    new_scales = torch.zeros(len(task_vectors))
    for m in range(len(task_vectors)):
        for n in vector_unified:
            p = vector_unified[n] * cosine_masks[n][m]
            new_scales[m] += torch.mean(torch.abs(p))
    rescalers = scales / new_scales

    return vector_unified, cosine_masks, rescalers

def emr_merge_hard_mask(task_vectors):
    sum_param = {}
    n2p = []
    for m in range(len(task_vectors)):
        n2p_temp = task_vectors[m].vector
        n2p.append(n2p_temp)
        for n in n2p_temp:
            if n not in sum_param:
                sum_param[n] = []
            sum_param[n].append(n2p_temp[n])

    sum_param = {k: torch.stack(v, 0).mean(0) for k, v in sum_param.items()}
    vector_unified = {}
    scales = torch.zeros(len(task_vectors))
    masks = {}
    threshold = 0
    for n in sum_param:
        masks[n] = []
        flag = (sum_param[n]>0) * 2 - 1
        param_max = torch.zeros_like(n2p[0][n])

        for m in range(len(task_vectors)):
            param = task_vectors[m].vector[n]
            mask = (param * flag) > threshold
            masks[n].append(mask)
            param_abs = torch.abs(mask*param)
            param_max = torch.where(param_abs>param_max, param_abs, param_max)
            scales[m] += torch.mean(torch.abs(param))

        vector_unified[n] =  param_max * flag
    new_scales = torch.zeros(len(task_vectors))
    for m in range(len(task_vectors)):
        for n in vector_unified:
            p = vector_unified[n] * masks[n][m]
            new_scales[m] += torch.mean(torch.abs(p))

    rescalers = scales / new_scales

    return vector_unified, masks, rescalers


def get_unified_vector(task_vectors, dataset_names=None, r=1, use_hierarchical=True, use_svd=True, log=None):
    """
    Flexible EMR Merge:
    - use_hierarchical: hierarchical ELECT
    - use_svd: layer-wise SVD
    """
    if not use_hierarchical and not use_svd:
        raise ValueError("At least one of use_hierarchical or use_svd must be True")

    # Step 1: Hierarchical ELECT merge
    if use_hierarchical:
        if dataset_names is None:
            raise ValueError("dataset_names must be provided for hierarchical merge")
        vector_unified = hierarchical_emr_merge(task_vectors, dataset_names)
    else:
        # 如果不做 hierarchical，先用每個層的原始 tensor 堆疊
        vector_unified = {n: v.clone() for n, v in task_vectors[0].vector.items()}

    # Step 2: Layer-wise SVD refinement
    if use_svd:
        print("Applying Layer-wise SVD refinement")
        vector_unified_svd = {}
        for i, n in enumerate(vector_unified.keys()):
            T_n = torch.stack([tv.vector[n].flatten() for tv in task_vectors], dim=1).cpu().numpy()
            max_rank = T_n.shape[1]
            r_eff = min(max(1, r), max_rank)
            
            U, S, Vt = np.linalg.svd(T_n, full_matrices=False)
            U_r = U[:, :r_eff]

            z_tasks = U_r.T @ T_n
            z_uni = np.sign(np.sum(z_tasks, axis=1)) * np.max(np.abs(z_tasks), axis=1)

            tau_uni = (U_r @ z_uni).flatten()
            sz = int(np.prod(task_vectors[0].vector[n].shape))
            vector_unified_svd[n] = torch.from_numpy(tau_uni[:sz])\
                                         .view(task_vectors[0].vector[n].shape)\
                                         .to(task_vectors[0].vector[n].device)
            if log:
                log.info(f"[EMR Merge] Merged layer {i+1}/{len(vector_unified)} : {n} with rank {r_eff}")
        vector_unified = vector_unified_svd

    return vector_unified



exam_datasets = ['SUN397', 'Cars', 'RESISC45', 'EuroSAT', 'SVHN', 'GTSRB', 'MNIST', 'DTD']
model = 'ViT-B-32'
args = parse_arguments()
args.home = '.' 
args.data_location = args.home + '/data'
args.model = model
args.save = args.home + '/checkpoints/' + model
args.logs_path = args.home + '/logs/' + model
args.batch_size = 16
args.openclip_cachedir = args.home + '/cache'
pretrained_checkpoint = args.home + '/checkpoints/'+model+'/zeroshot.pt'

str_time_ = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
log = create_log_dir(args.logs_path, 'log_{}_emr_merging.txt'.format(str_time_))

task_vectors = [
    TaskVector(pretrained_checkpoint, args.home + '/checkpoints/'+model+'/'+dataset_name+'/finetuned.pt') for dataset_name in exam_datasets
]

# merge models

vector_unified, masks, rescalers = emr_merge(task_vectors)


use_hierarchical = True if args.cluster else False
use_svd = True if args.lora is not None else False


if use_hierarchical or use_svd:
    vector_unified = get_unified_vector(task_vectors, dataset_names=exam_datasets, r=args.lora, use_hierarchical=use_hierarchical, use_svd=use_svd, log=log)


if args.soft_mask_alpha is not None:
    alpha = args.soft_mask_alpha
    print(f"Using soft mask, alpha: {alpha}")
    vector_unified, masks, rescalers = emr_merge_cosine_mask(task_vectors, alpha)

if args.trimmean is not None:
    print("Layer-wise Trimmed Mean Rescaling with ratio:", args.trimmean)
    rescalers = emr_merge_layerwise_scalers(task_vectors, log=log)

accs = []
for i, dataset in enumerate(exam_datasets):
    task_vector_recon = {}
    for n in vector_unified:
        if args.trimmean is not None:
            task_vector_recon[n] = vector_unified[n] * masks[n][i] * rescalers[n][i].to(vector_unified[n].device)
        else:
            task_vector_recon[n] =  vector_unified[n] * masks[n][i] * rescalers[i]
    image_encoder = apply_vector(task_vector_recon, pretrained_checkpoint)
    metrics = eval_single_dataset(image_encoder, dataset, args)
    log.info(str(dataset) + ':' + str(metrics.get('top1')*100)+'%')
    accs.append(metrics.get('top1')*100)
log.info('Avg ACC:' + str(np.mean(accs)) + '%')
