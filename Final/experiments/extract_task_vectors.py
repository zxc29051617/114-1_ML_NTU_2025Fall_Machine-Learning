import torch
import os
import gc  # 引入垃圾回收機制，這是省記憶體的關鍵

class TaskVectorManager:
    def __init__(self, pretrained_path, task_paths_list):
        """
        初始化管理器
        :param pretrained_path: 預訓練模型路徑 (Base Model)
        :param task_paths_list: 一個包含所有微調模型路徑的 List
        """
        self.pt_path = pretrained_path
        self.task_paths = task_paths_list
        
        print(f"Loading Base Model from: {pretrained_path}")
        self.pretrained_state = self._load_state_dict(pretrained_path)
        print("Base model loaded successfully.")

        # 掃描並建立層數清單 (例如: [0, 1, 2... 11, 'head'])
        self.layers_info = self._scan_layers()
        print(f"Detected Layers: {sorted([str(k) for k in self.layers_info])}")

    def _load_state_dict(self, path):
        """
        讀取模型權重 (包含你的 OpenCLIP 版本修正)
        """
        try:
            # 這裡保留 weights_only=False 以支援舊版 checkpoint
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
        except Exception as e:
            raise ValueError(f"無法讀取檔案 {path}: {e}")

        # 處理完整模型物件
        if hasattr(ckpt, 'state_dict'):
            return ckpt.state_dict()

        # 處理字典包裝
        if isinstance(ckpt, dict):
            if 'model' in ckpt: return ckpt['model']
            if 'state_dict' in ckpt: return ckpt['state_dict']
            
        return ckpt

    def _parse_layer_id(self, key):
        """解析層數 (支援 blocks, resblocks)"""
        parts = key.split('.')
        if "resblocks" in parts:
            return int(parts[parts.index("resblocks") + 1])
        elif "blocks" in parts:
            return int(parts[parts.index("blocks") + 1])
        elif "layers" in parts:
            return int(parts[parts.index("layers") + 1])
        else:
            if "head" in key: return "head"
            return "others"

    def _clean_key_name(self, key):
        """簡化參數名稱"""
        parts = key.split('.')
        if "resblocks" in parts:
            return ".".join(parts[parts.index("resblocks") + 2:])
        if "blocks" in parts:
            return ".".join(parts[parts.index("blocks") + 2:])
        return key

    def _scan_layers(self):
        """掃描預訓練模型有哪些層"""
        layers = set()
        for key in self.pretrained_state.keys():
            lid = self._parse_layer_id(key)
            layers.add(lid)
        return list(layers)

    def layer_generator(self):
        """
        【核心功能】產生器
        每次呼叫時，會吐出一層 (Layer) 的所有任務數據，並自動堆疊。
        """
        # 為了演示方便，我們先排序層數 (0, 1, 2...)
        sorted_layers = sorted([l for l in self.layers_info if isinstance(l, int)])
        
        for layer_id in sorted_layers:
            print(f"\n--- Processing Layer {layer_id} ---")
            
            # 暫存容器: { 'attn.weight': [Tensor_Task1, Tensor_Task2], ... }
            params_buffer = {}
            
            # 依序讀取每個任務模型
            for t_idx, t_path in enumerate(self.task_paths):
                # 1. 讀檔 (只在需要時讀入記憶體)
                # print(f"  Loading task {t_idx+1}/{len(self.task_paths)}: {os.path.basename(t_path)}")
                ft_state = self._load_state_dict(t_path)
                
                # 2. 提取該層參數並計算 Delta
                count = 0
                for key in self.pretrained_state.keys():
                    if key in ft_state:
                        # 檢查是否屬於當前正在處理的 layer_id
                        if self._parse_layer_id(key) == layer_id:
                            delta = ft_state[key] - self.pretrained_state[key]
                            clean_name = self._clean_key_name(key)
                            
                            if clean_name not in params_buffer:
                                params_buffer[clean_name] = []
                            
                            params_buffer[clean_name].append(delta)
                            count += 1
                
                # 3. 釋放該任務模型的記憶體 (關鍵!)
                del ft_state
                gc.collect() 
            
            # 4. 堆疊 (Stacking): List[Tensor] -> Tensor[N_Tasks, ...]
            # 這裡就是把 [MNIST_tensor, Cars_tensor] 變成 [2, shape]
            stacked_output = {}
            for name, tensor_list in params_buffer.items():
                if len(tensor_list) == len(self.task_paths):
                    stacked_output[name] = torch.stack(tensor_list)
                else:
                    print(f"Warning: Param {name} missing in some tasks.")
            
            # 把這一層整理好的數據交出去
            yield layer_id, stacked_output

# --- 執行與測試區塊 ---
if __name__ == "__main__":
    # 1. 設定路徑
    pt_path = os.path.join("checkpoints", "zeroshot.pt")
    
    task_dirs = [
        "SUN397", "Cars", "RESISC45", "EuroSAT", 
        "SVHN", "GTSRB", "MNIST", "DTD"
    ]
    task_list = [os.path.join("checkpoints", d, "finetuned.pt") for d in task_dirs]

    # 2. 執行提取並存檔
    if os.path.exists(pt_path) and all(os.path.exists(t) for t in task_list):
        print("=== 開始執行批量提取並存檔 ===")
        
        manager = TaskVectorManager(pt_path, task_list)
        
        # 準備一個大字典來存所有層的數據
        full_dump = {}
        
        # 使用產生器逐層讀取
        for layer_id, stacked_data in manager.layer_generator():
            print(f"  Collecting Layer {layer_id}...")
            full_dump[layer_id] = stacked_data
            
        # 3. 存成檔案 (這就是你要的「東西」！)
        save_name = "all_task_vectors.pt"
        print(f"\n正在寫入檔案: {save_name} ...")
        torch.save(full_dump, save_name)
        
        print(f"✅ 存檔完成！檔案大小約: {os.path.getsize(save_name) / 1024 / 1024:.2f} MB")
        print("你可以把這個 'all_task_vectors.pt' 直接丟給做 SVD 的隊友。")

    else:
        print("Error: 檔案缺漏，請檢查路徑。")