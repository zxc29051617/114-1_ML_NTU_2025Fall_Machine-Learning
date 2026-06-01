  [![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/uOK6onxy)
  # ML Assignment 3: Medical Image Segmentation

  This repository contains the implementation for Assignment 3 of the Machine Learning (CSIE5043) course, Fall 2025, at National Taiwan University. The goal of this assignment is to develop and evaluate segmentation models for medical images using different neural network architectures and training strategies.


  ## 🧩 Setup
  To install all required dependencies, simply run:
  ```
  uv sync
  ```
  > [!TIP]
  > Ensure that uv is installed beforehand: `pip install uv`
  If you wish to use any additional packages, please obtain approval from the TA beforehand.

  ## 📊 Data Analysis & Preprocessing
To prepare the data for training, we performed the following steps:
1. **Data Exploration:**
   - The dataset consists of medical images with varying dimensions.
   - We observed a class imbalance where the background dominates the target region.
2. **Preprocessing:**
   - **Resizing:** All images were resized to a fixed resolution of `448x576` to fit the model input.
   - **Normalization:** Pixel values were normalized to the range `[0, 1]` (or standardized using dataset mean/std).
   - **Data Splitting:** The dataset was split into training (80%) and validation (20%) sets to monitor overfitting.


  ## 🏋️ Training
  To start training:
  ```
  bash script/run_train.sh
  ```


  ## 📦 Generating Submission
  To generate predictions for submission:
  ```
  bash scripts/run_problem{1, 2, 3}.sh PUBLIC_DIR PRIVATE_DIR
  ```


  ## 🧠 Model Architecture

  ### Problem 1: Simple-CNN UNet
  Please include the figure of your model architecture here.
  (e.g., a diagram of your CNN-based UNet structure.)  
  ```
  Input image (Batch, 1, 448, 576)
      |
  --- Encoder Block 1 (inc) ---
  DoubleConv (64 channels) -> (Batch, 64, 448, 576)
      |
  MaxPool 2x2 -> (Batch, 64, 224, 288)
      |
  --- Encoder Block 2 (conv1) ---
  DoubleConv (128 channels) -> (Batch, 128, 224, 288)
      |
  MaxPool 2x2 -> (Batch, 128, 112, 144)
      |
  --- Encoder Block 3 (conv2) ---
  DoubleConv (256 channels) -> (Batch, 256, 112, 144)
      |
  MaxPool 2x2 -> (Batch, 256, 56, 72)
      |
  --- Encoder Block 4 (conv3) ---
  DoubleConv (512 channels) -> (Batch, 512, 56, 72)
      |
  MaxPool 2x2 -> (Batch, 512, 28, 36)
      |
  --- Bottleneck (conv4) ---
  DoubleConv (1024 channels) -> (Batch, 1024, 28, 36)
      |
  Up-Conv 2x2 -> (Batch, 512, 56, 72)
      |
  Concat -> (Batch, 1024, 56, 72) <--------+ Skip from Encoder Block 4 (512, 56, 72)
      |
  --- Decoder Block 1 (conv5) ---
  DoubleConv (512 channels) -> (Batch, 512, 56, 72)
      |
  Up-Conv 2x2 -> (Batch, 256, 112, 144)
      |
  Concat -> (Batch, 512, 112, 144) <--------+ Skip from Encoder Block 3 (256, 112, 144)
      |
  --- Decoder Block 2 (conv6) ---
  DoubleConv (256 channels) -> (Batch, 256, 112, 144)
      |
  Up-Conv 2x2 -> (Batch, 128, 224, 288)
      |
  Concat -> (Batch, 256, 224, 288) <--------+ Skip from Encoder Block 2 (128, 224, 288)
      |
  --- Decoder Block 3 (conv7) ---
  DoubleConv (128 channels) -> (Batch, 128, 224, 288)
      |
  Up-Conv 2x2 -> (Batch, 64, 448, 576)
      |
  Concat -> (Batch, 128, 448, 576) <--------+ Skip from Encoder Block 1 (64, 448, 576)
      |
  --- Decoder Block 4 (conv8) ---
  DoubleConv (64 channels) -> (Batch, 64, 448, 576)
      |
  conv1x1 (classification) -> (Batch, 1, 448, 576)
      |
  Output logits (Batch, 1, 448, 576)
  ```
  ### Problem 2: Transformer + UNet
  Please include the figure of your model architecture here.
  (e.g., a diagram illustrating how Transformer layers are integrated into UNet.)
  ```
  Input image (Batch, 1, 448, 576)
      |
--- CNN Encoder (Based on Simple-CNN UNet) --------------------------------
    - Encoder Block 1 (Skip x1) -> (Batch, 64, 448, 576)
      |
    - MaxPool 2x2
      |
    - Encoder Block 2 (Skip x2) -> (Batch, 128, 224, 288)
      |
    - MaxPool 2x2
      |
    - Encoder Block 3 (Skip x3) -> (Batch, 256, 112, 144)
      |
    - MaxPool 2x2
      |
    - Encoder Block 4 (Skip x4) -> (Batch, 512, 56, 72)
      |
    - MaxPool 2x2 -> (Batch, 512, 28, 36) 
      |
=========================================================================
Transformer Bottleneck (Hybrid Encoder)
(H_patch=28, W_patch=36, N=1008, D=768)
=========================================================================
      |
1. Patch Embedding (CNN Feature Map -> Sequence)
Conv 1x1 (512 -> embed_dim=768) -> (Batch, 768, 28, 36)
      |
Flatten & Transpose             -> (Batch, N=1008, embed_dim=768)
      |
Add Dynamic Positional Embedding
      |
2. Transformer Encoder (12 x TransformerEncoderBlock)
Transformer Blocks (L=12)       -> (Batch, N=1008, 768)
      |
3. Project Back (Sequence -> CNN Feature Map)
Linear Projection (768 -> 1024)
      |
Reshape                         -> (Batch, 1024, 28, 36)
      |
=========================================================================

      |
--- CNN Decoder (Based on Simple-CNN UNet) --------------------------------
    - Up-Conv 2x2 (1024 -> 512) -> (Batch, 512, 56, 72)
      |
    - Concat                    -> (Batch, 1024, 56, 72)    <--------+ Skip from Encoder Block 4 (x4)
      |
    - Decoder Block 1 (conv5)   -> (Batch, 512, 56, 72)
      |
    - Up-Conv 2x2 (512 -> 256)  -> (Batch, 256, 112, 144)
      |
    - Concat                    -> (Batch, 512, 112, 144)   <--------+ Skip from Encoder Block 3 (x3)
      |
    - Decoder Block 2 (conv6)   -> (Batch, 256, 112, 144)
      |
    - Up-Conv 2x2 (256 -> 128)  -> (Batch, 128, 224, 288)
      |
    - Concat                    -> (Batch, 256, 224, 288)   <--------+ Skip from Encoder Block 2 (x2)
      |
    - Decoder Block 3 (conv7)   -> (Batch, 128, 224, 288)
      |
    - Up-Conv 2x2 (128 -> 64)   -> (Batch, 64, 448, 576)
      |
    - Concat                    -> (Batch, 128, 448, 576)   <--------+ Skip from Encoder Block 1 (x1)
      |
    - Decoder Block 4 (conv8)   -> (Batch, 64, 448, 576)
      |
conv1x1 (classification)        -> (Batch, 1, 448, 576)
      |
Output logits (Batch, 1, 448, 576)
  ```


  ## Model Configuration

  ### Problem 1: Simple-CNN UNet
  Provide your model configuration and hyperparameters here. (e.g., number of epochs, epoch, batch size, optimizer, learning rate, and learning rate scheduler.)
  | Hyperparameter | Value | Source (from config file) |
  | :--- | :--- | :--- |
  | **Epochs** | 100 | `trainer.n_epoch` |
  | **Batch Size** | 4 | `train_loader.batch_size` |
  | **Accumulation Steps** | 4 | `trainer.accumulation_steps` |
  | **Effective Batch Size** | 16 | Calculated |
  | **Image Size** | `[448, 576]` | `transform.resized_imaged.size` |
  | **Loss Function** | `Binary Cross-Entropy` | `criterion.name` |
  | **Optimizer** | `AdamW` | `optimizer.name` |
  | **Learning Rate (LR)** | `0.0001` | `optimizer.args.lr` |
  | **LR Scheduler** | `CosineAnnealingLR` | `lr_scheduler.name` |

  ### Problem 2: Transformer + UNet
  Provide your model configuration and hyperparameters here. (e.g., Transformer depth, attention heads, patch size, hidden dimensions, and training details, etc.)  
  | Hyperparameter | Value | Source (from config file) |
  | :--- | :--- | :--- |
  | **Epochs** | 100 | `trainer.n_epoch` |
  | **Batch Size** | 1 | `train_loader.batch_size` |
  | **Cumulative gradient step count** | 16 |`trainer.accumulation_steps`|
  | **Efficient Batch Size (EBS)** | 1×16 | `train_loader.batch_size` | 
  | **Image Size** | `[448, 576]` | `transform.resized_imaged.size` |
  | **Loss Function** | `Binary Cross-Entropy` | `criterion.name` |
  | **Optimizer** | `AdamW` | `optimizer.name` |
  | **Learning Rate (LR)** | `0.0001` | `optimizer.args.lr` |
  | **Transformer Embed Dim (D)** | 768 | `model.args.embed_dim` |
  | **Transformer Blocks (L)** | 12 | `model.args.trans_blocks` |
  | **Attention Heads** | 12 | `model.args.heads` |
  | **MLP Dimension** | 3072 | `model.args.mlp_dim` |

  
  


  ## 🔍 Self-Supervised Learning

### 1. Pretext Task: Image Inpainting
We designed a self-supervised pretext task based on **Image Reconstruction**. Inspired by Masked Autoencoders (MAE), the objective is to reconstruct the original image from a corrupted version.
- **Input:** Images with random patches masked out (using `RandomPatchMasking` with a 75% masking ratio).
- **Target:** The original, uncorrupted raw images.
- **Objective:** The model must learn to infer the missing anatomical structures based on the remaining visible context.

### 2. Model Architecture
We utilized the **TransUNet** architecture (combining CNN Encoder and Transformer Bottleneck) for both stages.
- **During Pretraining:** The final segmentation head is replaced with a reconstruction head (output channels = input channels) to predict pixel values.
- **During Fine-tuning:** The reconstruction head is discarded and replaced with a standard segmentation head (output channels = 1) to predict binary masks.

### 3. Loss Function & Training Strategy
- **Loss Function:** We used **MSE Loss (Mean Squared Error)** to minimize the pixel-wise difference between the reconstructed image and the ground truth image.
- **Training Strategy:**
    1.  **Pretraining:** The model was trained on the dataset (ignoring labels) for 200 epochs to learn image features.
    2.  **Fine-tuning:** We loaded the pretrained encoder weights and fine-tuned the entire model on the downstream segmentation task using **BCE Loss** for 50 epochs.

### 4. Benefits to Downstream Task
This approach forces the model to understand the global geometry and texture of the medical images rather than just memorizing local patterns. By learning to fill in missing tissues or organs, the Transformer encoder acquires robust semantic representations. This initialization helps the model converge faster and achieve better generalization on the segmentation task compared to training from scratch.


  ## 📈 Learning Curve
  Include the following plots:
  - Learning rate (per step)

  ![alt text](image.png)

  - Training loss (per step)

  ![alt text](image-3.png)

  - Validation Dice Score (per epoch)
  
  ![alt text](image-2.png)
  

  > [!TIP]
  > You can download these logs and plots directly from Weights & Biases (wandb).


  ## 🧪 Experiment Results

  ### Data Augmentation Ablation
 <table>
   <tr>
     <td>Data Augmentation</td>
     <td>Validation DSC</td>
   </tr>
   <tr>
     <td>❌</td>
     <td> 0.4794</td>
   </tr>
   <tr>
     <td>✅</td>
     <td>0.5854</td>
   </tr>
 </table>

 > **Observation:** The results show a significant improvement (+10.6%) in validation DSC when data augmentation is applied. <br>
 > **Insight:** Medical images often vary in orientation and size. Augmentations like rotation (15°) and flipping (horizontal/vertical) help the model generalize better by simulating these variations, preventing the model from overfitting to the specific poses in the training set.

  ### Loss Ablations
 <table>
   <tr>
     <td>Loss</td>
     <td>Validation DSC</td>
   </tr>
   <tr>
     <td>Binary Cross-Entropy</td>
     <td> 0.5829</td>
   </tr>
   <tr>
     <td>Dice Loss</td>
     <td>0.2861</td>
   </tr>
 </table>

 > **Observation:** Binary Cross-Entropy (BCE) significantly outperformed Dice Loss in this experiment (0.5829 vs. 0.2861). <br>
 > **Insight:** While Dice Loss is theoretically better for handling class imbalance, it can suffer from training instability, especially with smaller batch sizes (batch size = 4 in this setup). Even with gradient accumulation, the loss is calculated on the mini-batch, where having few positive pixels can lead to erratic gradients. BCE provides a smoother loss landscape, leading to more stable convergence for this Simple-CNN model.

  ### Model Ablations
 </table>
   <tr>
     <td>Model</td>
     <td>Validation DSC</td>
     <td>Public DSC</td>
   </tr>
   <tr>
     <td>UNet + CNN</td>
     <td>0.5854</td>
     <td>0.5337</td>
   </tr>
   <tr>
     <td>UNet + Transformer</td>
     <td>0.5854</td>
     <td>0.5344</td>
   </tr>
   <tr>
     <td>UNet + Transformer + SSL</td>
     <td> 0.5825</td>
     <td>0.5344</td>
   </tr>
 <table>

 > **Observation:** When trained from scratch, the Simple-CNN and TransUNet achieved identical Validation DSC (0.5854), with TransUNet showing a negligible improvement in Public DSC. <br>
 > **Insight:** Transformers lack the inductive biases (like translation invariance) inherent in CNNs and typically require massive datasets to perform well. Without pre-training, the Transformer component struggles to learn better features than the CNN baseline on this limited dataset. SSL (Problem 3) addresses this by forcing the model to learn global context and texture through reconstruction, effectively "warming up" the weights for the downstream task.


  📚 Notes
  - Do not upload model checkpoints or data to GitHub repository.
  - Ensure your results are reproducible by fixing random seeds and clearly specifying configurations.
  - Figures, Tables, and experiment descriptions should be self-contained and interpretable without external references.
  - Please maintain clarity and organization throughout your report and code structure.
