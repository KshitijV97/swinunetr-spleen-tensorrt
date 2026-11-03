# Glossary

One-line definitions of the terms used in this repository, in alphabetical
order. The number in brackets is the [learning-track](README.md) chapter that
explains the term properly.

| Term | Meaning |
|---|---|
| **Activation** | Any intermediate tensor produced inside the network. Activations are stored for backprop, so they dominate training memory (3, 8) |
| **AdamW** | Adam optimiser (momentum + per-weight step normalisation) with weight decay applied directly to the weights (8) |
| **Affine (matrix)** | 4×4 matrix in a NIfTI header mapping voxel indices to millimetres, including orientation (1) |
| **AMP (automatic mixed precision)** | FP16 where safe, FP32 where needed, plus loss scaling. In PyTorch: `torch.autocast` + `GradScaler` (8) |
| **Amdahl's law** | The overall speed-up is limited by the part of the work you didn't speed up (11) |
| **Argmax** | Pick the class with the highest score per voxel, turning logits into a label map (4) |
| **Arithmetic intensity** | FLOPs per byte of memory traffic. Decides whether an op is compute- or memory-bound (10) |
| **Attention (self-attention)** | Each token takes a similarity-weighted average of other tokens' values: a soft dictionary lookup (6) |
| **Augmentation** | Random, label-preserving changes to training inputs (flips, rotations, intensity shifts) (2, 3) |
| **Axial / coronal / sagittal** | The three standard viewing planes: top–bottom, front–back, left–right cuts (1) |
| **Backpropagation** | Computing all gradients by applying the chain rule backwards through the computation graph (3) |
| **Batch** | The examples processed in one gradient step. Here, 4 crops (3) |
| **Beer–Lambert law** | X-ray intensity decays exponentially with the integral of attenuation along the path (1) |
| **BF16** | 16-bit float with FP32's exponent range and a 7-bit mantissa. Not supported by the T4 (8) |
| **CacheDataset** | MONAI dataset that memoises deterministic preprocessing in RAM (8) |
| **Checkpoint** | A saved file of model weights (+ config). Not to be confused with gradient checkpointing (8) |
| **Constant folding** | Pre-computing parts of a graph that don't depend on the input (10) |
| **Contrast phase** | When the scan is taken after iodine injection. Arterial (~30 s) or portal-venous (~70 s, this dataset) (1) |
| **Convolution (layer)** | A learned multi-channel FIR filter bank, slid over the image (2, 5) |
| **Cross-entropy** | Classification loss $-\log p_\text{true}$. Gradient w.r.t. logits is $p - y$ (4) |
| **CT (computed tomography)** | 3D X-ray imaging reconstructed from many projections. Voxel values in HU (1) |
| **CUDA** | NVIDIA's GPU programming platform (10) |
| **cuDNN** | NVIDIA's library of tuned deep-learning kernels (10) |
| **Cyclic shift** | Rolling the token grid so shifted windows can be computed as regular windows (7) |
| **Dice score** | Overlap $2\lvert P \cap G\rvert / (\lvert P\rvert + \lvert G\rvert)$, from 0 to 1. Equals voxelwise F1 (8, 9) |
| **DiceCE loss** | Soft Dice loss + cross-entropy (8) |
| **DICOM** | Hospital image format: one file per slice, rich metadata (1) |
| **Engine (TensorRT)** | A compiled, GPU- and version-specific network binary (10) |
| **Epoch** | One pass over all training scans. Here, 33 steps (3) |
| **FP32 / FP16** | 32-bit and 16-bit IEEE floats. FP16: max 65,504, ~3 significant digits (8) |
| **Fusion (operator / layer)** | Combining several ops into one GPU kernel to avoid memory round-trips (10) |
| **GELU** | Smooth ReLU-like activation used in transformer MLPs (4) |
| **Gradient** | Vector of partial derivatives of the loss w.r.t. every parameter (3) |
| **Gradient checkpointing** | Recompute activations during backprop instead of storing them: compute for memory (8) |
| **GradScaler** | Dynamically scales the loss so small FP16 gradients don't underflow (8) |
| **HD95** | 95th-percentile Hausdorff distance: a boundary-error metric in mm. Not implemented here (9) |
| **Hounsfield unit (HU)** | Calibrated CT scale: air −1000, water 0 (1) |
| **Hyperparameter** | A setting chosen before training (learning rate, epochs, crop size) (3) |
| **InstanceNorm** | Normalises each channel of each sample over its spatial positions. Used in the decoder (4) |
| **IoU (Jaccard)** | $\lvert P\cap G\rvert / \lvert P\cup G\rvert$. $\mathrm{Dice} = 2J/(1+J)$ (9) |
| **LayerNorm** | Normalises each token over its channels. In every transformer block. FP16-sensitive (4, 10) |
| **Learning rate** | Gradient-descent step size, like LMS's $\mu$ (3, 8) |
| **LMS filter** | Least-mean-squares adaptive filter: SGD on squared error (3) |
| **Logits** | Raw network scores before softmax (4) |
| **Loss** | Single number measuring how wrong the predictions are. Training minimises it (3) |
| **MONAI** | PyTorch framework for medical imaging AI (monai.io) |
| **MPS** | PyTorch's Apple Silicon GPU backend |
| **MSD** | Medical Segmentation Decathlon. Task09 is our spleen dataset (1) |
| **Multi-head attention** | Several smaller attention operations in parallel, then mixed (6) |
| **NIfTI** | Research image format (`.nii.gz`): whole volume + affine in one file (1) |
| **nnU-Net** | Self-configuring U-Net framework. The standard strong baseline (5) |
| **ONNX** | Framework-neutral model graph format. A compiler IR for networks (10) |
| **ONNX Runtime** | Independent ONNX inference engine, used here to verify the export (10) |
| **Opset** | Version of the ONNX operator set (10) |
| **Overfitting** | Fitting training-set noise at the expense of generalisation (3) |
| **p95** | 95th percentile: 95 % of samples are at or below it (11) |
| **Partial volume effect** | Boundary voxels average several tissues, blurring edges (2) |
| **Patch embedding** | Cut input into non-overlapping patches and project each to a token (a strided conv) (6, 7) |
| **Patch merging** | Swin downsampling: concatenate 2×2×2 tokens, project to 2× channels (7) |
| **Pixdim / spacing** | Physical voxel size in mm per axis: the sampling period (2) |
| **Portal-venous phase** | Scan ~60–70 s after contrast injection. The spleen looks uniform (1) |
| **RAS** | Axis convention: +x Right, +y Anterior, +z Superior (1, 2) |
| **Receptive field** | The input region that can influence one output (5) |
| **Relative position bias** | Learned attention-score offset depending on token displacement (6, 7) |
| **Residual connection** | $y = x + F(x)$: a bypass that keeps gradients flowing (4) |
| **Resampling** | Interpolating a volume onto a new voxel grid (2) |
| **ROI size** | The 3D crop size for training and the sliding-window size: 96³ (2, 9) |
| **Roofline model** | Attainable FLOP/s = min(peak compute, bandwidth × intensity) (10) |
| **Segmentation** | Assigning a class to every voxel (1) |
| **SGD** | Stochastic gradient descent: gradient steps estimated from mini-batches (3) |
| **Shifted windows (SW-MSA)** | Swin blocks that offset windows by half a window so information crosses borders (7) |
| **Sliding-window inference** | Predicting a large volume with overlapping fixed-size windows, blended (9) |
| **Softmax** | Turns logits into probabilities: $e^{z_k}/\sum_j e^{z_j}$ (4) |
| **Spleen** | Blood-filtering, immune organ in the upper left abdomen. Highly vascular (1) |
| **Splenomegaly** | Enlarged spleen. A clinical reason to measure spleen volume (1) |
| **Stream (CUDA)** | An ordered GPU work queue. Calls return before work finishes (10) |
| **Swin Transformer** | Transformer with windowed, shifted attention and a multi-resolution hierarchy (7) |
| **SwinUNETR** | Swin encoder + U-Net CNN decoder for 3D medical segmentation (7) |
| **Synchronize** | Wait for all queued GPU work. Required for correct timing (11) |
| **Tactic (TensorRT)** | One candidate kernel for a layer. The builder times tactics and keeps the fastest (10) |
| **Tensor** | n-dimensional array (4) |
| **Tensor Cores** | GPU units for fast small matrix multiply-accumulate in FP16 (and other formats) (10) |
| **TensorRT** | NVIDIA's inference optimiser (compiler) and runtime (10) |
| **TF32** | 19-bit internal matmul format on Ampere and newer GPUs. Not on the T4 (8) |
| **Token** | One vector in a transformer's input set. Here, one embedded 2×2×2 patch (6) |
| **Tracing (ONNX export)** | Recording the ops executed for one example input. Python control flow is frozen (10) |
| **Transposed convolution** | Learned upsampling: zero-stuff then filter (5) |
| **U-Net** | Encoder–decoder with skip connections across matching resolutions (5) |
| **Validation set** | Held-out data used to choose checkpoints/hyperparameters. Here, 8 scans (3) |
| **Voxel** | 3D pixel: one sample of the volume (2) |
| **Warm-up (benchmarking)** | Untimed runs that absorb one-off start-up costs (11) |
| **Weight decay** | Regulariser pulling weights toward zero (8) |
| **Window (attention)** | A 7×7×7 block of tokens within which attention is computed (7) |
| **Windowing (intensity)** | Clipping HU to a range and rescaling it, like setting an ADC's full scale (1, 2) |
| **Workspace (TensorRT)** | Scratch GPU memory the builder may use while timing tactics (10) |
