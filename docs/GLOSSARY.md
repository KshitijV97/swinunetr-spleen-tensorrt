# Glossary

Short definitions of terms used across this repository, in alphabetical order.

| Term | Meaning |
|---|---|
| **Affine (matrix)** | 4×4 matrix stored in a NIfTI file that maps voxel indices (i, j, k) to physical coordinates in millimetres, including orientation |
| **AdamW** | Adam optimiser with *decoupled* weight decay: the weights shrink directly, rather than through the gradient. Standard for transformers |
| **AMP (automatic mixed precision)** | Running most operations in FP16 while keeping precision-sensitive ones in FP32. In PyTorch: `torch.autocast` + `GradScaler` |
| **Argmax** | Picking the class with the highest logit for each voxel; turns network output into a label map |
| **Attention (self-attention)** | Each token computes weighted sums of all other tokens' values, with weights from query–key similarity. Lets the model relate distant positions |
| **CacheDataset** | MONAI dataset that stores the output of deterministic transforms in RAM, so they are not recomputed each epoch |
| **CT (computed tomography)** | 3D X-ray imaging. Voxel values are in Hounsfield units |
| **CUDA** | NVIDIA's GPU programming platform. PyTorch's GPU backend on NVIDIA hardware |
| **cuDNN** | NVIDIA's library of tuned deep-learning kernels (convolutions etc.) used by PyTorch |
| **Dice score** | Overlap metric: 2·|P∩G| / (|P|+|G|). 1 = perfect, 0 = no overlap |
| **DiceCE loss** | Sum of soft Dice loss and cross-entropy loss |
| **Engine (TensorRT)** | A compiled, GPU-specific, optimised version of a network, produced by the TensorRT builder |
| **Epoch** | One pass over all training volumes |
| **FP32 / FP16 / TF32 / INT8** | Number formats: 32-bit float (default), 16-bit float (faster, less range/precision), TensorFloat-32 (FP32 range with a 10-bit mantissa, used by Ampere+ GPUs for matmuls), 8-bit integer (needs calibration) |
| **Gradient checkpointing** | Not storing intermediate activations during the forward pass and recomputing them during the backward pass. Trades compute for memory |
| **GradScaler** | Scales the loss up before backward so small FP16 gradients don't underflow to zero, then scales gradients back down |
| **Hounsfield unit (HU)** | CT intensity scale: air −1000, water 0, bone up to several thousand |
| **Intensity windowing** | Clipping intensities to a range of interest (e.g. −57…164 HU for soft tissue) and rescaling it |
| **Latency** | Time to process one input (here: one CT volume) |
| **LayerNorm** | Normalises each token's features to zero mean and unit variance; used in every transformer block |
| **Logits** | Raw, unnormalised network outputs before softmax |
| **MONAI** | PyTorch-based open-source framework for medical imaging AI (<https://monai.io>) |
| **MPS** | Metal Performance Shaders: PyTorch's backend for Apple Silicon GPUs |
| **MSD** | Medical Segmentation Decathlon: 10 public 3D segmentation tasks |
| **NIfTI** | `.nii` / `.nii.gz` file format for 3D/4D medical images, with the affine in its header |
| **ONNX** | Open Neural Network Exchange: framework-neutral model file format |
| **ONNX Runtime** | Microsoft's inference engine for ONNX models. Used here to verify the export on CPU |
| **Opset** | Version of the ONNX operator set a model uses |
| **Orientation (RAS)** | Axis convention: +x towards the patient's Right, +y Anterior, +z Superior |
| **p95** | 95th percentile: 95% of samples are at or below this value |
| **Patch embedding** | Splitting the input into small non-overlapping patches and projecting each to a vector (a strided convolution) |
| **Patch merging** | Swin's downsampling: concatenate 2×2×2 neighbouring tokens and project to 2× the channels |
| **ROI (region of interest) size** | Here: the 3D crop size used for training and as the sliding-window size (96³) |
| **Sliding-window inference** | Segmenting a large volume by running the model on overlapping fixed-size windows and blending the outputs |
| **Spacing (voxel spacing / pixdim)** | Physical size of a voxel in mm along each axis |
| **Swin Transformer** | Transformer that computes attention within local windows and shifts the windows between layers, building a hierarchy like a CNN |
| **SwinUNETR** | Swin Transformer encoder + U-Net-style CNN decoder, for 3D medical segmentation |
| **Tactic (TensorRT)** | One candidate kernel implementation for a layer. The builder times tactics and keeps the fastest |
| **TensorRT** | NVIDIA's inference optimiser and runtime for NVIDIA GPUs |
| **Token** | One vector in a transformer's sequence. Here, one embedded 2×2×2 patch |
| **Transposed convolution** | Learnable upsampling layer used in the decoder |
| **U-Net** | Encoder–decoder segmentation network with skip connections between matching resolutions |
| **Voxel** | 3D pixel |
| **Warm-up (benchmarking)** | Untimed runs before measurement, to exclude one-off initialisation costs |
| **Workspace (TensorRT)** | Scratch GPU memory the builder may use while trying tactics |
