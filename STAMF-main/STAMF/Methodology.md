# Methodology

## 1. Theoretical Motivation: Frequency Decoupling

The design of our proposed network is grounded in the complementary frequency preferences of different neural architectures. Recent studies have revealed distinct behaviors in how deep networks process visual information:

*   **CNNs Bias Towards High-Frequency:** Wang et al. (CVPR 2020) demonstrated that Convolutional Neural Networks (CNNs) tend to capture **high-frequency components (HFC)**, such as local textures and edges. While this enables precise boundary delineation, standard CNNs often struggle with global semantic consistency and long-range dependencies due to their limited receptive fields.
*   **Mamba Bias Towards Low-Frequency:** Conversely, recent analyses on State Space Models (SSMs) like Mamba suggest they act as low-pass filters during long-sequence modeling. They excel at capturing **low-frequency components (LFC)**, such as global structure and illumination distribution, but inherently suppress high-frequency details, leading to blurred features.

Underwater Salient Object Detection (USOD) faces a unique dual challenge: severe **global illumination degradation** (low-frequency noise) and **blurred object boundaries** (high-frequency loss). To address this, we propose the **Dual-Stream Illumination-Guided Mamba Network (DS-IGMNet)**. Instead of relying on a single architecture, DS-IGMNet explicitly performs **Physically-Guided Frequency Decoupling**, assigning the global illumination correction task (low-frequency) to Mamba and the local structural refinement task (high-frequency) to CNNs.

## 2. Overall Architecture

As illustrated in Figure 1, DS-IGMNet adopts a dual-stream encoder-decoder structure. The core innovation lies in the **Symmetric Illumination-Guided Mamba (S-IGM)** module embedded in the encoder, which realizes the frequency decoupling strategy, and the **Light-Illumination-Guided Attention Module (LIQAM)** in the decoder, which performs reliability-aware fusion.

## 3. Symmetric Illumination-Guided Mamba (S-IGM)

The S-IGM module is designed to decompose features into frequency-specific flows based on physical priors and process them with architecturally matched components.

Let $X \in \mathbb{R}^{H \times W \times C}$ be the input feature. We utilize the **Illumination Map ($P_{ill}$)** and **Gradient Map ($P_{grad}$)** as physical proxies for low-frequency and high-frequency information, respectively.

### 3.1. Physically-Guided Decomposition
We decompose the input feature $X$ into two distinct components:

1.  **Low-Frequency Component ($X_{low}$)**:
    Illumination variations in underwater scenes are typically smooth and globally distributed. We generate a soft mask from the illumination prior to isolate these low-frequency regions:
    $$
    M_{ill} = \sigma(\text{Conv}([X, P_{ill}]))
    $$
    $$
    X_{low} = X \odot M_{ill}
    $$
    Here, $X_{low}$ primarily contains the base structure and lighting information.

2.  **High-Frequency Component ($X_{high}$)**:
    Object boundaries and textures correspond to high-frequency signals. We explicitly enhance these details using the gradient prior:
    $$
    X_{high} = X \odot (1 + \alpha \cdot P_{grad})
    $$
    This operation injects high-frequency energy into the features, preventing them from being overshadowed by strong background noise.

### 3.2. Symmetric Frequency-Matched Modeling
Based on the theoretical motivation, we design a symmetric dual-path processing scheme:

**Path 1: Global Illumination Flow via Mamba (Low-Frequency Stream).**
Since Mamba excels at modeling low-frequency global contexts, we feed $X_{low}$ into a **Visual Mamba (VSS Block)**. The Selective Scan Mechanism (SSM) enables the network to propagate illumination information across the entire image, effectively correcting global non-uniform lighting (e.g., vignetting or backscattering) without being hindered by local high-frequency noise.
$$
F_{ill} = \text{Norm}(\text{Mamba}(X_{low}))
$$
The output $F_{ill}$, termed **Illumination Flow**, represents the globally corrected context.

**Path 2: Local Gradient Flow via CNN (High-Frequency Stream).**
To preserve the sharp boundaries that Mamba might smooth out, we process $X_{high}$ using a **Local CNN Block**. The inductive bias of CNNs towards high-frequency signals ensures that the enhanced edges and textures are retained and refined.
$$
F_{grad} = \text{LocalCNN}(X_{high})
$$
The output $F_{grad}$, termed **Gradient Flow**, provides precise structural cues.

**Reconstruction:** The final enhanced feature combines the strengths of both paths:
$$
X_{out} = F_{ill} + \gamma \cdot F_{grad}
$$
This design ensures that the network benefits from Mamba's global receptive field while avoiding its detail-smoothing drawback.

## 4. Light-Illumination-Guided Attention Module (LIQAM)

In the decoding stage, simply summing features from RGB and Depth streams is suboptimal due to the frequency-dependent quality variations (e.g., depth maps may lack high-frequency details but provide good low-frequency structure). To address this and further align with the frequency decoupling philosophy of HEHP, we redesign LIQAM to incorporate **Dual Calibration** and **Frequency-Aware Processing**.

### 4.1. Frequency Decoupling via Octave Convolution
Instead of treating features as a single entity, we employ **Octave Convolution** to explicitly split the input features into high-frequency ($H$) and low-frequency ($L$) components.
*   **High-Frequency Branch ($H$)**: Preserves fine-grained details (edges, textures) at the original resolution.
*   **Low-Frequency Branch ($L$)**: Captures global structure and semantics at a reduced resolution (0.5x).
This mechanism ensures that the subsequent calibration and fusion steps can operate on the appropriate frequency components, preventing noise in high-frequency channels from corrupting global structure, and vice versa.

### 4.2. Dual Calibration Mechanism (HFA++)
We upgrade the High-Frequency Alignment (HFA) to a **Dual Calibration** system that aligns features in both channel and spatial dimensions:

1.  **Channel Calibration**: Instead of a global scalar weight, we compute a channel-wise weight vector $W_c \in \mathbb{R}^C$. This allows the network to selectively emphasize channels containing valid structural information while suppressing noisy channels (e.g., those affected by water turbidity).
    $$
    W_c = \sigma(\text{MLP}(\text{GAP}(X)))
    $$
2.  **Spatial Calibration**: We generate a pixel-wise weight map $W_s \in \mathbb{R}^{H \times W}$ to suppress unreliable regions (e.g., depth artifacts). The spatial calibration is enhanced by Octave Convolution to perceive reliability across frequencies.
    $$
    W_s = \sigma(\text{OctaveConv}(X))
    $$

The calibrated feature is obtained by: $X_{calib} = X \odot W_c \odot W_s$.

### 4.3. Large-Kernel Illumination Guidance (IGHA++)
To better capture the global distribution of underwater light, we enhance the Illumination-Guided Hybrid Attention (IGHA) with **Large-Kernel Convolutions (7x7)**. This expands the receptive field, allowing the module to modulate features based on broad lighting patterns rather than just local pixel intensities.
$$
S_{gate} = \Psi_{large\_kernel}(F_{target} \cdot F_{source} + \text{Proj}(F_{ill}))
$$

### 4.4. Refinement and Fusion
Finally, the calibrated and modulated features are fused and passed through a **Refinement Module** to eliminate fusion artifacts and reconstruct a coherent feature map. The fusion logic explicitly combines the calibrated RGB features with the spatially modulated Depth and Gradient features:
$$
F_{fused} = \text{Refine}(F_{rgb}^{calib} + F_{depth}^{calib} \cdot \beta_d + F_{grad}^{calib} \cdot \beta_g)
$$
Here, $F_{rgb}^{calib}$, $F_{depth}^{calib}$, and $F_{grad}^{calib}$ are the features processed by Dual Calibration, and $\beta_d$, $\beta_g$ are the spatial weights generated by IGHA. This updated LIQAM ensures robust performance by explicitly handling frequency-specific challenges and performing fine-grained alignment, directly addressing the limitations in structural similarity ($S_{\alpha}$) observed in previous versions.

## 5. Loss Function

To supervise both frequency components, we impose constraints on both the saliency map (low & high frequency combined) and the gradient map (explicit high frequency).
$$
\mathcal{L}_{total} = \mathcal{L}_{sal}(S_{pred}, S_{gt}) + \lambda \mathcal{L}_{grad}(G_{pred}, G_{gt})
$$
$\mathcal{L}_{sal}$ includes BCE, IoU, and SSIM losses to enforce global structure and local accuracy. $\mathcal{L}_{grad}$ (MSE loss on gradients) explicitly forces the network to learn high-frequency boundary information, reinforcing the CNN stream's objective.
