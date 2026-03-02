# Introduction

Underwater Salient Object Detection (USOD) aims to identify and segment the most visually distinctive objects in underwater scenes. It serves as a fundamental step for various marine applications, such as underwater robot navigation, marine biological monitoring, and underwater search and rescue. Unlike terrestrial scenes, underwater environments present unique and severe challenges, primarily characterized by **wavelength-dependent light attenuation** and **scattering**. As light propagates through water, it undergoes exponential attenuation that varies with wavelength, leading to color casts and low contrast. Furthermore, suspended particles cause backscattering, which introduces haze and noise. These physical phenomena result in images with non-uniform illumination (e.g., vignetting, artificial light spots, "marine snow") and degraded structural details (e.g., blurred edges, low object-background contrast), making accurate saliency detection notoriously difficult.

Deep learning-based methods have dominated the USOD field in recent years, significantly outperforming traditional handcrafted feature-based approaches. These methods can be broadly categorized into CNN-based and Transformer-based architectures.
Convolutional Neural Networks (CNNs) have been the de facto standard. By stacking convolutional layers with local receptive fields, CNNs excel at extracting local features such as edges, textures, and shape boundaries. However, as empirically analyzed by Wang et al. (CVPR 2020), CNNs exhibit a strong bias towards **high-frequency components (HFC)**. While this property is beneficial for delineating precise object boundaries, standard CNNs often struggle to capture long-range dependencies and global semantic context due to the limited effective receptive field of convolution operations. Consequently, they may fail to distinguish salient objects from cluttered backgrounds in scenes with complex global lighting variations or large-scale degradation patterns.

To address the limitations of CNNs in global modeling, Vision Transformers (ViTs) have been introduced to USOD. Transformers leverage self-attention mechanisms to model global pairwise dependencies between image patches, effectively capturing long-range context. However, the quadratic computational complexity of self-attention with respect to image size hinders its efficiency, especially on high-resolution underwater images required for fine-grained detection. Moreover, Transformers can sometimes be overly sensitive to high-frequency noise, which is prevalent in underwater scenarios due to suspended particles.

Recently, State Space Models (SSMs), particularly **Mamba**, have emerged as a compelling alternative to Transformers. Mamba achieves linear computational complexity while maintaining the ability to model long-range dependencies through its Selective Scan Mechanism (SSM). Crucially, recent frequency analysis studies reveal that Mamba behaves as a low-pass filter during sequence modeling. It exhibits a strong preference for **low-frequency components (LFC)**, such as global structure, spatial layout, and illumination distribution, effectively filtering out high-frequency noise.

This observation presents a "double-edged sword" for USOD. On one hand, Mamba's low-frequency bias makes it an ideal candidate for modeling and correcting global underwater illumination degradation, which is inherently a low-frequency phenomenon. It can effectively "smooth out" lighting irregularities and capture the overall scene layout. On the other hand, its tendency to suppress high-frequency signals poses a significant risk of smoothing out critical object details, such as fine edges and textures, which are already faint and blurred in underwater imagery. Therefore, a naive application of Mamba to USOD may result in saliency maps with correct localization but coarse, imprecise boundaries.

To resolve this dilemma, we propose a novel perspective: **Physically-Guided Frequency Decoupling**. We argue that an optimal USOD framework should not rely on a single architectural inductive bias. Instead, it should synergistically leverage Mamba for global low-frequency modeling (illumination correction) and CNNs for local high-frequency refinement (structural preservation). This aligns with the physical nature of underwater images, where degradation affects different frequency bands differently.

In this paper, we present the **Illumination-Guided Mamba Network (IGMNet)**. Instead of a cumbersome dual-network architecture, we propose a lightweight yet effective **Physically-Guided Frequency Decomposition** strategy embedded within the feature extraction process. Specifically, we design a novel module that acts as a physical prior generator. It utilizes an **illumination map** to isolate low-frequency regions and processes them with Mamba to rectify global lighting. Simultaneously, it uses a **gradient map** to enhance high-frequency details, refined by a local CNN branch. By parsing features into these frequency-specific flows, IGMNet effectively reconstructs features that are both globally consistent and locally sharp. Furthermore, recognizing that depth maps in underwater environments often suffer from quality variations, we introduce a **Light-Illumination-Guided Attention Module (LIQAM)**. This module uses the generated illumination flow as a global reliability gauge to robustly fuse multi-modal features.

Our main contributions are summarized as follows:
1.  We propose **IGMNet**, a novel architecture that theoretically unifies CNNs and Mamba through the lens of frequency decoupling. This design specifically addresses the dual challenges of global illumination degradation and local detail blurring in underwater imaging.
2.  We design the **Symmetric Illumination-Guided Mamba (S-IGM)** module, which functions as a **physical prior generator**. It explicitly decomposes features into frequency-specific components using physical priors and processes them with architecturally matched components (Mamba for low-frequency, CNN for high-frequency).
3.  We introduce **LIQAM**, a robust fusion module that leverages the learned global illumination context to dynamically assess feature reliability. It employs a novel High-Frequency Alignment (HFA) mechanism and an Illumination-Guided Hybrid Attention (IGHA) to ensure effective integration of RGB and depth information even under severe degradation.
4.  Extensive experiments on benchmark USOD datasets demonstrate that our method achieves state-of-the-art performance, significantly outperforming existing approaches, validating the effectiveness of the proposed frequency-aware design.

# Methodology

## 1. Theoretical Motivation: Frequency Decoupling

The design of our proposed network is grounded in the complementary frequency preferences of different neural architectures. Recent studies have revealed distinct behaviors in how deep networks process visual information:

*   **CNNs Bias Towards High-Frequency:** Wang et al. (CVPR 2020) demonstrated that Convolutional Neural Networks (CNNs) tend to capture **high-frequency components (HFC)**, such as local textures and edges. While this enables precise boundary delineation, standard CNNs often struggle with global semantic consistency due to limited receptive fields.
*   **Mamba Bias Towards Low-Frequency:** Conversely, recent analyses on State Space Models (SSMs) like Mamba suggest they act as low-pass filters during long-sequence modeling. They excel at capturing **low-frequency components (LFC)**, such as global structure and illumination distribution, but inherently suppress high-frequency details.

To address the unique dual challenge of USOD (illumination degradation and blurred boundaries), we propose the **Illumination-Guided Mamba Network (IGMNet)**. Instead of relying on a single architecture, IGMNet explicitly performs **Physically-Guided Frequency Decoupling**. We assign the global illumination correction (low-frequency) to Mamba and local structural refinement (high-frequency) to CNNs within a unified feature parsing module.

## 2. Overall Architecture

As illustrated in Figure 1, IGMNet adopts a streamlined encoder-decoder structure. The core innovation lies in the **Symmetric Illumination-Guided Mamba (S-IGM)** module embedded in the encoder stages. This module does not split the network into two heavy streams; rather, it acts as an internal **frequency parser and prior generator**, decomposing intermediate features into optimized flows before fusing them back. Additionally, the **Light-Illumination-Guided Attention Module (LIQAM)** in the decoder performs reliability-aware multi-modal fusion.

For the backbone, we employ **T2T-ViT** to preserve local structure during initial feature extraction. The network consists of four stages, with S-IGM modules applied to enhance features at each stage.

## 3. Symmetric Illumination-Guided Mamba (S-IGM)

The S-IGM module is the engine of our frequency decoupling strategy. It is designed to decompose features into frequency-specific flows based on physical priors and process them with architecturally matched components.

Let $X \in \mathbb{R}^{H \times W \times C}$ be the input feature map from a specific backbone stage. We utilize two physical priors derived from the input image: the **Illumination Map ($P_{ill}$)**, which represents lighting distribution, and the **Gradient Map ($P_{grad}$)**, which represents edge information.

### 3.1. Physically-Guided Decomposition
We decompose the input feature $X$ into two distinct components, leveraging the physical correlation between visual appearance and frequency:

1.  **Low-Frequency Component ($X_{low}$)**:
    Illumination variations in underwater scenes (e.g., light attenuation, scattering) are typically smooth and globally distributed signals. We generate a soft attention mask from the illumination prior to isolate these low-frequency regions.
    $$
    M_{ill} = \sigma(\text{Conv}([X, P_{ill}]))
    $$
    $$
    X_{low} = X \odot M_{ill}
    $$
    Here, $X_{low}$ primarily contains the base structure and lighting information of the scene. By masking out high-frequency noise, we prepare a clean input for the global modeling stream.

2.  **High-Frequency Component ($X_{high}$)**:
    Object boundaries, textures, and fine details correspond to high-frequency signals. In underwater images, these are often attenuated. We explicitly enhance these details using the gradient prior, injecting high-frequency energy into the features:
    $$
    X_{high} = X \odot (1 + \alpha \cdot P_{grad})
    $$
    where $\alpha$ is a learnable parameter. This operation effectively highlights edges and textures, preventing them from being overshadowed by strong background noise or smoothed out in subsequent processing.

### 3.2. Symmetric Frequency-Matched Modeling
Based on the theoretical motivation, we design a symmetric dual-path processing scheme that matches the nature of the data with the inductive bias of the architecture:

**Path 1: Global Illumination Flow via Mamba (Low-Frequency Stream).**
Since Mamba excels at modeling low-frequency global contexts, we feed the decomposed $X_{low}$ into a **Visual Mamba (VSS Block)**. The Selective Scan Mechanism (SSM) enables the network to propagate illumination information across the entire image with linear complexity. This allows the network to effectively correct global non-uniform lighting (e.g., vignetting or backscattering) by establishing long-range dependencies between different image regions.
$$
F_{ill} = \text{Norm}(\text{Mamba}(X_{low}))
$$
The output $F_{ill}$, termed **Illumination Flow**, represents the globally corrected context and lighting distribution.

**Path 2: Local Gradient Flow via CNN (High-Frequency Stream).**
To preserve the sharp boundaries that Mamba might naturally smooth out, we process the enhanced $X_{high}$ using a **Local CNN Block** (a stack of convolution layers). The inductive bias of CNNs towards high-frequency signals ensures that the enhanced edges and textures are retained and refined.
$$
F_{grad} = \text{LocalCNN}(X_{high})
$$
The output $F_{grad}$, termed **Gradient Flow**, provides precise structural cues and boundary information.

**Reconstruction:** The final enhanced feature combines the strengths of both paths:
$$
X_{out} = F_{ill} + \gamma \cdot F_{grad}
$$
This design ensures that the network benefits from Mamba's global receptive field for scene understanding while avoiding its detail-smoothing drawback, maintaining sharp object boundaries.

## 4. Light-Illumination-Guided Attention Module (LIQAM)

In the decoding stage, we need to fuse features from the RGB stream ($F_r$) and the Depth stream ($F_d$), as well as the auxiliary Gradient Flow ($F_g$). Simply summing these features is suboptimal because underwater depth maps often suffer from quality degradation (e.g., noise from scattering) and may not always be reliable. LIQAM leverages the **Illumination Flow ($F_{ill}$)** generated by the encoder as a global reliability prior to guide the fusion process.

### 4.1. High-Frequency Alignment (HFA)
HFA evaluates the consistency of high-frequency details across modalities. It computes channel-wise correlations (similar to Intersection-over-Union) between RGB, Depth, and Gradient features at multiple scales ($s \in \{1, 1/2, 1/4\}$).
High alignment scores indicate regions/channels where structural details are consistent across modalities, suggesting high reliability. Conversely, low alignment suggests conflict or noise. The module aggregates these multi-scale alignment scores to generate channel-wise weights ($\alpha_d, \alpha_g$), allowing the network to adaptively emphasize reliable high-frequency channels and suppress noisy ones.

### 4.2. Illumination-Guided Hybrid Attention (IGHA)
IGHA spatially modulates the features based on the lighting condition. Recognizing that illumination is a low-frequency phenomenon that varies gradually over space, IGHA employs **Dilated Convolutions** to expand the receptive field, matching the scale of illumination variations.
$$
S_{gate} = \Psi_{dilated}(F_{target} \cdot F_{source} + \text{Proj}(F_{ill}))
$$
This design allows the module to perceive the broader lighting context (e.g., identifying dark corners or bright spots). It suppresses feature activations in regions with poor lighting reliability (e.g., dark or scattering-heavy areas) where features are likely degraded, while enhancing them in well-lit regions where features are more trustworthy.

### 4.3. Modulated Fusion
The final fusion integrates the frequency-decoupled cues through a modulation mechanism:
$$
F_{fused} = F_r + (\alpha_d \cdot F_d \cdot \beta_d) + (\alpha_g \cdot F_g \cdot \beta_g)
$$
where $\beta_d$ and $\beta_g$ are the spatial attention maps from IGHA. By explicitly modulating Depth and Gradient features with both channel-wise weights (representing high-frequency structural consistency) and spatial weights (representing low-frequency illumination reliability), LIQAM ensures a robust fusion that is resilient to the specific degradations of underwater imagery.

## 5. Loss Function

To supervise both frequency components effectively, we employ a multi-task learning strategy. We impose constraints on both the final saliency map (which contains both low and high frequency information) and the intermediate gradient map (which explicitly represents high frequency information).
$$
\mathcal{L}_{total} = \mathcal{L}_{sal}(S_{pred}, S_{gt}) + \lambda \mathcal{L}_{grad}(G_{pred}, G_{gt})
$$
*   $\mathcal{L}_{sal}$ is a combination of Binary Cross Entropy (BCE), Intersection over Union (IoU), and Structural Similarity (SSIM) losses. This combination enforces pixel-level accuracy, global structural coherence, and local patch similarity.
*   $\mathcal{L}_{grad}$ is the Mean Squared Error (MSE) loss applied to the predicted gradient maps. This term explicitly forces the network (especially the CNN stream) to learn and preserve high-frequency boundary information, acting as a structural constraint.
We apply deep supervision to the outputs of all decoder stages to facilitate gradient flow and accelerate convergence.
