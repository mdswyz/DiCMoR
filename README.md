# Distribution-Consistent Modal Recovering for Incomplete Multimodal Learning, ICCV 2023.

![](https://img.shields.io/badge/Platform-PyTorch-blue)
![](https://img.shields.io/badge/Language-Python-{green}.svg)
![](https://img.shields.io/npm/l/express.svg)

We propose a distribution-consistent modal recovering (DiCMoR) method to preserve the multimodal understanding performance under missing modality cases. The key contributions include:
- We propose a novel missing modality recovery framework by transferring the distributions from the available modalities to the missing modalities, which reduces the distribution gap between the recovered data and the vanilla available data.
- We propose a cross-modal distribution transformation method by designing class-specific multimodal flows, which not only ensures the congruence of the distributions but also enhances the discriminative capacity.

## The motivation.
Multimodal machine learning dedicates to designing a strong model for understanding, reasoning, and learning by fusing multimodal data, such as language, acoustic, and image. However, in real-world scenarios, the well-trained model may be deployed when certain modalities are not available, e.g., language may be unavailable due to speech recognition errors; acoustic modality may be lost due to background noise or sensor sensing limitations; visual data may be unavailable due to lighting, occlusion, or social privacy security. In practice, the problem of missing modality inevitably degrades the multimodal understanding performance.

## The main idea of DiCMoR.

Different from the previous paradigm, the main idea of DiCMoR is to transfer the distribution from available modalities to missing ones through the nice properties of normalizing flow (i.e., invertibility and exact density estimation), and generate more confident prediction with high distribution consistency.

![](main_idea.png)

Available modality $\mathbf{X}_{\text{avail}}$ is first projected into a latent state $\mathbf{Z}$ by the forward flow function of $\mathcal{F}\_{\text{avail}}$. Then, the latent state $\mathbf{Z}$ is injected into the reverse flow function of $\mathcal{F}\_{\text{miss}}$ and transferred to the missing modality $\mathbf{X}\_{\text{miss}}$ abided by its original distribution.

## The Framework.

![](figure2.png)

The framework of DMD. Please refer to our paper for details.

## Usage

### Prerequisites
- Python 3.8
- PyTorch 1.9.0
- CUDA 11.4

### Datasets
Data files (containing processed MOSI, MOSEI datasets) can be downloaded from . 
You can put the downloaded datasets into `./dataset` directory.
Please note that the meta information and the raw data are not available due to privacy of Youtube content creators. For more details, please follow the [official website](https://github.com/A2Zadeh/CMU-MultimodalSDK) of these datasets.

### Run the Codes

Coming soon.

