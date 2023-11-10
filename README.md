# README for TDFNet

Welcome to the GitHub repository for TDFNet: An Efficient Audio-Visual Speech Separation Model with Top-down Fusion. This repository contains the implementation of TDFNet, a state-of-the-art method for audio-visual speech separation.

## Abstract

Audio-visual speech separation has gained significant traction in recent years due to its potential applications in various fields such as speech recognition, diarization, scene analysis and assistive technologies. Designing a lightweight audio-visual speech separation network is important for low-latency applications, but existing methods often require higher computational costs and more parameters to achieve better separation performance. In this paper, we present an audio-visual speech separation model called Top-Down-Fusion Net (TDFNet), a state-of-the-art (SOTA) model for audio-visual speech separation, which builds upon the architecture of TDANet, an audio-only speech separation method. TDANet serves as the architectural foundation for the auditory and visual networks within TDFNet, offering an efficient model with fewer parameters. On the LRS2-2Mix dataset, TDFNet achieves a performance increase of up to 10% across all performance metrics compared with the previous SOTA method CTCNet. Remarkably, these results are achieved using fewer parameters and only 28% of the multiply-accumulate operations (MACs) of CTCNet. In essence, our method presents a highly effective and efficient solution to the challenges of speech separation within the audio-visual domain, making significant strides in harnessing visual information optimally. 

## Introduction

TDFNet is a cutting-edge method in the field of audio-visual speech separation. It introduces a multi-scale and multi-stage framework, leveraging the strengths of TDANet and CTCNet. This model is designed to address the inefficiencies and limitations of existing multimodal speech separation models, particularly in real-time tasks.

## Key Features

- **Efficient Design:** TDFNet showcases remarkable efficiency in computation while delivering superior performance. 
- **Multi-Stage Fusion:** The model fuses features of different modalities several times during the fusion stage, enhancing speech separation quality.
- **Adaptable Feature Extraction:** TDFNet explores the impact of different global feature extraction structures, with a particular emphasis on the use of GRU for sequence modeling to improve performance and reduce computational demands.

## Model Variants

- **TDFNet-small:** A smaller configuration with one fusion layer and three audio-only layers, offering excellent performance with very low computational cost.
- **TDFNet (MHSA + Shared):** This variant uses MHSA as the recurrent operator in both audio and video sub-networks with shared video sub-network parameters. It outperforms the SOTA method CTCNet significantly, using fewer parameters and MACs.
- **TDFNet-large:** The full-bodied version, utilizing GRU in the audio sub-network and three separate instances for the video sub-network. It surpasses all other models across all metrics with efficient MAC usage.

## Results and Comparison

- **Performance:** TDFNet outperforms the current state-of-the-art model CTCNet in several audio separation quality metrics.
- **Efficiency:** It achieves this superior performance using only 30% of the MACs required by CTCNet.
- **Improvement:** The SI-SNRi score of 15.8 indicates a 10% increase in performance compared to CTCNet.

## Conclusion

TDFNet represents a significant advancement in audio-visual speech separation. Its efficient, multi-stage fusion framework and adaptable feature extraction capabilities set a new standard in the field.

## Installation and Usage

To install and use TDFNet, follow these steps:

1. **Clone the Repository:**
   - Ensure you have Git installed on your system.
   - Clone this repository to your local machine using `git clone https://github.com/spkgyk/av-separation.git`.

2. **Setup Environment:**
   - Make sure you have Conda installed. If not, install [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
   - Navigate to the cloned repository's directory in your terminal.

3. **Modify Requirements File (Optional for Users in China):**
   - Open the `setup/requirements.yml` file.
   - If you are in China and need to use a pip mirror, uncomment the line `- -i https://pypi.tuna.tsinghua.edu.cn/simple` under the `pip` section.

4. **Run the Setup Script:**
   - Execute the `setup/conda.sh` script by running `bash setup/conda.sh`.
   - This script will activate the base Conda environment, remove any existing environment named 'av', create a new 'av' environment, and install all required dependencies from the `setup/requirements.yml` file.

---

We hope this repository aids researchers and developers in their pursuit of efficient and effective audio-visual speech separation solutions. For more details, please refer to our paper "TDFNet: An Efficient Audio-Visual Speech Separation Model with Top-down Fusion".