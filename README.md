<div align="center">
  
## [CVPR26' Findings track] Fine-tuning is Not Enough: A Parallel Framework for Collaborative Imitation and Reinforcement Learning in End-to-end Autonomous Driving

**Zhexi Lian**<sup>1,†</sup>, **Haoran Wang**<sup>1,†</sup>, **Xuerun Yan**<sup>1,2,†</sup>, **Weimeng Lin**<sup>1</sup>, **Xianhong Zhang**<sup>1</sup>,  
**Yongyu Chen**<sup>3</sup>, **Jia Hu**<sup>1,✉</sup>

<sup>1</sup> Tongji University, <sup>2</sup> Nanyang Technological University, <sup>3</sup> Chery Automobile

<sup>†</sup> Equal contribution &nbsp;&nbsp;&nbsp; <sup>✉</sup> Corresponding author

</div>

<p align="center">
  <a href="https://arxiv.org/abs/2603.13842">
    <img src="https://img.shields.io/badge/arXiv-PaIR-b31b1b?style=for-the-badge" />
  </a>
</p>

## Abstract

End-to-end autonomous driving is typically built upon imitation learning (IL), yet its performance is constrained by the quality of human demonstrations. To overcome this limitation, recent methods incorporate reinforcement learning (RL) through sequential fine-tuning. However, such a paradigm remains suboptimal: sequential RL fine-tuning can introduce policy drift and often leads to a performance ceiling due to its dependence on the pretrained IL policy. To address these issues, we propose **PaIR-Drive**, a general **Pa**rallel framework for collaborative **I**mitation and **R**einforcement learning in end-to-end autonomous driving. During training, PaIR-Drive separates IL and RL into two **parallel** branches with conflict-free training objectives, enabling fully collaborative optimization. This design eliminates the need to retrain RL when applying a new IL policy. During inference, RL leverages the IL policy to further optimize the final plan, allowing performance beyond the prior knowledge of IL. Furthermore, we introduce a tree-structured trajectory neural sampler for GRPO in the RL branch, which enhances exploration capability. Extensive analysis on NAVSIM v1 and v2 benchmarks demonstrates that PaIR-Drive achieves **competitive** performance of **91.2 PDMS** and **87.9 EPDMS**, building upon Transfuser and DiffusionDrive IL baselines. PaIR-Drive consistently outperforms existing RL fine-tuning methods, and could even correct human experts' suboptimal behaviors. Qualitative results further confirm that PaIR-Drive can effectively explore and generate high-quality trajectories.

## Overview
<div align="center">
  <img src="assets/overall_framework.png" alt="figure" width="800">
</div>
Training process illustration of the parallel scheme of PaIR-Drive. IL branch follows a typical end-to-end planning fashion and is supervised by the human trajectory. Simultaneously, the RL branch builds upon human trajectories and aims to further explore better trajectories. In the RL branch, a tree-structured trajectory neural sampler is designed to recurrently predict the trajectory point offsets of driving intentions unseen in human demonstrations. Finally, we use trajectories and their simulated rewards for GRPO to update the policy.

## Quick Start

See [QuickStart.md](QuickStart.md) for environment setup, checkpoint download,
metric-cache construction, offline candidate caching, and RWM PDM evaluation.


## NEWS

`[2026/03/19]` [ArXiv](https://arxiv.org/abs/2603.13842) paper release.  
`[2026/06/01]` [openacess.thecvf](https://openaccess.thecvf.com/content/CVPR2026F/papers/Lian_Fine-tuning_is_Not_Enough_A_Parallel_Framework_for_Collaborative_Imitation_CVPRF_2026_paper.pdf) offical paper release.  
`[2026/08/27]` Model checkpoints and inference pipeline release.

## TODO list

- [√] Release paper
- [√] Release pretrained checkpoints
- [√] Release inference code

## Citation
```
@InProceedings{Lian_2026_CVPR,
    author    = {Lian, Zhexi and Wang, Haoran and Yan, Xuerun and Lin, Weimeng and Zhang, Xianhong and Chen, Yongyu and Hu, Jia},
    title     = {Fine-tuning is Not Enough: A Parallel Framework for Collaborative Imitation and Reinforcement Learning in End-to-end Autonomous Driving},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings},
    month     = {June},
    year      = {2026},
    pages     = {920-930}
}
```
