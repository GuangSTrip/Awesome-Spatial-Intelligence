<h1 align="center">Awesome-Spatial-Intelligence</h1>

## Introduction

Spatial Intelligence is becoming increasingly important in the field of Artificial Intelligence. This repository aims to provide a comprehensive and systematic collection of research related to Spatial Intelligence.

Any suggestion is welcome, please feel free to raise an issue. ^_^

## Table of Contents

- [Introduction](#introduction)
- [Table of Contents](#table-of-contents)
- [Related Survey and Tutorial](#related-survey-and-tutorial)
- [1. Spatial Intelligence in various areas/tasks](#1-spatial-intelligence-in-various-areastasks)
  - [1.1 NLP](#11-nlp)
    - [1.1.1 Spatial Language Understanding](#111-spatial-language-understanding)
      - [1.1.1.1 Spatial Representation](#1111-spatial-representation)
        - [1.1.1.1.1 Natural Language Based](#11111-natural-language-based)
        - [1.1.1.1.2 Spatial Role Lableing (SpRL)](#11112-spatial-role-lableing-sprl)
        - [1.1.1.1.3 Qualitative Spatial Relations](#11113-qualitative-spatial-relations)
        - [1.1.1.1.4 ISO-Space](#11114-iso-space)
        - [1.1.1.1.5 Others](#11115-others)
      - [1.1.1.2 Spatial Language Reasoning](#1112-spatial-language-reasoning)
    - [1.1.2 Spatial Language Generation](#112-spatial-language-generation)
  - [1.2 CV](#12-cv)
    - [1.2.1 Understanding](#121-understanding)
    - [1.2.2 Detection](#122-detection)
    - [1.2.3 Depth Estimation](#123-depth-estimation)
    - [1.2.4 3D Reconstruction](#124-3d-reconstruction)
    - [1.2.5 Scene Generation](#125-scene-generation)
    - [1.2.6 Spatial Simulation](#126-spatial-simulation)
    - [1.2.7 Point Distribution/Tracking](#127-point-distributiontracking)
    - [1.2.8 Object Positioning](#128-object-positioning)
    - [1.2.9 Occupancy Prediction](#129-occupancy-prediction)
    - [1.2.10 Scene Completion](#1210-scene-completion)
    - [1.2.11 Segmentation](#1211-segmentation)
    - [1.2.12 Pose Estimation](#1212-pose-estimation)
  - [1.3 Multi-modal](#13-multi-modal)
    - [1.3.1 Understanding](#131-understanding)
    - [1.3.2 Spatial Reasoning](#132-spatial-reasoning)
    - [1.3.3 Navigation](#133-navigation)
  - [1.4 Others (radar/GPS etc.)](#14-others-radargps-etc)
- [2. Datasets and Benchmarks](#2-datasets-and-benchmarks)
- [3. Spatial Intelligence Methods](#3-spatial-intelligence-methods)
  - [3.1 old/mathematical/rule-based](#31-oldmathematicalrule-based)
    - [3.1.1 Neural Radiance Fields](#311-neural-radiance-fields)
      - [3.1.1.1 Survey](#3111-survey)
      - [3.1.1.2 Basis](#3112-basis)
        - [3.1.1.2.1 Neural Rendering Pretraining](#31121-neural-rendering-pretraining)
        - [3.1.1.2.2 Signed Distance Function](#31122-signed-distance-function)
      - [3.1.1.3 Downstream Tasks](#3113-downstream-tasks)
        - [3.1.1.3.1 Detection](#31131-detection)
        - [3.1.1.3.2 Segmentation](#31132-segmentation)
        - [3.1.1.3.3 Navigation](#31133-navigation)
    - [3.1.2 Gaussian Splatting](#312-gaussian-splatting)
      - [3.1.2.1 Survey](#3121-survey)
      - [3.1.2.2 Basis](#3122-basis)
        - [3.1.2.2.1 Optimized Rendering](#31221-optimized-rendering)
        - [3.1.2.2.2 Geometric and Material](#31222-geometric-and-material)
        - [3.1.2.2.3 Physics Simulation](#31223-physics-simulation)
      - [3.1.2.3 Scene Reconstruction](#3123-scene-reconstruction)
        - [3.1.2.3.1 Dynamic Scene Reconstruction](#31231-dynamic-scene-reconstruction)
        - [3.1.2.3.2 Large-Scale Scene Reconstruction](#31232-large-scale-scene-reconstruction)
        - [3.1.2.3.3 Sparse-View Reconstruction](#31233-sparse-view-reconstruction)
        - [3.1.2.3.4 Unposed Scene Reconstruction](#31234-unposed-scene-reconstruction)
        - [3.1.2.3.5 Sparse-View Scene Extension](#31235-sparse-view-scene-extension)
      - [3.1.2.4 Downstream Tasks](#3124-downstream-tasks)
        - [3.1.2.4.1 Detection](#31241-detection)
        - [3.1.2.4.2 Segmentation](#31242-segmentation)
        - [3.1.2.4.3 Occupancy Prediction](#31243-occupancy-prediction)
        - [3.1.2.4.4 Scene Graph Generation](#31244-scene-graph-generation)
        - [3.1.2.4.5 Navigation](#31245-navigation)
        - [3.1.2.4.6 SLAM](#31246-slam)
      - [3.1.2.5 Gaussian Splatting based 3D Foundation Model](#3125-gaussian-splatting-based-3d-foundation-model)
    - [3.1.3 Geometry Method](#313-geometry-method)
    - [3.1.4 Point Cloud](#314-point-cloud)
      - [3.1.4.1 Survey](#3141-survey)
      - [3.1.4.2 Base Model](#3142-base-model)
      - [3.1.4.3 Usage](#3143-usage)
        - [3.1.4.3.1 Multimodal Alignment](#31431-multimodal-alignment)
        - [3.1.4.3.2 LLM](#31432-llm)
      - [3.1.4.4 Downstream Tasks](#3144-downstream-tasks)
        - [3.1.4.4.1 Detection](#31441-detection)
        - [3.1.4.4.2 Segmentation](#31442-segmentation)
        - [3.1.4.4.3 Occupancy Prediction](#31443-occupancy-prediction)
        - [3.1.4.4.4 Visual Grounding](#31444-visual-grounding)
  - [3.2 Machine Learning](#32-machine-learning)
  - [3.3 deep learning](#33-deep-learning)
    - [3.3.1 Geometry Based](#331-geometry-based)
  - [3.4 LLM](#34-llm)
    - [3.4.1 Spatial Reasoning](#341-spatial-reasoning)
    - [3.4.2 Recognition](#342-recognition)
    - [3.4.3 Reinforcement Learning](#343-reinforcement-learning)
- [4. Application](#4-application)
  - [4.1 Robotics](#41-robotics)
  - [4.2 GIScience/Geo AI](#42-gisciencegeo-ai)
  - [4.3 Medicine](#43-medicine)
  - [4.4 AR/VR/XR](#44-arvrxr)
  - [4.5 Beyond AI](#45-beyond-ai)
  - [4.6 Integrated apps](#46-integrated-apps)
    - [4.6.1 World Model](#461-world-model)
    - [4.6.2 Others](#462-others)
- [Other](#other)
- [Reference Repository](#reference-repository)

## Related Survey and Tutorial

- **[Tutorial] Spatial and Temporal Language Understanding: Representation, Reasoning, and Grounding**  
  *the cutting-edge research on spatial and temporal language understanding and its applications*  
  [[Paper]](https://aclanthology.org/2020.emnlp-tutorials.5.pdf)
  [[Tutorial-Page]](https://spatial-language-tutorial.github.io/)

## 1. Spatial Intelligence in various areas/tasks

### 1.1 NLP

#### 1.1.1 Spatial Language Understanding

- **Evaluating Spatial Understanding of Large Language Models**  
  *design natural-language navigation tasks and evaluate the ability of LLMs*  
  [[Paper]](https://arxiv.org/abs/2310.14540)
  [[Code]](https://github.com/runopti/SpatialEvalLLM)

##### 1.1.1.1 Spatial Representation

###### 1.1.1.1.1 Natural Language Based

- **A linguistic ontology of space for natural language processing**  
  [[Paper]](https://www.sciencedirect.com/science/article/pii/S0004370210000858?via%3Dihub)

- **The Fundamental System of Spatial Schemas in Language**  
  [[Paper]](https://www.acsu.buffalo.edu/~talmy/talmyweb/Recent/hampevi.pdf)

###### 1.1.1.1.2 Spatial Role Lableing (SpRL)

- **Spatial Role Labeling: Task Definition and Annotation Scheme**  
  [[Paper]](http://www.lrec-conf.org/proceedings/lrec2010/pdf/846_Paper.pdf)

- **Spatial role labeling: Towards extraction of spatial relations from natural language**  
  [[Paper]](https://dl.acm.org/doi/10.1145/2050104.2050105)

- **UNITOR-HMM-TK: Structured Kernel-based Learning for Spatial Role Labeling**  
  [[Paper]](https://aclanthology.org/S13-2096.pdf)

- **Deep Embedding for Spatial Role Labeling**  
  [[Paper]](https://arxiv.org/abs/1603.08474)

- **Transfer Learning with Synthetic Corpora for Spatial Role Labeling and Reasoning**  
  [[Paper]](https://arxiv.org/abs/2210.16952)

- **From Spatial Relations to Spatial Configurations**  
  [[Paper]](https://arxiv.org/abs/2007.09557)

- **Spatial AMR: Expanded spatial annotation in the context of a grounded Minecraft corpus**  
  [[Paper]](https://www.academia.edu/download/83342661/2020.lrec-1.601.pdf)

- **A dataset of chest X-ray reports annotated with Spatial Role Labeling annotations**  
  [[Paper]](https://www.sciencedirect.com/science/article/pii/S2352340920309501)

- **SpatialNet: A Declarative Resource for Spatial Relations**  
  [[Paper]](https://www.academia.edu/download/77510693/W19-1607.pdf)

- **Rad-SpatialNet: A Frame-based Resource for Fine-Grained Spatial Relations in Radiology Reports**  
  [[Paper]](https://www.academia.edu/download/77510608/2020.lrec-1.274.pdf)

###### 1.1.1.1.3 Qualitative Spatial Relations

- **Qualitative spatial representation and reasoning: An overview**  
  [[Paper]](https://www.academia.edu/download/68031136/Qualitative_Spatial_Representation_and_R20210712-24613-1ye9z4j.pdf)

- **Qualitative spatial reasoning: Cardinal directions as an example**  
  [[Paper]](https://www.frank.gerastree.at/PublicationList/resources/docs/docsH/ijgis-frank.pdf)

- **Learning to interpret spatial natural language in terms of qualitative spatial relations**  
  [[Paper]](https://www.academia.edu/download/42762153/Learning_to_interpret_spatial_natural_la20160217-3141-5pstlm.pdf)

###### 1.1.1.1.4 ISO-Space

- **Integrating Motion Predicate Classes with Spatial and Temporal Annotations**  
  [[Paper]](https://aclanthology.org/C08-2024.pdf)

- **The Role of Model Testing in Standards Development: The Case of ISO-Space**  
  [[Paper]](https://www.researchgate.net/profile/James-Pustejovsky/publication/267560950_The_Role_of_Model_Testing_in_Standards_Development_The_Case_of_ISO-Space/links/546c9b5b0cf21e510f63ebbf/The-Role-of-Model-Testing-in-Standards-Development-The-Case-of-ISO-Space.pdf)

###### 1.1.1.1.5 Others

- **SpatialML: annotation scheme, resources, and evaluation**  
  [[Paper]](https://www.academia.edu/download/46836575/mani2010spatialml.pdf)

##### 1.1.1.2 Spatial Language Reasoning

- **StepGame: A New Benchmark for Robust Multi-Hop Spatial Reasoning in Texts**  
  *a new Question-Answering dataset called StepGame for robust multi-hop spatial reasoning in texts*  
  [[Paper]](https://arxiv.org/abs/2204.08292)
  [[Code]](https://github.com/ShiZhengyan/StepGame)

- **Advancing Spatial Reasoning in Large Language Models: An In-Depth Evaluation and Enhancement Using the StepGame Benchmark**  
  *provide a flawless solution to the benchmark by combining template-to-relation mapping with logic-based reasoning.*  
  [[Paper]](https://arxiv.org/abs/2401.03991)
  [[Code]](https://github.com/Fangjun-Li/SpatialLM-StepGame)

- **Neuro-symbolic Training for Reasoning over Spatial Language**  
  *propose training language models with neuro-symbolic techniques that exploit the spatial logical rules as constraints, providing additional supervision to improve spatial reasoning and question answering*  
  [[Paper]](https://arxiv.org/abs/2406.13828)

#### 1.1.2 Spatial Language Generation

### 1.2 CV

#### 1.2.1 Understanding

- **ImageNet3D: Towards General-Purpose Object-Level 3D Understanding**  
  *a large dataset for general-purpose object-level 3D understanding.*  
  [[Paper]](https://arxiv.org/abs/2406.09613)
  [[Project-Page]](https://imagenet3d.github.io/)
  [[Code]](https://github.com/wufeim/imagenet3d_exp)

- **SpatialSense: An Adversarially Crowdsourced Benchmark for Spatial Relation Recognition**  
  *a dataset specializing in spatial relation recognition which captures a broad spectrum of such challenges, allowing for proper benchmarking of computer vision techniques.*  
  [[Paper]](https://arxiv.org/abs/1908.02660)
  [[Code]](https://github.com/princeton-vl/SpatialSense)

- **Pri3D: Can 3D Priors Help 2D Representation Learning?**  
  *employ contrastive learning under both multi-view im-age constraints and image-geometry constraints to encode3D priors into learned 2D representations*  
  [[Paper]](https://arxiv.org/abs/2104.11225)
  [[Code]](https://github.com/Sekunde/Pri3D)

- **SpatialFormer: Towards Generalizable Vision Transformers with Explicit Spatial Understanding**  
  *SpatialFormer, with explicit spatial understanding for generalizable image representation learning.*  
  [[Paper]](https://link.springer.com/chapter/10.1007/978-3-031-72624-8_3)

- **Probing the 3D Awareness of Visual Foundation Models**  
  [[Paper]](https://arxiv.org/abs/2404.08636)
  [[Code]](https://github.com/mbanani/probe3d)

- **Improving 2D Feature Representations by 3D-Aware Fine-Tuning**  
  [[Paper]](https://arxiv.org/abs/2407.20229)
  [[Project-Page]](https://ywyue.github.io/FiT3D/)
  [[Code]](https://github.com/ywyue/FiT3D)

- **Near, far: Patch-ordering enhances vision foundation models' scene understanding**  
  [[Paper]](https://arxiv.org/abs/2408.11054)
  [[Project-Page]](https://vpariza.github.io/NeCo/)
  [[Code]](https://github.com/vpariza/NeCo)

- **Lexicon3D: Probing Visual Foundation Models for Complex 3D Scene Understanding**  
  [[Paper]](https://arxiv.org/abs/2409.03757)
  [[Project-Page]](https://yunzeman.github.io/lexicon3d/)
  [[Code]](https://github.com/YunzeMan/Lexicon3D)

- **Feat2GS: Probing Visual Foundation Models with Gaussian Splatting**  
  [[Paper]](https://arxiv.org/abs/2412.09606)
  [[Project-Page]](https://fanegg.github.io/Feat2GS/)
  [[Code]](https://github.com/fanegg/Feat2GS)

- **SimC3D: A Simple Contrastive 3D Pretraining Framework Using RGB Images**  
  [[Paper]](https://arxiv.org/abs/2412.05274)

- **Gaussian Masked Autoencoders**  
  [[Paper]](https://arxiv.org/abs/2501.03229)
  [[Project-Page]](https://brjathu.github.io/gmae/)

- **EgoDTM: Towards 3D-Aware Egocentric Video-Language Pretraining**  
  [[Paper]](https://arxiv.org/abs/2503.15470)
  [[Code]](https://github.com/xuboshen/EgoDTM)

- **Beyond Semantics: Rediscovering Spatial Awareness in Vision-Language Models**  
  [[Paper]](https://arxiv.org/abs/2503.17349)
  [[Project-Page]](https://user074.github.io/respatialaware/)

- **PE3R: Perception-Efficient 3D Reconstruction**  
  [[Paper]](https://arxiv.org/abs/2503.07507)
  [[Code]](https://github.com/hujiecpp/PE3R)

- **WildSeg3D: Segment Any 3D Objects in the Wild from 2D Images**  
  [[Paper]](https://arxiv.org/abs/2503.08407)

- **Detect Anything 3D in the Wild**  
  [[Paper]](https://arxiv.org/abs/2504.07958)
  [[Project-Page]](https://jjxjiaxue.github.io/DetAny3D/)

- **Perception Encoder: The best visual embeddings are not at the output of the network**  
  [[Paper]](https://arxiv.org/abs/2504.13181)
  [[Code]](https://github.com/facebookresearch/perception_models)

- **PerceptionLM: Open-Access Data and Models for Detailed Visual Understanding**  
  [[Paper]](https://arxiv.org/abs/2504.13180)
  [[Code]](https://github.com/facebookresearch/perception_models)

- **Evolved Hierarchical Masking for Self-Supervised Learning**  
  [[Paper]](https://arxiv.org/abs/2504.09155)

- **I-Con: A Unifying Framework for Representation Learning**  
  [[Paper]](https://arxiv.org/abs/2504.16929)
  [[Project-Page]](https://mhamilton.net/icon)
  [[Code]](https://github.com/ShadeAlsha/ICon)

- **Stronger, Steadier & Superior: Geometric Consistency in Depth VFM Forges Domain Generalized Semantic Segmentation**  
  [[Paper]](https://arxiv.org/abs/2504.12753)
  [[Code]](https://github.com/anonymouse-xzrptkvyqc/DepthForge)

#### 1.2.2 Detection

- **BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers**  
  [[Paper]](https://arxiv.org/abs/2203.17270)
  [[Code]](https://github.com/fundamentalvision/BEVFormer)

- **HV-BEV: Decoupling Horizontal and Vertical Feature Sampling for Multi-View 3D Object Detection**  
  [[Paper]](https://arxiv.org/abs/2412.18884)

- **Open Vocabulary Monocular 3D Object Detection**  
  [[Paper]](https://arxiv.org/abs/2411.16833)
  [[Project-Page]](https://uva-computer-vision-lab.github.io/ovmono3d/)
  [[Code]](https://github.com/UVA-Computer-Vision-Lab/ovmono3d)

- **UniDrive: Towards Universal Driving Perception Across Camera Configurations**  
  [[Paper]](https://arxiv.org/abs/2410.13864)
  [[Project-Page]](https://wzzheng.net/UniDrive/)
  [[Code]](https://github.com/ywyeli/UniDrive)

- **BEVDepth: Acquisition of Reliable Depth for Multi-view 3D Object Detection**  
  [[Paper]](https://arxiv.org/abs/2206.10092)
  [[Code]](https://github.com/Megvii-BaseDetection/BEVDepth)

- **Training an Open-Vocabulary Monocular 3D Object Detection Model without 3D Data**  
  [[Paper]](https://arxiv.org/abs/2411.15657)
  [[Project-Page]](https://ovm3d-det.github.io/)
  [[Code]](https://github.com/LeapLabTHU/OVM3D-Det)

- **Towards Unified 3D Object Detection via Algorithm and Data Unification**  
  [[Paper]](https://arxiv.org/abs/2402.18573)

- **MonoDINO-DETR: Depth-Enhanced Monocular 3D Object Detection Using a Vision Foundation Model**  
  [[Paper]](https://arxiv.org/abs/2502.00315)
  [[Code]](https://github.com/JihyeokKim/MonoDINO-DETR)

#### 1.2.3 Depth Estimation

- **DepthMaster: Taming Diffusion Models for Monocular Depth Estimation**  
  *a single-step diffusion model designed to adapt generative features for the discriminative depth estimation task.*  
  [[Paper]](https://arxiv.org/abs/2501.02576)
  [[Project-Page]](https://indu1ge.github.io/DepthMaster_page/)
  [[Code]](https://github.com/indu1ge/DepthMaster)

- **DepthLab: From Partial to Complete**  
  *a foundation depth inpainting model powered by image diffusion priors.*  
  [[Paper]](https://arxiv.org/abs/2412.18153)
  [[Project-Page]](https://johanan528.github.io/depthlab_web/)
  [[Code]](https://github.com/ant-research/DepthLab)

- **Depth Anything V2**  
  *The version produces much finer and more robust depth predictions compared to Depth Anything V1*  
  [[Paper]](https://arxiv.org/abs/2406.09414)
  [[Project-Page]](https://depth-anything-v2.github.io/)
  [[Code]](https://github.com/DepthAnything/Depth-Anything-V2)

- **Metric3Dv2: A Versatile Monocular Geometric Foundation Model for Zero-shot Metric Depth and Surface Normal Estimation**  
  *a geometric foundation model for zero-shot metric depth and surface normal estimation from a single image, which is crucial for metric 3D recovery.*  
  [[Paper]](https://arxiv.org/abs/2404.15506)
  [[Project-Page]](https://jugghm.github.io/Metric3Dv2/)
  [[Code]](https://github.com/YvanYin/Metric3D)

- **Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data**  
  *a highly practical solution for robust monocular depth estimation.*  
  [[Paper]](https://arxiv.org/abs/2401.10891)
  [[Project-Page]](https://depth-anything.github.io/)
  [[Code]](https://github.com/LiheYoung/Depth-Anything)

- **Metric3D: Towards Zero-shot Metric 3D Prediction from A Single Image**  
  *a strong and robust geometry foundation model for high-quality and zero-shot metric depth and surface normal estimation from a single image.*  
  [[Paper]](https://arxiv.org/abs/2307.10984)
  [[Code]](https://github.com/YvanYin/Metric3D)

- **MVSAnywhere: Zero-Shot Multi-View Stereo**  
  [[Paper]](https://arxiv.org/abs/2503.22430)
  [[Project-Page]](https://nianticlabs.github.io/mvsanywhere/)
  [[Code]](https://github.com/nianticlabs/mvsanywhere)

#### 1.2.4 3D Reconstruction

- **Light3R-SfM: Towards Feed-forward Structure-from-Motion**  
  *Light3R-SfM, a feed-forward, end-to-end learnable framework for efficient large-scale Structure-from-Motion (SfM) from unconstrained image collections.*  
  [[Paper]](https://arxiv.org/abs/2501.14914)

- **Fast3R: Towards 3D Reconstruction of 1000+ Images in One Forward Pass**  
  *Fast 3D Reconstruction (Fast3R), a novel multi-view generalization to DUSt3R that achieves efficient and scalable 3D reconstruction by processing many views in parallel.*  
  [[Paper]](https://arxiv.org/abs/2501.13928)
  [[Project-Page]](https://fast3r-3d.github.io/)
  [[Code]](https://github.com/facebookresearch/fast3r)

- **Stereo4D: Learning How Things Move in 3D from Internet Stereo Videos**  
  *a system for mining high-quality 4D reconstructions from internet stereoscopic, wide-angle videos, fuses and filters the outputs of camera pose estimation, stereo depth estimation, and temporal tracking methods into high-quality dynamic 3D reconstructions.*  
  [[Paper]](https://arxiv.org/abs/2412.09621)
  [[Project-Page]](https://stereo4d.github.io/)
  [[Code]](https://github.com/Stereo4d/stereo4d-code)

- **Grounding Image Matching in 3D with MASt3R**  
  *DUSt3R, a recent and powerful 3D reconstruction framework based on Transformers.*  
  [[Paper]](https://arxiv.org/abs/2406.09756)
  [[Project-Page]](https://europe.naverlabs.com/blog/mast3r-matching-and-stereo-3d-reconstruction/)
  [[Code]](https://github.com/naver/mast3r)

- **Know Your Neighbors: Improving Single-View Reconstruction via Spatial Vision-Language Reasoning**  
  *a novel method for single-view scene reconstruction that reasons about semantic and spatial context to predict each point's density.*  
  [[Paper]](https://arxiv.org/abs/2404.03658)
  [[Project-Page]](https://ruili3.github.io/kyn/)
  [[Code]](https://github.com/ruili3/Know-Your-Neighbors)

- **DUSt3R: Geometric 3D Vision Made Easy**  
  *DUSt3R, a radically novel paradigm for Dense and Unconstrained Stereo 3D Reconstruction of arbitrary image collections, i.e. operating without prior information about camera calibration nor viewpoint poses.*  
  [[Paper]](https://arxiv.org/abs/2312.14132)
  [[Project-Page]](https://dust3r.europe.naverlabs.com/)
  [[Code]](https://github.com/naver/dust3r)

- **Triplane Meets Gaussian Splatting: Fast and Generalizable Single-View 3D Reconstruction with Transformers**  
  [[Paper]](https://arxiv.org/abs/2312.09147)
  [[Project-Page]](https://zouzx.github.io/TriplaneGaussian/)
  [[Code]](https://github.com/VAST-AI-Research/TriplaneGaussian)

- **UniK3D: Universal Camera Monocular 3D Estimation**  
  [[Paper]](https://arxiv.org/abs/2503.16591)
  [[Project-Page]](https://lpiccinelli-eth.github.io/pub/unik3d/)
  [[Code]](https://github.com/lpiccinelli-eth/unik3d) 

- **Visual Geometry Grounded Deep Structure From Motion**  
  [[Paper]](https://arxiv.org/abs/2312.04563)
  [[Project-Page]](https://vggsfm.github.io/)
  [[Code]](https://github.com/facebookresearch/vggsfm)

- **Can Video Diffusion Model Reconstruct 4D Geometry?**  
  [[Paper]](https://arxiv.org/abs/2503.21082)
  [[Project-Page]](https://wayne-mai.github.io/publication/sora3r_arxiv_2025/)

- **St4RTrack: Simultaneous 4D Reconstruction and Tracking in the World**  
  [[Paper]](https://arxiv.org/abs/2504.13152)
  [[Project-Page]](https://st4rtrack.github.io/)

- **POMATO: Marrying Pointmap Matching with Temporal Motion for Dynamic 3D Reconstruction**  
  [[Paper]](https://arxiv.org/abs/2504.05692)
  [[Code]](https://github.com/wyddmw/POMATO)

#### 1.2.5 Scene Generation

- **SceneCraft: Layout-Guided 3D Scene Generation**  
  *SceneCraft, a novel method for generating detailed indoor scenes that adhere to textual descriptions and spatial layout preferences provided by users.*  
  [[Paper]](https://arxiv.org/abs/2410.09049)
  [[Project-Page]](https://orangesodahub.github.io/SceneCraft/)
  [[Code]](https://github.com/OrangeSodahub/SceneCraft/)

- **SceneScript: Reconstructing Scenes With An Autoregressive Structured Language Model**  
  *SceneScript, a method that directly produces full scene models as a sequence of structured language commands using an autoregressive, token-based approach.*  
  [[Paper]](https://arxiv.org/abs/2403.13064)
  [[Project-Page]](https://www.projectaria.com/scenescript/)

- **Holodeck: Language Guided Generation of 3D Embodied AI Environments**  
  *Holodeck, a system that generates 3D environments to match a user-supplied prompt fully automatedly.*  
  [[Paper]](https://arxiv.org/abs/2312.09067)
  [[Project-Page]](https://yueyang1996.github.io/holodeck/)
  [[Code]](https://github.com/allenai/Holodeck)

#### 1.2.6 Spatial Simulation

- **AI2-THOR: An Interactive 3D Environment for Visual AI**  
  *AI2-THOR consists of near photo-realistic 3D indoor scenes, where AI agents can navigate in the scenes and interact with objects to perform tasks.*  
  [[Paper]](https://arxiv.org/abs/1712.05474)
  [[Project-Page]](https://ai2thor.allenai.org/)
  [[Code]](https://github.com/allenai/ai2thor)

- **Habitat: A Platform for Embodied AI Research**  
  *Habitat, a platform for research in embodied artificial intelligence.*  
  [[Paper]](https://arxiv.org/abs/1904.01201)
  [[Project-Page]](https://aihabitat.org/)

- **RoboCasa: Large-Scale Simulation of Everyday Tasks for Generalist Robots**  
  *RoboCasa, a large-scale simulation framework for training generalist robots in everyday environments, features realistic and diverse scenes focusing on kitchen environments.*  
  [[Paper]](https://arxiv.org/abs/2406.02523)
  [[Project-Page]](https://robocasa.ai/)
  [[Code]](https://github.com/robocasa/robocasa)

#### 1.2.7 Point Distribution/Tracking

- **3D Shape Tokenization via Latent Flow Matching**  
  [[Paper]](https://arxiv.org/abs/2412.15618)
  [[Project-Page]](https://machinelearning.apple.com/research/3d-shape-tokenization)

- **Voint Cloud: Multi-View Point Cloud Representation for 3D Understanding**  
  [[Paper]](https://arxiv.org/abs/2111.15363)
  [[Code]](https://github.com/ajhamdi/vointcloud)

- **TAPIP3D: Tracking Any Point in Persistent 3D Geometry**  
  [[Paper]](https://arxiv.org/abs/2504.14717)
  [[Project-Page]](https://tapip3d.github.io/)
  [[Code]](https://github.com/zbw001/TAPIP3D)

#### 1.2.8 Object Positioning

#### 1.2.9 Occupancy Prediction

- **VoxFormer: Sparse Voxel Transformer for Camera-based 3D Semantic Scene Completion**  
  [[Paper]](https://arxiv.org/abs/2302.12251)
  [[Code]](https://github.com/NVlabs/VoxFormer)

- **Tri-Perspective View for Vision-Based 3D Semantic Occupancy Prediction**  
  [[Paper]](https://arxiv.org/abs/2302.07817)
  [[Project-Page]](https://wzzheng.net/TPVFormer/)
  [[Code]](https://github.com/wzzheng/TPVFormer)

- **OctreeOcc: Efficient and Multi-Granularity Occupancy Prediction Using Octree Queries**  
  [[Paper]](https://arxiv.org/abs/2312.03774)
  [[Code]](https://github.com/4DVLab/OctreeOcc)

- **OccFormer: Dual-path Transformer for Vision-based 3D Semantic Occupancy Prediction**  
  [[Paper]](https://arxiv.org/abs/2304.05316)
  [[Code]](https://github.com/zhangyp15/OccFormer)

- **SparseOcc: Rethinking Sparse Latent Representation for Vision-Based Semantic Occupancy Prediction**  
  [[Paper]](https://arxiv.org/abs/2404.09502)
  [[Project-Page]](https://pintang1999.github.io/sparseocc.html)
  [[Code]](https://github.com/VISION-SJTU/SparseOcc)

- **LowRankOcc: Tensor Decomposition and Low-Rank Recovery for Vision-Based 3D Semantic Occupancy Prediction**  
  [[Paper]](http://openaccess.thecvf.com/content/CVPR2024/papers/Zhao_LowRankOcc_Tensor_Decomposition_and_Low-Rank_Recovery_for_Vision-based_3D_Semantic_CVPR_2024_paper.pdf)

- **SelfOcc: Self-Supervised Vision-Based 3D Occupancy Prediction**  
  [[Paper]](https://arxiv.org/abs/2311.12754)
  [[Project-Page]](https://huang-yh.github.io/SelfOcc/)
  [[Code]](https://github.com/huang-yh/SelfOcc)

- **Monocular Occupancy Prediction for Scalable Indoor Scenes**  
  [[Paper]](https://arxiv.org/abs/2407.11730)
  [[Project-Page]](https://hongxiaoy.github.io/ISO/)
  [[Code]](https://github.com/hongxiaoy/ISO)

- **OPUS: Occupancy Prediction Using a Sparse Set**  
  [[Paper]](https://arxiv.org/abs/2409.09350)
  [[Code]](https://github.com/jbwang1997/OPUS)

- **Fast Occupancy Network**  
  [[Paper]](https://arxiv.org/abs/2412.07163)

- **Lightweight Spatial Embedding for Vision-based 3D Occupancy Prediction**  
  [[Paper]](https://arxiv.org/abs/2412.05976)

- **ViPOcc: Leveraging Visual Priors from Vision Foundation Models for Single-View 3D Occupancy Prediction**  
  [[Paper]](https://arxiv.org/abs/2412.11210)
  [[Project-Page]](https://mias.group/ViPOcc/)
  [[Code]](https://github.com/fengyi233/ViPOcc)

- **LOMA: Language-assisted Semantic Occupancy Network via Triplane Mamba**  
  [[Paper]](https://arxiv.org/abs/2412.08388)

- **Semi-Supervised Vision-Centric 3D Occupancy World Model for Autonomous Driving**  
  [[Paper]](https://arxiv.org/abs/2502.07309)
  [[Code]](https://github.com/getterupper/PreWorld)

- **SliceOcc: Indoor 3D Semantic Occupancy Prediction with Vertical Slice Representation**  
  [[Paper]](https://arxiv.org/abs/2501.16684)
  [[Code]](https://github.com/NorthSummer/SliceOcc)

#### 1.2.10 Scene Completion

- **Semantic Scene Completion from a Single Depth Image**  
  [[Paper]](https://arxiv.org/abs/1611.08974)

- **MonoScene: Monocular 3D Semantic Scene Completion**  
  [[Paper]](https://arxiv.org/abs/2112.00726)
  [[Project-Page]](https://astra-vision.github.io/MonoScene/)
  [[Code]](https://github.com/astra-vision/MonoScene)

- **OccDepth: A Depth-Aware Method for 3D Semantic Scene Completion**  
  [[Paper]](https://arxiv.org/abs/2302.13540)
  [[Code]](https://github.com/megvii-research/OccDepth)

- **NDC-Scene: Boost Monocular 3D Semantic Scene Completion in Normalized Device Coordinates Space**  
  [[Paper]](https://arxiv.org/abs/2309.14616)
  [[Project-Page]](https://jiawei-yao0812.github.io/NDC-Scene/)
  [[Code]](https://github.com/Jiawei-Yao0812/NDCScene)

- **Not All Voxels Are Equal: Hardness-Aware Semantic Scene Completion with Self-Distillation**  
  [[Paper]](https://arxiv.org/abs/2404.11958)
  [[Code]](https://github.com/songw-zju/HASSC)

- **Symphonize 3D Semantic Scene Completion with Contextual Instance Queries**  
  [[Paper]](https://arxiv.org/abs/2306.15670)
  [[Code]](https://github.com/hustvl/Symphonies)

- **SGFormer: Satellite-Ground Fusion for 3D Semantic Scene Completion**  
  [[Paper]](https://arxiv.org/abs/2503.16825)
  [[Project-Page]](https://zju3dv.github.io/sgformer/)
  [[Code]](https://github.com/gxytcrc/SGFormer)

#### 1.2.11 Segmentation

- **OnlineAnySeg: Online Zero-Shot 3D Segmentation by Visual Foundation Model Guided 2D Mask Merging**  
  [[Paper]](https://arxiv.org/abs/2503.01309)
  [[Project]](https://yjtang249.github.io/OnlineAnySeg/)
  [[Code]](https://github.com/yjtang249/OnlineAnySeg)

#### 1.2.12 Pose Estimation

- **PoseDiffusion: Solving Pose Estimation via Diffusion-aided Bundle Adjustment**  
  [[Paper]](https://arxiv.org/abs/2306.15667)
  [[Project-Page]](https://posediffusion.github.io/)
  [[Code]](https://github.com/facebookresearch/PoseDiffusion)

- **RelPose++: Recovering 6D Poses from Sparse-view Observations**  
  [[Paper]](https://arxiv.org/abs/2305.04926)
  [[Project-Page]](https://amyxlase.github.io/relpose-plus-plus/)
  [[Code]](https://github.com/amyxlase/relpose-plus-plus)

- **Visual Geometry Grounded Deep Structure From Motion**  
  [[Paper]](https://arxiv.org/abs/2312.04563)
  [[Project-Page]](https://vggsfm.github.io/)
  [[Code]](https://github.com/facebookresearch/vggsfm)

- **Relative Pose Estimation through Affine Corrections of Monocular Depth Priors**  
  [[Paper]](https://arxiv.org/abs/2501.05446)
  [[Code]](https://github.com/MarkYu98/madpose)

### 1.3 Multi-modal

#### 1.3.1 Understanding

- **Seeing from Another Perspective: Evaluating Multi-View Understanding in MLLMs**  
  *show that MLLMs are particularly underperforming under two aspects: (1) cross-view correspondence for partially occluded views and (2) establishing the coarse camera poses.*  
  [[Paper]](https://arxiv.org/abs/2504.15280)
  [[Project-Page]](https://danielchyeh.github.io/All-Angles-Bench/)
  [[Code]](https://github.com/Chenyu-Wang567/All-Angles-Bench)

- **Thinking in Space: How Multimodal Large Language Models See, Remember, and Recall Spaces**  
  *probe models to express how they think in space both linguistically and visually and find that while spatial reasoning capabilities remain the primary bottleneck for MLLMs to reach higher benchmark performance, local world models and spatial awareness do emerge within these models.*  
  [[Paper]](https://arxiv.org/abs/2412.14171)
  [[Project-Page]](https://vision-x-nyu.github.io/thinking-in-space.github.io/)
  [[Code]](https://github.com/vision-x-nyu/thinking-in-space)

- **SAT: Dynamic Spatial Aptitude Training for Multimodal Language Models**  
  *Leveraging our SAT datasets and 6 existing static spatial benchmarks, we systematically investigate what improves both static and dynamic spatial awareness.*  
  [[Paper]](https://arijitray.com/SAT/)
  [[Project-Page]](https://arxiv.org/abs/2412.07755)

- **ProVision: Programmatically Scaling Vision-centric Instruction Data for Multimodal Language Models**  
  *a programmatic approach that employs scene graphs as symbolic representations of images and human-written programs to systematically synthesize vision-centric instruction data.*  
  [[Paper]](https://arxiv.org/abs/2412.07012)
  [[Code]](https://github.com/JieyuZ2/ProVision)

- **TIPS: Text-Image Pretraining with Spatial awareness**  
  *close the gap between image-text and self-supervised learning, by proposing a novel general-purpose image-text model, which can be effectively used off the shelf for dense and global vision tasks.*  
  [[Paper]](https://arxiv.org/abs/2410.16512)
  [[Project-Page]](https://gdm-tips.github.io/)
  [[Code]](https://github.com/google-deepmind/tips)

- **Cambrian-1: Cambrian-1: A Fully Open, Vision-Centric Exploration of Multimodal LLMs**  
  *Spatial Vision Aggregator (SVA), a dynamic and spatially-aware connector that integrates high-resolution vision features with LLMs while reducing the number of tokens.*  
  [[Paper]](https://arxiv.org/abs/2406.16860)
  [[Project-Page]](https://cambrian-mllm.github.io/)
  [[Code]](https://github.com/cambrian-mllm/cambrian)

- **SpatialBot: Precise Spatial Understanding with Vision Language Models**  
  *SpatialBot for better spatial understanding by feeding both RGB and depth images.*  
  [[Paper]](https://arxiv.org/abs/2406.13642)
  [[Code]](https://github.com/BAAI-DCAI/SpatialBot)

- **SpatialRGPT: Grounded Spatial Reasoning in Vision Language Models**  
  *(1) a data curation pipeline that enables effective learning of regional representation from 3D scene graphs, and (2) a flexible plugin module for integrating depth information into the visual encoder of existing VLMs.*  
  [[Paper]](https://arxiv.org/abs/2406.01584)
  [[Project-Page]](https://www.anjiecheng.me/SpatialRGPT)
  [[Code]](https://github.com/AnjieCheng/SpatialRGPT)

- **Learning to Localize Objects Improves Spatial Reasoning in Visual-LLMs**  
  *discover optimal coordinate representations, data-efficient instruction fine-tuning objectives, and pseudo-data generation strategies that lead to improved spatial awareness in V-LLMs.*  
  [[Paper]](https://arxiv.org/abs/2404.07449)

- **SpatialVLM: Endowing Vision-Language Models with Spatial Reasoning Capabilities**  
  *an automatic 3D spatial VQA data generation framework that scales up to 2 billion VQA examples on 10 million real-world images. We then investigate various factors in the training recipe, including data quality, training pipeline, and VLM architecture.*  
  [[Paper]](https://arxiv.org/abs/2401.12168)
  [[Project-Page]](https://spatial-vlm.github.io/)
  [[Code]](https://github.com/remyxai/VQASynth)

#### 1.3.2 Spatial Reasoning

- **Mind the Gap: Benchmarking Spatial Reasoning in Vision-Language Models**  
  *delineates the core elements of spatial reasoning: spatial relations, orientation and navigation, mental rotation, and spatial visualization, and then assesses the performance of these models in both synthetic and real-world images, bridging controlled and naturalistic contexts.*  
  [[Paper]](https://arxiv.org/abs/2503.19707)
  [[Code]](https://github.com/stogiannidis/srbench)

- **Open3DVQA: A Benchmark for Comprehensive Spatial Reasoning with Multimodal Large Language Model in Open Space**  
  *a novel benchmark, Open3DVQA, to comprehensively evaluate the spatial reasoning capacities of current state-of-the-art (SOTA) foundation models in open 3D space.*  
  [[Paper]](https://www.arxiv.org/abs/2503.11094)
  [[Code]](https://github.com/WeichenZh/Open3DVQA)

#### 1.3.3 Navigation

- **Navigation Instruction Generation with BEV Perception and Large Language Models**  
  [[Paper]](https://arxiv.org/abs/2407.15087)
  [[Code]](https://github.com/FanScy/BEVInstructor)

- **RoomTour3D: Geometry-Aware Video-Instruction Tuning for Embodied Navigation**  
  [[Paper]](https://arxiv.org/abs/2412.08591)
  [[Project-Page]](https://roomtour3d.github.io/)
  [[Code]](https://github.com/roomtour3d/roomtour3d-NaviLLM)

- **FrontierNet: Learning Visual Cues to Explore**  
  [[Paper]](https://arxiv.org/abs/2501.04597)
  [[Code]](https://github.com/cvg/FrontierNet)

- **MapNav: A Novel Memory Representation via Annotated Semantic Maps for VLM-based Vision-and-Language Navigation**  
  [[Paper]](https://arxiv.org/abs/2502.13451)

- ****  
  [[Paper]]()
  [[Project-Page]]()
  [[Code]]()

- ****  
  [[Paper]]()
  [[Project-Page]]()
  [[Code]]()

- ****  
  [[Paper]]()
  [[Project-Page]]()
  [[Code]]()

- ****  
  [[Paper]]()
  [[Project-Page]]()
  [[Code]]()

### 1.4 Others (radar/GPS etc.)

## 2. Datasets and Benchmarks

**<div style="text-align: center;">2025</div>**

- **Seeing from Another Perspective: Evaluating Multi-View Understanding in MLLMs**  
  [[Paper]](https://arxiv.org/abs/2504.15280)
  [[Project-Page]](https://danielchyeh.github.io/All-Angles-Bench/)
  [[Dataset-All-Angles-Bench]](https://huggingface.co/datasets/ch-chenyu/All-Angles-Bench)

- **Spatial-R1: Enhancing MLLMs in Video Spatial Reasoning**  
  [[Paper]](https://arxiv.org/abs/2504.01805)
  [[Project-Page]](https://github.com/OuyangKun10/Spatial-R1)
  [[Dataset-Spatial-R1-151k]](https://huggingface.co/datasets/RUBBISHLIKE/Spatial-R1-151k)

- **Improved Visual-Spatial Reasoning via R1-Zero-Like Training**  
  [[Paper]](https://arxiv.org/abs/2504.00883)
  [[Project-Page]](https://github.com/zhijie-group/R1-Zero-VSI)

- **From Flatland to Space: Teaching Vision-Language Models to Perceive and Reason in 3D**  
  [[Paper]](https://arxiv.org/pdf/2503.22976)
  [[Project-Page]](https://fudan-zvg.github.io/spar/)
  [[Dataset-SPAR-7M]](https://huggingface.co/datasets/jasonzhango/SPAR-7M)
  [[Dataset-SPAR-Bench]](https://huggingface.co/datasets/jasonzhango/SPAR-Bench)

- **Gemini Robotics: Bringing AI into the Physical World**  
  [[Paper]](https://arxiv.org/abs/2503.20020)
  [[Dataset-PhysicalAI-Robotics-GR00T-X-Embodiment-Sim]](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim)

- **Mind the Gap: Benchmarking Spatial Reasoning in Vision-Language Models**  
  [[Paper]](https://arxiv.org/abs/2503.19707)
  [[Project-Page]](https://github.com/stogiannidis/srbench)
  [[Dataset-srbench]](https://github.com/stogiannidis/srbench)

- **Open3DVQA: A Benchmark for Comprehensive Spatial Reasoning with Multimodal Large Language Model in Open Space**  
  [[Paper]](https://www.arxiv.org/abs/2503.11094)
  [[Project-Page]](https://github.com/WeichenZh/Open3DVQA)
  [[Dataset-Open3DVQA]](https://github.com/WeichenZh/Open3DVQA)

- **Spatial457: A Diagnostic Benchmark for 6D Spatial Reasoning of Large Multimodal Models**  
  [[Paper]](https://arxiv.org/abs/2502.08636)
  [[Project-Page]](https://xingruiwang.github.io/projects/Spatial457/)
  [[Dataset-Spatial457]](https://huggingface.co/datasets/RyanWW/Spatial457)

- **iVISPAR — An Interactive Visual-Spatial Reasoning Benchmark for VLMs**  
  [[Paper]](https://arxiv.org/abs/2502.03214)
  [[Project-Page]](https://ivispar.ai/)
  [[Dataset-iVISPAR]](https://github.com/SharkyBamboozle/iVISPAR)

- **SoFar: Language-Grounded Orientation Bridges Spatial Reasoning and Object Manipulation**  
  [[Paper]](https://arxiv.org/abs/2502.13143)
  [[Project-Page]](https://qizekun.github.io/sofar/)
  [[Dataset-OrienText300K]](https://huggingface.co/datasets/qizekun/OrienText300K)
  [[Dataset-6DoF-SpatialBench]](https://huggingface.co/datasets/qizekun/6DoF-SpatialBench)

- **PhysBench: Benchmarking and Enhancing Vision-Language Models for Physical World Understanding**  
  [[Paper]](https://arxiv.org/abs/2501.16411)
  [[Project-Page]](https://physbench.github.io/)
  [[Dataset-PhysBench]](https://huggingface.co/datasets/USC-GVL/PhysBench)

---
**<div style="text-align: center;">2024</div>**

- **Do Multimodal Language Models Really Understand Direction? A Benchmark for Compass Direction Reasoning**  
  [[Paper]](https://arxiv.org/abs/2412.16599)

- **Thinking in Space: How Multimodal Large Language Models See, Remember, and Recall Spaces**  
  [[Paper]](https://arxiv.org/abs/2412.14171)
  [[Project-Page]](https://vision-x-nyu.github.io/thinking-in-space.github.io/)
  [[Dataset-VSI-Bench]](https://huggingface.co/datasets/nyu-visionx/VSI-Bench)

- **SPHERE: Unveiling Spatial Blind Spots in Vision-Language Models Through Hierarchical Evaluation**  
  [[Paper]](https://arxiv.org/abs/2412.12693)
  [[Project-Page]](https://github.com/zwenyu/SPHERE-VLM)
  [[Dataset-SPHERE-VLM]](https://huggingface.co/datasets/wei2912/SPHERE-VLM)

- **Stereo4D: Learning How Things Move in 3D from Internet Stereo Videos**  
  [[Paper]](https://arxiv.org/abs/2412.09621)
  [[Project-Page]](https://stereo4d.github.io/)
  [[Dataset-Stereo4D]](https://console.cloud.google.com/storage/browser/stereo4d/)

- **3DSRBench: A Comprehensive 3D Spatial Reasoning Benchmark**  
  [[Paper]](https://arxiv.org/abs/2412.07825)
  [[Project-Page]](https://3dsrbench.github.io/)
  [[Dataset-3DSRBench]](https://huggingface.co/datasets/ccvl/3DSRBench)

- **SAT: Dynamic Spatial Aptitude Training for Multimodal Language Models**  
  [[Paper]](https://arxiv.org/abs/2412.07755)
  [[Project-Page]](https://arijitray.com/SAT/)
  [[Dataset-SAT]](https://huggingface.co/datasets/array/SAT)

- **ProVision: Programmatically Scaling Vision-centric Instruction Data for Multimodal Language Models**  
  [[Paper]](https://arxiv.org/abs/2412.07012)
  [[Project-Page]](https://github.com/JieyuZ2/ProVision)
  [[Dataset-ProVision-10M]](https://huggingface.co/datasets/Salesforce/ProVision-10M)

- **Video-3D LLM: Learning Position-Aware Video Representation for 3D Scene Understanding**  
  [[Paper]](https://arxiv.org/abs/2412.00493)
  [[Project-Page]](https://github.com/LaVi-Lab/Video-3D-LLM)
  [[Dataset-Video-3D-LLM_data]](https://huggingface.co/datasets/zd11024/Video-3D-LLM_data)

- **RoboSpatial: Teaching Spatial Understanding to 2D and 3D Vision-Language Models for Robotics**  
  [[Paper]](https://arxiv.org/abs/2411.16537)
  [[Project-Page]](https://chanh.ee/RoboSpatial/)
  [[Dataset-RoboSpatial-Home]](https://huggingface.co/datasets/chanhee-luke/RoboSpatial-Home)

- **An Empirical Analysis on Spatial Reasoning Capabilities of Large Multimodal Models**  
  [[Paper]](https://arxiv.org/abs/2411.06048)
  [[Project-Page]](https://github.com/FatemehShiri/Spatial-MM)
  [[Dataset-Spatial-MM]](https://github.com/FatemehShiri/Spatial-MM)

- **Do Vision-Language Models Represent Space and How? Evaluating Spatial Frame of Reference Under Ambiguities**  
  [[Paper]](https://arxiv.org/abs/2410.17385)
  [[Project-Page]](https://spatial-comfort.github.io/)
  [[Dataset-COMFORT]](https://huggingface.co/datasets/sled-umich/COMFORT)

- **MEGA-Bench: Scaling Multimodal Evaluation to over 500 Real-World Tasks**  
  [[Paper]](https://arxiv.org/abs/2410.10563)
  [[Project-Page]](https://tiger-ai-lab.github.io/MEGA-Bench/)
  [[Dataset-MEGA-Bench]](https://huggingface.co/datasets/TIGER-Lab/MEGA-Bench)

- **Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Vision-Language Models**  
  [[Paper]](https://arxiv.org/abs/2409.17146)
  [[Project-Page]](https://allenai.org/blog/molmo)
  [[Dataset-PixMo]](https://huggingface.co/collections/allenai/pixmo-674746ea613028006285687b)

- **Can Vision Language Models Learn from Visual Demonstrations of Ambiguous Spatial Reasoning?**  
  [[Paper]](https://arxiv.org/abs/2409.17080)
  [[Project-Page]](https://github.com/groundlight/vlm-visual-demonstrations)

- **Reasoning Paths with Reference Objects Elicit Quantitative Spatial Reasoning in Large Vision-Language Models**  
  [[Paper]](https://arxiv.org/abs/2409.09788)
  [[Project-Page]](https://andrewliao11.github.io/spatial_prompt/)
  [[Dataset-Q-Spatial-Bench]](https://huggingface.co/datasets/andrewliao11/Q-Spatial-Bench)

- **GRASP: A Grid-Based Benchmark for Evaluating Commonsense Spatial Reasoning**  
  [[Paper]](https://arxiv.org/abs/2407.01892)
  [[Project-Page]](https://github.com/jasontangzs0/GRASP)
  [[Dataset-GRASP]](https://github.com/jasontangzs0/GRASP)

- **Cambrian-1: Cambrian-1: A Fully Open, Vision-Centric Exploration of Multimodal LLMs**  
  [[Paper]](https://arxiv.org/abs/2406.16860)
  [[Project-Page]](https://cambrian-mllm.github.io/)
  [[Dataset-CV-Bench]](https://huggingface.co/datasets/nyu-visionx/CV-Bench)
  [[Dataset-Cambrian-10M]](https://huggingface.co/datasets/nyu-visionx/Cambrian-10M)

- **Is A Picture Worth A Thousand Words? Delving Into Spatial Reasoning for Vision Language Models**  
  [[Paper]](https://arxiv.org/abs/2406.14852)
  [[Project-Page]](https://spatialeval.github.io/)
  [[Dataset-SpatialEval]](https://huggingface.co/datasets/MilaWang/SpatialEval)

- **GSR-Bench: A Benchmark for Grounded Spatial Reasoning Evaluation via Multimodal LLMs**  
  [[Paper]](https://arxiv.org/abs/2406.13246)

- **RoboPoint: A Vision-Language Model for Spatial Affordance Prediction for Robotics**  
  [[Paper]](https://arxiv.org/abs/2406.10721)
  [[Project-Page]](https://robo-point.github.io/)
  [[Dataset-ReboPoint-Data]](https://huggingface.co/datasets/wentao-yuan/robopoint-data)

- **ImageNet3D: Towards General-Purpose Object-Level 3D Understanding**  
  [[Paper]](https://arxiv.org/abs/2406.09613)
  [[Project-Page]](https://imagenet3d.github.io/)
  [[Dataset-ImageNet3D]](https://huggingface.co/datasets/ccvl/ImageNet3D)

- **EmbSpatial-Bench: Benchmarking Spatial Understanding for Embodied Tasks with Large Vision-Language Models**  
  [[Paper]](https://arxiv.org/abs/2406.05756)
  [[Project-Page]](https://github.com/mengfeidu/EmbSpatial-Bench)
  [[Dataset-EmbSpatial-Bench]](https://huggingface.co/datasets/Phineas476/EmbSpatial-Bench)

- **SpaRC and SpaRP: Spatial Reasoning Characterization and Path Generation for Understanding Spatial Reasoning Capability of Large Language Models**  
  [[Paper]](https://arxiv.org/abs/2406.04566)
  [[Project-Page]](https://github.com/UKPLab/acl2024-sparc-and-sparp)
  [[Dataset-sparp]](https://huggingface.co/datasets/UKPLab/sparp)

- **TopViewRS: Vision-Language Models as Top-View Spatial Reasoners**  
  [[Paper]](https://arxiv.org/abs/2406.02537)
  [[Project-Page]](https://topviewrs.github.io/)
  [[Dataset-topviewrs]](https://huggingface.co/datasets/chengzu/topviewrs)

- **SpatialRGPT: Grounded Spatial Reasoning in Vision Language Models**  
  [[Paper]](https://arxiv.org/abs/2406.01584)
  [[Project-Page]](https://www.anjiecheng.me/SpatialRGPT)
  [[Dataset-SpatialRGPT-Bench]](https://huggingface.co/datasets/a8cheng/SpatialRGPT-Bench)
  [Dataset-OpenSpatialDataset](https://huggingface.co/datasets/a8cheng/OpenSpatialDataset)

- **Compositional 4D Dynamic Scenes Understanding with Physics Priors for Video Question Answering**  
  [[Paper]](https://arxiv.org/abs/2406.00622)
  [[Project-Page]](https://xingruiwang.github.io/projects/DynSuperCLEVR/)
  [[Dataset-DynSuperCLEVR]](https://github.com/XingruiWang/DynSuperCLEVR)

- **Reframing Spatial Reasoning Evaluation in Language Models: A Real-World Simulation Benchmark for Qualitative Reasoning**  
  [[Paper]](https://arxiv.org/abs/2405.15064)
  [[Project-Page]](https://github.com/Fangjun-Li/RoomSpace)
  [[Dataset-RoomSpace]](https://huggingface.co/datasets/Fangjun/RoomSpace)
  
- **BLINK: Multimodal Large Language Models Can See but Not Perceive**  
  [[Paper]](https://arxiv.org/abs/2404.12390)
  [[Project-Page]](https://zeyofu.github.io/blink/)
  [[Dataset-BLINK]](https://huggingface.co/datasets/BLINK-Benchmark/BLINK)

- **Proximity QA: Unleashing the Power of Multi-Modal Large Language Models for Spatial Proximity Analysis**  
  [[Paper]](https://arxiv.org/abs/2401.17862)
  [[Project-Page]](https://github.com/NorthSummer/ProximityQA)
  [[Dataset-ProximityQA]](https://huggingface.co/Electronics/ProximityQA)

- **R2D3:ImpartingSpatial Reasoning by Reconstructing 3D Scenes from 2D Images**  
  [[Paper]](https://openreview.net/pdf?id=Ku4lylDpjq)
  [[Dataset-r2d3_data]](https://huggingface.co/datasets/array/r2d3_data)

- **Holistic Autonomous Driving Understanding by Bird's-Eye-View Injected Multi-Modal Large Models**  
  [[Paper]](https://arxiv.org/abs/2401.00988)
  [[Project-Page]](https://github.com/xmed-lab/NuInstruct)
  [[Dataset-NuInstruct]](https://github.com/xmed-lab/NuInstruct)

---
**<div style="text-align: center;">2023</div>**

- **3D-Aware Visual Question Answering about Parts, Poses and Occlusions**  
  [[Paper]](https://arxiv.org/abs/2310.17914)
  [[Project-Page]](https://github.com/XingruiWang/3D-Aware-VQA)
  [[Dataset-superclevr-3D-question]](https://github.com/XingruiWang/superclevr-3D-question)

- **Evaluating Spatial Understanding of Large Language Models**  
  *design natural-language navigation tasks and evaluate the ability of LLMs*  
  [[Paper]](https://arxiv.org/abs/2310.14540)
  [[Dataset-SpatialEvalLLM]](https://huggingface.co/datasets/yyamada/SpatialEvalLLM/tree/main)

- **Open X-Embodiment: Robotic Learning Datasets and RT-X Models**  
  [[Paper]](https://arxiv.org/abs/2310.08864)
  [[Project-Page]](https://robotics-transformer-x.github.io/)
  [[Dataset-Open X-Embodiment]](https://docs.google.com/spreadsheets/d/1rPBD77tk60AEIGZrGSODwyyzs5FgCU9Uz3h-3_t2A9g/edit?gid=0#gid=0)

---
**<div style="text-align: center;">2022</div>**

- **Things not Written in Text: Exploring Spatial Commonsense from Visual Signals**  
  [[Paper]](https://arxiv.org/abs/2203.08075)
  [[Project-Page]](https://github.com/xxxiaol/spatial-commonsense)
  [[Dataset-spatial-commonsense]](https://github.com/xxxiaol/spatial-commonsense)

---
**<div style="text-align: center;">2020</div>**

- **SPARE3D: A Dataset for SPAtial REasoning on Three-View Line Drawings**  
  [[Paper]](https://arxiv.org/abs/2003.14034)
  [[Project-Page]](https://github.com/ai4ce/SPARE3D)
  [[Dataset-SPARE3D]](https://drive.google.com/drive/folders/1Mi2KZyKAlUOGYRQTDz3E5nhiXY5GhUB2)

---
**<div style="text-align: center;">2019</div>**

- **SpatialSense: An Adversarially Crowdsourced Benchmark for Spatial Relation Recognition**  
  *a dataset specializing in spatial relation recognition which captures a broad spectrum of such challenges, allowing for proper benchmarking of computer vision techniques.*  
  [[Paper]](https://arxiv.org/abs/1908.02660)
  [[Project-Page]](https://github.com/princeton-vl/SpatialSense)
  [[Dataset-SpatialSense]](https://zenodo.org/records/8104370)

## 3. Spatial Intelligence Methods

### 3.1 old/mathematical/rule-based

#### 3.1.1 Neural Radiance Fields

##### 3.1.1.1 Survey

- **NeRF in Robotics: A Survey**  
  [[Paper]](https://arxiv.org/abs/2405.01333)
  [[Project-Page]]()
  [[Code]]()

##### 3.1.1.2 Basis

###### 3.1.1.2.1 Neural Rendering Pretraining

- **NeRF-MAE: Masked AutoEncoders for Self-Supervised 3D Representation Learning for Neural Radiance Fields**  
  [[Paper]](https://arxiv.org/abs/2404.01300)
  [[Project-Page]](https://nerf-mae.github.io/)
  [[Code]](https://github.com/zubair-irshad/NeRF-MAE)

- **UniPAD: A Universal Pre-training Paradigm for Autonomous Driving**  
  [[Paper]](https://arxiv.org/abs/2310.08370)
  [[Code]](https://github.com/Nightmare-n/UniPAD)

- **Ponder: Point Cloud Pre-training via Neural Rendering**  
  [[Paper]](https://arxiv.org/abs/2301.00157)

- **PonderV2: Pave the Way for 3D Foundation Model with A Universal Pre-training Paradigm**  
  [[Paper]](https://arxiv.org/abs/2310.08586)
  [[Code]](https://github.com/OpenGVLab/PonderV2)

- **SPA: 3D Spatial-Awareness Enables Effective Embodied Representation**  
  [[Paper]](https://arxiv.org/abs/2410.08208)
  [[Project-Page]](https://haoyizhu.github.io/spa/)
  [[Code]](https://github.com/HaoyiZhu/SPA)

###### 3.1.1.2.2 Signed Distance Function

- **NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction**  
  [[Paper]](https://arxiv.org/abs/2106.10689)

- **NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction**  
  [[Paper]](https://arxiv.org/abs/2106.10689)

##### 3.1.1.3 Downstream Tasks

###### 3.1.1.3.1 Detection

- **NeRF-Det++: Incorporating Semantic Cues and Perspective-aware Depth Supervision for Indoor Multi-View 3D Detection**  
  [[Paper]](https://arxiv.org/abs/2402.14464)
  [[Code]](https://github.com/mrsempress/NeRF-Detplusplus)

###### 3.1.1.3.2 Segmentation

- **Decomposing NeRF for Editing via Feature Field Distillation**  
  [[Paper]](https://arxiv.org/abs/2205.15585)
  [[Project-Page]](https://pfnet-research.github.io/distilled-feature-fields/)
  [[Code]](https://github.com/pfnet-research/distilled-feature-fields)

- **3D Concept Learning and Reasoning from Multi-View Images**  
  [[Paper]](https://arxiv.org/abs/2303.11327)
  [[Code]](https://github.com/evelinehong/3D-CLR-Official)

- **Learning 3D Scene Priors with 2D Supervision**  
  [[Paper]](https://arxiv.org/abs/2211.14157)
  [[Project-Page]](https://yinyunie.github.io/sceneprior-page/)
  [[Code]](https://github.com/yinyunie/ScenePriors)

- **LERF: Language Embedded Radiance Fields**  
  [[Paper]](https://arxiv.org/abs/2303.09553)
  [[Project-Page]](https://www.lerf.io/)
  [[Code]](https://github.com/kerrj/lerf)

- **OV-NeRF: Open-vocabulary Neural Radiance Fields with Vision and Language Foundation Models for 3D Semantic Understanding**  
  [[Paper]](https://arxiv.org/abs/2402.04648)
  [[Code]](https://github.com/pcl3dv/OV-NeRF)

- **ConDense: Consistent 2D/3D Pre-training for Dense and Sparse Features from Multi-View Images**  
  [[Paper]](https://arxiv.org/abs/2408.17027)

- **O2V-Mapping: Online Open-Vocabulary Mapping with Neural Implicit Representation**  
  [[Paper]](https://arxiv.org/abs/2404.06836)
  [[Code]](https://github.com/Fudan-MAGIC-Lab/O2Vmapping)

###### 3.1.1.3.3 Navigation

- **Lookahead Exploration with Neural Radiance Representation for Continuous Vision-Language Navigation**  
  [[Paper]](https://arxiv.org/abs/2404.01943)
  [[Code]](https://github.com/MrZihan/HNR-VLN)

#### 3.1.2 Gaussian Splatting

##### 3.1.2.1 Survey

- **A Survey on 3D Gaussian Splatting**  
  [[Paper]](https://arxiv.org/abs/2401.03890)
  [[Project-Page]](https://github.com/guikunchen/Awesome3DGS)

- **3D Gaussian Splatting in Robotics: A Survey**  
  [[Paper]](https://arxiv.org/abs/2410.12262)
  [[Project-Page]](https://github.com/zstsandy/Awesome-3D-Gaussian-Splatting-in-Robotics)

##### 3.1.2.2 Basis

###### 3.1.2.2.1 Optimized Rendering

- **2D Gaussian Splatting for Geometrically Accurate Radiance Fields**  
  [[Paper]](https://arxiv.org/abs/2403.17888)
  [[Project-Page]](https://surfsplatting.github.io/)
  [[Code]](https://github.com/hbb1/2d-gaussian-splatting)

- **pixelSplat: 3D Gaussian Splats from Image Pairs for Scalable Generalizable 3D Reconstruction**  
  [[Paper]](https://arxiv.org/abs/2312.12337)
  [[Project-Page]](https://davidcharatan.com/pixelsplat/)
  [[Code]](https://github.com/dcharatan/pixelsplat)

- **Scaffold-GS: Structured 3D Gaussians for View-Adaptive Rendering**  
  [[Paper]](https://arxiv.org/abs/2312.00109)
  [[Project-Page]](https://city-super.github.io/scaffold-gs/)
  [[Code]](https://github.com/city-super/Scaffold-GS)

- **Octree-GS: Towards Consistent Real-time Rendering with LOD-Structured 3D Gaussians**  
  [[Paper]](https://arxiv.org/abs/2403.17898)
  [[Project-Page]](https://city-super.github.io/octree-gs/)
  [[Code]](https://github.com/city-super/Octree-GS)

- **Sparse Voxels Rasterization: Real-time High-fidelity Radiance Field Rendering**  
  [[Paper]](https://arxiv.org/abs/2412.04459)
  [[Project-Page]](https://svraster.github.io/)
  [[Code]](https://github.com/NVlabs/svraster)

- **Latent Radiance Fields with 3D-aware 2D Representations**  
  [[Paper]](https://arxiv.org/abs/2502.09613)
  [[Project-Page]](https://latent-radiance-field.github.io/LRF/)
  [[Code]](https://github.com/ChaoyiZh/latent-radiance-field)

- **NeuralGS: Bridging Neural Fields and 3D Gaussian Splatting for Compact 3D Representations**  
  [[Paper]](https://arxiv.org/abs/2503.23162)
  [[Project-Page]](https://pku-yuangroup.github.io/NeuralGS/)
  [[Code]](https://github.com/PKU-YuanGroup/NeuralGS)

###### 3.1.2.2.2 Geometric and Material

- **SuGaR: Surface-Aligned Gaussian Splatting for Efficient 3D Mesh Reconstruction and High-Quality Mesh Rendering**  
  [[Paper]](https://arxiv.org/abs/2311.12775)
  [[Project-Page]](https://anttwo.github.io/sugar/)
  [[Code]](https://github.com/Anttwo/SuGaR)

- **NeuSG: Neural Implicit Surface Reconstruction with 3D Gaussian Splatting Guidance**  
  [[Paper]](https://arxiv.org/abs/2312.00846)

- **GSDF: 3DGS Meets SDF for Improved Rendering and Reconstruction**  
  [[Paper]](https://arxiv.org/abs/2403.16964)
  [[Project-Page]](https://city-super.github.io/GSDF/)
  [[Code]](https://github.com/city-super/GSDF)

- **3DGSR: Implicit Surface Reconstruction with 3D Gaussian Splatting**  
  [[Paper]](https://arxiv.org/abs/2404.00409)

- **GS-ROR^2: Bidirectional-guided 3DGS and SDF for Reflective Object Relighting and Reconstruction**  
  [[Paper]](https://arxiv.org/abs/2406.18544)

- **SplatSDF: Boosting Neural Implicit SDF via Gaussian Splatting Fusion**  
  [[Paper]](https://arxiv.org/abs/2411.15468)
  [[Project-Page]](https://blarklee.github.io/splatsdf/)

- **GLS: Geometry-aware 3D Language Gaussian Splatting**  
  [[Paper]](https://arxiv.org/abs/2411.18066)
  [[Project-Page]](https://jiaxiongq.github.io/GLS_ProjectPage/)
  [[Code]](https://github.com/JiaxiongQ/GLS)

- **FeatureGS: Eigenvalue-Feature Optimization in 3D Gaussian Splatting for Geometrically Accurate and Artifact-Reduced Reconstruction**  
  [[Paper]](https://arxiv.org/abs/2501.17655)

###### 3.1.2.2.3 Physics Simulation

- **OmniPhysGS: 3D Constitutive Gaussians for General Physics-Based Dynamics Generation**  
  [[Paper]](https://arxiv.org/abs/2501.18982)
  [[Project-Page]](https://wgsxm.github.io/projects/omniphysgs/)
  [[Code]](https://github.com/wgsxm/OmniPhysGS)

- **From Sparse to Dense: Camera Relocalization with Scene-Specific Detector from Feature Gaussian Splatting**  
  [[Paper]](https://arxiv.org/abs/2503.19358)
  [[Project-Page]](https://zju3dv.github.io/STDLoc/)
  [[Code]](https://github.com/zju3dv/STDLoc)

- **DecoupledGaussian: Object-Scene Decoupling for Physics-Based Interaction**  
  [[Paper]](https://arxiv.org/abs/2503.05484)
  [[Project-Page]](https://wangmiaowei.github.io/DecoupledGaussian.github.io/)
  [[Code]](https://github.com/wangmiaowei/DecoupledGaussian/tree/main)

##### 3.1.2.3 Scene Reconstruction

###### 3.1.2.3.1 Dynamic Scene Reconstruction

- **4D Gaussian Splatting for Real-Time Dynamic Scene Rendering**  
  [[Paper]](https://arxiv.org/abs/2310.08528)
  [[Project-Page]](https://guanjunwu.github.io/4dgs/)
  [[Code]](https://github.com/hustvl/4DGaussians)

- **GaussianVideo: Efficient Video Representation via Hierarchical Gaussian Splatting**  
  [[Paper]](https://arxiv.org/abs/2501.04782)
  [[Project-Page]](https://cyberiada.github.io/GaussianVideo/)

- **1000+ FPS 4D Gaussian Splatting for Dynamic Scene Rendering**  
  [[Paper]](https://arxiv.org/abs/2503.16422)
  [[Project-Page]](https://4dgs-1k.github.io/)

- **MoDec-GS: Global-to-Local Motion Decomposition and Temporal Interval Adjustment for Compact Dynamic 3D Gaussian Splatting**  
  [[Paper]](https://arxiv.org/abs/2501.03714)
  [[Project-Page]](https://kaist-viclab.github.io/MoDecGS-site/)

- **Seeing World Dynamics in a Nutshell**  
  [[Paper]](https://arxiv.org/abs/2502.03465)
  [[Code]](https://github.com/Nut-World/NutWorld)

- **Instant Gaussian Stream: Fast and Generalizable Streaming of Dynamic Scene Reconstruction via Gaussian Splatting**  
  [[Paper]](https://arxiv.org/abs/2503.16979)

- **Geo4D: Leveraging Video Generators for Geometric 4D Scene Reconstruction**  
  [[Paper]](https://arxiv.org/abs/2504.07961)
  [[Project-Page]](https://geo4d.github.io/)
  [[Code]](http://github.com/jzr99/Geo4D)

###### 3.1.2.3.2 Large-Scale Scene Reconstruction

- **VastGaussian: Vast 3D Gaussians for Large Scene Reconstruction**  
  [[Paper]](https://arxiv.org/abs/2402.17427)
  [[Project-Page]](https://vastgaussian.github.io/)

- **CityGaussian: Real-time High-quality Large-Scale Scene Rendering with Gaussians**  
  [[Paper]](https://arxiv.org/abs/2404.01133)
  [[Project-Page]](https://dekuliutesla.github.io/citygs/)
  [[Code]](https://github.com/Linketic/CityGaussian)

- **OccluGaussian: Occlusion-Aware Gaussian Splatting for Large Scene Reconstruction and Rendering**  
  [[Paper]](https://arxiv.org/abs/2503.16177)
  [[Project-Page]](https://occlugaussian.github.io/)

###### 3.1.2.3.3 Sparse-View Reconstruction

- **MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-View Images**  
  [[Paper]](https://arxiv.org/abs/2403.14627)
  [[Project-Page]](https://donydchen.github.io/mvsplat)
  [[Code]](https://github.com/donydchen/mvsplat)

- **TranSplat: Generalizable 3D Gaussian Splatting from Sparse Multi-View Images with Transformers**  
  [[Paper]](https://arxiv.org/abs/2408.13770)
  [[Project-Page]](https://xingyoujun.github.io/transplat/)
  [[Code]](https://github.com/xingyoujun/transplat)

- **PixelGaussian: Generalizable 3D Gaussian Reconstruction from Arbitrary Views**  
  [[Paper]](https://arxiv.org/abs/2410.18979)
  [[Project-Page]](https://wzzheng.net/PixelGaussian/)
  [[Code]](https://github.com/Barrybarry-Smith/PixelGaussian)

- **Flash3D: Feed-Forward Generalisable 3D Scene Reconstruction from a Single Image**  
  [[Paper]](https://arxiv.org/abs/2406.04343)
  [[Project-Page]](https://www.robots.ox.ac.uk/~vgg/research/flash3d/)
  [[Code]](https://github.com/eldar/flash3d)

- **FatesGS: Fast and Accurate Sparse-View Surface Reconstruction using Gaussian Splatting with Depth-Feature Consistency**  
  [[Paper]](https://arxiv.org/abs/2501.04628)
  [[Project-Page]](https://alvin528.github.io/FatesGS/)
  [[Code]](https://github.com/yulunwu0108/FatesGS)

- **SplatFormer: Point Transformer for Robust 3D Gaussian Splatting**  
  [[Paper]](https://arxiv.org/abs/2411.06390)
  [[Project-Page]](https://sergeyprokudin.github.io/splatformer/)
  [[Code]](https://github.com/ChenYutongTHU/SplatFormer)

###### 3.1.2.3.4 Unposed Scene Reconstruction

- **Splatt3R: Zero-shot Gaussian Splatting from Uncalibrated Image Pairs**  
  [[Paper]](https://arxiv.org/abs/2408.13912)
  [[Project-Page]](https://splatt3r.active.vision/)
  [[Code]](https://github.com/btsmart/splatt3r)

- **No Pose, No Problem: Surprisingly Simple 3D Gaussian Splats from Sparse Unposed Images**  
  [[Paper]](https://arxiv.org/abs/2410.24207)
  [[Project-Page]](https://noposplat.github.io/)
  [[Code]](https://github.com/cvg/NoPoSplat)

- **VicaSplat: A Single Run is All You Need for 3D Gaussian Splatting and Camera Estimation from Unposed Video Frames**  
  [[Paper]](https://arxiv.org/abs/2503.10286)
  [[Project-Page]](https://lizhiqi49.github.io/VicaSplat/)
  [[Code]](https://github.com/WU-CVGL/VicaSplat)

- **FLARE: Feed-forward Geometry, Appearance and Camera Estimation from Uncalibrated Sparse Views**  
  [[Paper]](https://arxiv.org/abs/2502.12138)
  [[Project-Page]](https://zhanghe3z.github.io/FLARE/)
  [[Code]](https://github.com/ant-research/FLARE)

- **Coca-Splat: Collaborative Optimization for Camera Parameters and 3D Gaussians**  
  [[Paper]](https://arxiv.org/abs/2504.00639)

###### 3.1.2.3.5 Sparse-View Scene Extension

- **Taming Video Diffusion Prior with Scene-Grounding Guidance for 3D Gaussian Splatting from Sparse Inputs**  
  [[Paper]](https://arxiv.org/abs/2503.05082)
  [[Project-Page]](https://zhongyingji.github.io/guidevd-3dgs/)
  [[Code]](https://github.com/zhongyingji/guidedvd-3dgs)

- **ExScene: Free-View 3D Scene Reconstruction with Gaussian Splatting from a Single Image**  
  [[Paper]](https://arxiv.org/abs/2503.23881)

- **Free360: Layered Gaussian Splatting for Unbounded 360-Degree View Synthesis from Extremely Sparse and Unposed Views**  
  [[Paper]](https://arxiv.org/abs/2503.24382)
  [[Project-Page]](https://zju3dv.github.io/free360/)
  [[Code]](https://github.com/chobao/Free360)

- **ERUPT: Efficient Rendering with Unposed Patch Transformer**  
  [[Paper]](https://arxiv.org/abs/2503.24374)

##### 3.1.2.4 Downstream Tasks

###### 3.1.2.4.1 Detection

- **3DGS-DET: Empower 3D Gaussian Splatting with Boundary Guidance and Box-Focused Sampling for 3D Object Detection**  
  [[Paper]](https://arxiv.org/abs/2410.01647)
  [[Code]](https://github.com/yangcaoai/3DGS-DET)

###### 3.1.2.4.2 Segmentation

- **Segment Any 3D Gaussians**  
  [[Paper]](https://arxiv.org/abs/2312.00860)
  [[Project-Page]](https://jumpat.github.io/SAGA/)
  [[Code]](https://github.com/Jumpat/SegAnyGAussians)

- **LangSplat: 3D Language Gaussian Splatting**  
  [[Paper]](https://arxiv.org/abs/2312.16084)
  [[Project-Page]](https://langsplat.github.io/)
  [[Code]](https://github.com/minghanqin/LangSplat)

- **Feature 3DGS: Supercharging 3D Gaussian Splatting to Enable Distilled Feature Fields**  
  [[Paper]](https://arxiv.org/abs/2312.03203)
  [[Project-Page]](https://feature-3dgs.github.io/)
  [[Code]](https://github.com/ShijieZhou-UCLA/feature-3dgs)

- **Language Embedded 3D Gaussians for Open-Vocabulary Scene Understanding**  
  [[Paper]](https://arxiv.org/abs/2311.18482)
  [[Project-Page]](https://buaavrcg.github.io/LEGaussians/)
  [[Code]](https://github.com/buaavrcg/LEGaussians)

- **SAGD: Boundary-Enhanced Segment Anything in 3D Gaussian via Gaussian Decomposition**  
  [[Paper]](https://arxiv.org/abs/2401.17857)

- **FMGS: Foundation Model Embedded 3D Gaussian Splatting for Holistic 3D Scene Understanding**  
  [[Paper]](https://arxiv.org/abs/2401.01970)
  [[Project-Page]](https://xingxingzuo.github.io/fmgs/)
  [[Code]](https://github.com/google-research/foundation-model-embedded-3dgs)

- **EgoLifter: Open-world 3D Segmentation for Egocentric Perception**  
  [[Paper]](https://arxiv.org/abs/2403.18118)
  [[Project-Page]](https://egolifter.github.io/)
  [[Code]](https://github.com/facebookresearch/egolifter)

- **Semantic Gaussians: Open-Vocabulary Scene Understanding with 3D Gaussian Splatting**  
  [[Paper]](https://arxiv.org/abs/2403.15624)
  [[Project-Page]](https://sharinka0715.github.io/semantic-gaussians/)
  [[Code]](https://github.com/sharinka0715/semantic-gaussians)

- **GOI: Find 3D Gaussians of Interest with an Optimizable Open-vocabulary Semantic-space Hyperplane**  
  [[Paper]](https://arxiv.org/abs/2405.17596)
  [[Project-Page]](https://quyans.github.io/GOI-Hyperplane/)
  [[Code]](https://github.com/Quyans/GOI-Hyperplane)

- **OpenGaussian: Towards Point-Level 3D Gaussian-based Open Vocabulary Understanding**  
  [[Paper]](https://arxiv.org/abs/2406.02058)
  [[Project-Page]](https://3d-aigc.github.io/OpenGaussian/)
  [[Code]](https://github.com/yanmin-wu/OpenGaussian)

- **3D Vision-Language Gaussian Splatting**  
  [[Paper]](https://arxiv.org/abs/2410.07577)

- **SADG: Segment Any Dynamic Gaussian Without Object Trackers**  
  [[Paper]](https://arxiv.org/abs/2411.19290)
  [[Project-Page]](https://yunjinli.github.io/project-sadg/)
  [[Code]](https://github.com/yunjinli/SADG-SegmentAnyDynamicGaussian)

- **DCSEG: Decoupled 3D Open-Set Segmentation using Gaussian Splatting**  
  [[Paper]](https://arxiv.org/abs/2412.10972)

- **SuperGSeg: Open-Vocabulary 3D Segmentation with Structured Super-Gaussians**  
  [[Paper]](https://arxiv.org/abs/2412.10231)
  [[Project-Page]](https://supergseg.github.io/)
  [[Code]](https://github.com/supergseg/supergseg)

- **GaussTR: Foundation Model-Aligned Gaussian Transformer for Self-Supervised 3D Spatial Understanding**  
  [[Paper]](https://arxiv.org/abs/2412.13193)
  [[Project-Page]](https://hustvl.github.io/GaussTR/)
  [[Code]](https://github.com/hustvl/GaussTR)

- **LangSurf: Language-Embedded Surface Gaussians for 3D Scene Understanding**  
  [[Paper]](https://arxiv.org/abs/2412.17635)
  [[Project-Page]](https://langsurf.github.io/)
  [[Code]](https://github.com/lifuguan/LangSurf)

- **GSemSplat: Generalizable Semantic 3D Gaussian Splatting from Uncalibrated Image Pairs**  
  [[Paper]](https://arxiv.org/abs/2412.16932)

- **CLIP-GS: CLIP-Informed Gaussian Splatting for Real-time and View-consistent 3D Semantic Understanding**  
  [[Paper]](https://arxiv.org/abs/2404.14249)
  [[Project-Page]](https://gbliao.github.io/CLIP-GS.github.io/)
  [[Code]](https://github.com/gbliao/CLIP-GS)

- **SLGaussian: Fast Language Gaussian Splatting in Sparse Views**  
  [[Paper]](https://arxiv.org/abs/2412.08331)
  [[Project-Page]]()
  [[Code]]()

- **InstanceGaussian: Appearance-Semantic Joint Gaussian Representation for 3D Instance-Level Perception**  
  [[Paper]](https://arxiv.org/abs/2411.19235)
  [[Project-Page]](https://lhj-git.github.io/InstanceGaussian/)

- **OVGaussian: Generalizable 3D Gaussian Segmentation with Open Vocabularies**  
  [[Paper]](https://arxiv.org/abs/2501.00326)
  [[Code]](https://github.com/runnanchen/OVGaussian)

- **Lifting by Gaussians: A Simple, Fast and Flexible Method for 3D Instance Segmentation**  
  [[Paper]](https://arxiv.org/abs/2502.00173)

- **PanoGS: Gaussian-based Panoptic Segmentation for 3D Open Vocabulary Scene Understanding**  
  [[Paper]](https://arxiv.org/abs/2503.18107)
  [[Project-Page]](https://zju3dv.github.io/panogs/)
  [[Code]](https://github.com/zhaihongjia/PanoGS)

- **SceneSplat: Gaussian Splatting-based Scene Understanding with Vision-Language Pretraining**  
  [[Paper]](https://arxiv.org/abs/2503.18052)
  [[Code]](https://github.com/unique1i/SceneSplat)

- **M3: 3D-Spatial MultiModal Memory**  
  [[Paper]](https://arxiv.org/abs/2503.16413)
  [[Project-Page]](https://m3-spatial-memory.github.io/)
  [[Code]](https://github.com/MaureenZOU/m3-spatial)

- **EgoSplat: Open-Vocabulary Egocentric Scene Understanding with Language Embedded 3D Gaussian Splatting**  
  [[Paper]](https://arxiv.org/abs/2503.11345)

- **Segment then Splat: A Unified Approach for 3D Open-Vocabulary Segmentation based on Gaussian Splatting**  
  [[Paper]](https://arxiv.org/abs/2503.22204)
  [[Project-Page]](https://vulab-ai.github.io/Segment-then-Splat/)

- **Rethinking End-to-End 2D to 3D Scene Segmentation in Gaussian Splatting**  
  [[Paper]](https://arxiv.org/abs/2503.14029)
  [[Project-Page]]()
  [[Code]](https://github.com/Runsong123/Unified-Lift)

- **CAGS: Open-Vocabulary 3D Scene Understanding with Context-Aware Gaussian Splatting**  
  [[Paper]](https://arxiv.org/abs/2504.11893)

- **NVSMask3D: Hard Visual Prompting with Camera Pose Interpolation for 3D Open Vocabulary Instance Segmentation**  
  [[Paper]](https://arxiv.org/abs/2504.14638)

- **Training-Free Hierarchical Scene Understanding for Gaussian Splatting with Superpoint Graphs**  
  [[Paper]](https://arxiv.org/abs/2504.13153)
  [[Code]](https://github.com/Atrovast/THGS)

###### 3.1.2.4.3 Occupancy Prediction

- **GaussianFormer: Scene as Gaussians for Vision-Based 3D Semantic Occupancy Prediction**  
  [[Paper]](https://arxiv.org/abs/2405.17429)
  [[Code]](https://github.com/huang-yh/GaussianFormer)

- **GaussianFormer-2: Probabilistic Gaussian Superposition for Efficient 3D Occupancy Prediction**  
  [[Paper]](https://arxiv.org/abs/2412.04384)
  [[Code]](https://github.com/huang-yh/GaussianFormer)

- **AutoOcc: Automatic Open-Ended Semantic Occupancy Annotation via Vision-Language Guided Gaussian Splatting**  
  [[Paper]](https://arxiv.org/abs/2502.04981)

- **GSRender: Deduplicated Occupancy Prediction via Weakly Supervised 3D Gaussian Splatting**  
  [[Paper]](https://arxiv.org/abs/2412.14579)

- **GaussianAD: Gaussian-Centric End-to-End Autonomous Driving**  
  [[Paper]](https://arxiv.org/abs/2412.10371)
  [[Code]](https://github.com/wzzheng/GaussianAD)

- **EmbodiedOcc: Embodied 3D Occupancy Prediction for Vision-based Online Scene Understanding**  
  [[Paper]](https://arxiv.org/abs/2412.04380)
  [[Project-Page]](https://ykiwu.github.io/EmbodiedOcc/)
  [[Code]](https://github.com/YkiWu/EmbodiedOcc)

- **GaussRender: Learning 3D Occupancy with Gaussian Rendering**  
  [[Paper]](https://arxiv.org/abs/2502.05040)
  [[Code]](https://github.com/valeoai/GaussRender)

- **GaussianFlowOcc: Sparse and Weakly Supervised Occupancy Estimation using Gaussian Splatting and Temporal Flow**  
  [[Paper]](https://arxiv.org/abs/2502.17288)
  [[Code]](https://github.com/boschresearch/GaussianFlowOcc)

###### 3.1.2.4.4 Scene Graph Generation

- **DynamicGSG: Dynamic 3D Gaussian Scene Graphs for Environment Adaptation**  
  [[Paper]](https://arxiv.org/abs/2502.15309)
  [[Code]](https://github.com/GeLuzhou/Dynamic-GSG)

- **GaussianGraph: 3D Gaussian-based Scene Graph Generation for Open-world Scene Understanding**  
  [[Paper]](https://arxiv.org/abs/2503.04034)
  [[Project-Page]](https://wangxihan-bit.github.io/GaussianGraph/)
  [[Code]](https://github.com/WangXihan-bit/GaussianGraph/)

- **Intelligent Spatial Perception by Building Hierarchical 3D Scene Graphs for Indoor Scenarios with the Help of LLMs**  
  [[Paper]](https://arxiv.org/abs/2503.15091)

###### 3.1.2.4.5 Navigation

- **GaussNav: Gaussian Splatting for Visual Navigation**  
  [[Paper]](https://arxiv.org/abs/2403.11625)
  [[Project-Page]](https://xiaohanlei.github.io/projects/GaussNav/)
  [[Code]](https://github.com/XiaohanLei/GaussNav)

- **UnitedVLN: Generalizable Gaussian Splatting for Continuous Vision-Language Navigation**  
  [[Paper]](https://arxiv.org/abs/2411.16053)

###### 3.1.2.4.6 SLAM

- **WildGS-SLAM: Monocular Gaussian Splatting SLAM in Dynamic Environments**  
  [[Paper]](https://arxiv.org/abs/2504.03886)
  [[Project-Page]](https://wildgs-slam.github.io/)
  [[Code]](https://github.com/GradientSpaces/WildGS-SLAM)

##### 3.1.2.5 Gaussian Splatting based 3D Foundation Model

- **Large Spatial Model: End-to-end Unposed Images to Semantic 3D**  
  [[Paper]](https://arxiv.org/abs/2410.18956)
  [[Project-Page]](https://largespatialmodel.github.io/)
  [[Code]](https://github.com/NVlabs/LSM)

- **GaussianPretrain: A Simple Unified 3D Gaussian Representation for Visual Pre-training in Autonomous Driving**  
  [[Paper]](https://arxiv.org/abs/2411.12452)
  [[Code]](https://github.com/Public-BOTs/GaussianPretrain)

- **VisionPAD: A Vision-Centric Pre-training Paradigm for Autonomous Driving**  
  [[Paper]](https://arxiv.org/abs/2411.14716)

#### 3.1.3 Geometry Method

- **TetSphere Splatting: Representing High-Quality Geometry with Lagrangian Volumetric Meshes**  
  [[Paper]](https://arxiv.org/abs/2405.20283)
  [[Code]](https://github.com/gmh14/tssplat)

- **PoseDiffusion: Solving Pose Estimation via Diffusion-aided Bundle Adjustment**  
  [[Paper]](https://arxiv.org/abs/2306.15667)
  [[Project-Page]](https://posediffusion.github.io/)
  [[Code]](https://github.com/facebookresearch/PoseDiffusion)

- **RelPose++: Recovering 6D Poses from Sparse-view Observations**  
  [[Paper]](https://arxiv.org/abs/2305.04926)
  [[Project-Page]](https://amyxlase.github.io/relpose-plus-plus/)
  [[Code]](https://github.com/amyxlase/relpose-plus-plus)

- **Visual Geometry Grounded Deep Structure From Motion**  
  [[Paper]](https://arxiv.org/abs/2312.04563)
  [[Project-Page]](https://vggsfm.github.io/)
  [[Code]](https://github.com/facebookresearch/vggsfm)

- **MoGe: Unlocking Accurate Monocular Geometry Estimation for Open-Domain Images with Optimal Training Supervision**  
  [[Paper]](https://arxiv.org/abs/2410.19115)
  [[Project-Page]](https://wangrc.site/MoGePage/)
  [[Code]](https://github.com/microsoft/moge)

- **LoRA3D: Low-Rank Self-Calibration of 3D Geometric Foundation Models**  
  [[Paper]](https://arxiv.org/abs/2412.07746)

- **Relative Pose Estimation through Affine Corrections of Monocular Depth Priors**  
  [[Paper]](https://arxiv.org/abs/2501.05446)
  [[Code]](https://github.com/MarkYu98/madpose)

- **Uni4D: Unifying Visual Foundation Models for 4D Modeling from a Single Video**  
  [[Paper]](https://arxiv.org/abs/2503.21761)
  [[Project-Page]](https://davidyao99.github.io/uni4d/)
  [[Code]](https://github.com/Davidyao99/uni4d/tree/main)

- **Can Video Diffusion Model Reconstruct 4D Geometry?**  
  [[Paper]](https://arxiv.org/abs/2503.21082)
  [[Project-Page]](https://wayne-mai.github.io/publication/sora3r_arxiv_2025/)

#### 3.1.4 Point Cloud

##### 3.1.4.1 Survey

- **Advances in 3D pre-training and downstream tasks: a survey**  
  [[Paper]](https://link.springer.com/article/10.1007/s44336-024-00007-4)

##### 3.1.4.2 Base Model

- **PointNeXt: Revisiting PointNet++ with Improved Training and Scaling Strategies**  
  [[Paper]](https://arxiv.org/abs/2206.04670)
  [[Code]](https://github.com/guochengqian/pointnext)

- **Masked Autoencoders for Point Cloud Self-supervised Learning**  
  [[Paper]](https://arxiv.org/abs/2203.06604)
  [[Code]](https://github.com/Pang-Yatian/Point-MAE)

- **Swin3D: A Pretrained Transformer Backbone for 3D Indoor Scene Understanding**  
  [[Paper]](https://arxiv.org/abs/2304.06906)
  [[Project-Page]](https://yukichiii.github.io/project/swin3D/swin3D.html)
  [[Code]](https://github.com/microsoft/Swin3D)

- **Point Transformer V3: Simpler, Faster, Stronger**  
  [[Paper]](https://arxiv.org/abs/2312.10035)
  [[Code]](https://github.com/Pointcept/PointTransformerV3)

- **ShapeSplat: A Large-scale Dataset of Gaussian Splats and Their Self-Supervised Pretraining**  
  [[Paper]](https://arxiv.org/abs/2408.10906)
  [[Project-Page]](https://unique1i.github.io/ShapeSplat/)
  [[Code]](https://github.com/qimaqi/ShapeSplat-Gaussian_MAE)

- **MAP: Unleashing Hybrid Mamba-Transformer Vision Backbone's Potential with Masked Autoregressive Pretraining**  
  [[Paper]](https://arxiv.org/abs/2410.00871)
  [[Code]](https://github.com/yunzeliu/MAP)

- **Point Cloud Understanding via Attention-Driven Contrastive Learning**  
  [[Paper]](https://arxiv.org/abs/2411.14744)

- **Point Cloud Unsupervised Pre-training via 3D Gaussian Splatting**  
  [[Paper]](https://arxiv.org/abs/2411.18667)

- **Masked Scene Modeling: Narrowing the Gap Between Supervised and Self-Supervised Learning in 3D Scene Understanding**  
  [[Paper]](https://arxiv.org/abs/2504.06719)
  [[Project-Page]](https://phermosilla.github.io/msm/)
  [[Code]](https://github.com/phermosilla/msm)

##### 3.1.4.3 Usage

###### 3.1.4.3.1 Multimodal Alignment

- ****  
  [[Paper]]()
  [[Project-Page]]()
  [[Code]]()

- ****  
  [[Paper]]()
  [[Project-Page]]()
  [[Code]]()

###### 3.1.4.3.2 LLM

- **3D-LLM: Injecting the 3D World into Large Language Models**  
  [[Paper]](https://arxiv.org/abs/2307.12981)
  [[Code]](https://github.com/UMass-Embodied-AGI/3D-LLM)

- **SceneVerse: Scaling 3D Vision-Language Learning for Grounded Scene Understanding**  
  [[Paper]](https://arxiv.org/abs/2401.09340)
  [[Project-Page]](https://scene-verse.github.io/)
  [[Code]](https://github.com/scene-verse/sceneverse)

- **ShapeLLM: Universal 3D Object Understanding for Embodied Interaction**  
  [[Paper]](https://arxiv.org/abs/2402.17766)
  [[Project-Page]](https://qizekun.github.io/shapellm/)
  [[Code]](https://github.com/qizekun/ShapeLLM)

- **Any2Point: Empowering Any-modality Large Models for Efficient 3D Understanding**  
  [[Paper]](https://arxiv.org/abs/2404.07989)
  [[Code]](https://github.com/Ivan-Tang-3D/Any2Point)

- **MiniGPT-3D: Efficiently Aligning 3D Point Clouds with Large Language Models using 2D Priors**  
  [[Paper]](https://arxiv.org/abs/2405.01413)
  [[Project-Page]](https://tangyuan96.github.io/minigpt_3d_project_page/)
  [[Code]](https://github.com/TangYuan96/MiniGPT-3D)

- **LLaVA-3D: A Simple yet Effective Pathway to Empowering LMMs with 3D-awareness**  
  [[Paper]](https://arxiv.org/abs/2409.18125)
  [[Project-Page]](https://zcmax.github.io/projects/LLaVA-3D/)
  [[Code]](https://github.com/ZCMax/LLaVA-3D)

- **3DGraphLLM: Combining Semantic Graphs and Large Language Models for 3D Scene Understanding**  
  [[Paper]](https://arxiv.org/abs/2412.18450)
  [[Code]](https://github.com/CognitiveAISystems/3DGraphLLM)

- **Oryx MLLM: On-Demand Spatial-Temporal Understanding at Arbitrary Resolution**  
  [[Paper]](https://arxiv.org/abs/2409.12961)
  [[Project-Page]](https://oryx-mllm.github.io/)
  [[Code]](https://github.com/Oryx-mllm/Oryx)

##### 3.1.4.4 Downstream Tasks

###### 3.1.4.4.1 Detection

- **Unifying Voxel-based Representation with Transformer for 3D Object Detection**  
  [[Paper]](https://arxiv.org/abs/2206.00630)
  [[Code]](https://github.com/dvlab-research/UVTR)

- **Uni3DETR: Unified 3D Detection Transformer**  
  [[Paper]](https://arxiv.org/abs/2310.05699)
  [[Code]](https://github.com/zhenyuw16/Uni3DETR)

- **UniDet3D: Multi-dataset Indoor 3D Object Detection**  
  [[Paper]](https://arxiv.org/abs/2409.04234)
  [[Code]](https://github.com/filapro/unidet3d)

- **Cocoon: Robust Multi-Modal Perception with Uncertainty-Aware Sensor Fusion**  
  [[Paper]](https://arxiv.org/abs/2410.12592)

###### 3.1.4.4.2 Segmentation

- **ConceptFusion: Open-set Multimodal 3D Mapping**  
  [[Paper]](https://arxiv.org/abs/2302.07241)
  [[Project-Page]](https://concept-fusion.github.io/)
  [[Code]](https://github.com/concept-fusion/concept-fusion)

- **OpenScene: 3D Scene Understanding with Open Vocabularies**  
  [[Paper]](https://arxiv.org/abs/2211.15654)
  [[Project-Page]](https://pengsongyou.github.io/openscene)
  [[Code]](https://github.com/pengsongyou/openscene)

- **CLIP2Scene: Towards Label-efficient 3D Scene Understanding by CLIP**  
  [[Paper]](https://arxiv.org/abs/2301.04926)
  [[Code]](https://github.com/runnanchen/CLIP2Scene)

- **Language-Assisted 3D Scene Understanding**  
  [[Paper]](https://arxiv.org/abs/2312.11451)

- **OneFormer3D: One Transformer for Unified Point Cloud Segmentation**  
  [[Paper]](https://arxiv.org/abs/2311.14405)

- **Open3DIS: Open-Vocabulary 3D Instance Segmentation with 2D Mask Guidance**  
  [[Paper]](https://arxiv.org/abs/2312.10671)
  [[Project-Page]](https://open3dis.github.io/)
  [[Code]](https://github.com/VinAIResearch/Open3DIS)

- **MaskClustering: View Consensus based Mask Graph Clustering for Open-Vocabulary 3D Instance Segmentation**  
  [[Paper]](https://arxiv.org/abs/2401.07745)
  [[Project-Page]](https://pku-epic.github.io/MaskClustering/)
  [[Code]](https://github.com/PKU-EPIC/MaskClustering)

- **SAI3D: Segment Any Instance in 3D Scenes**  
  [[Paper]](https://arxiv.org/abs/2312.11557)
  [[Project-Page]](https://yd-yin.github.io/SAI3D/)
  [[Code]](https://github.com/yd-yin/SAI3D)

- **OpenMask3D: Open-Vocabulary 3D Instance Segmentation**  
  [[Paper]](https://arxiv.org/abs/2306.13631)
  [[Project-Page]](https://openmask3d.github.io/)
  [[Code]](https://github.com/OpenMask3D/openmask3d)

- **Towards Label-free Scene Understanding by Vision Foundation Models**  
  [[Paper]](https://arxiv.org/abs/2306.03899)
  [[Code]](https://github.com/runnanchen/Label-Free-Scene-Understanding)

- **SAM3D: Segment Anything in 3D Scenes**  
  [[Paper]](https://arxiv.org/abs/2306.03908)
  [[Code]](https://github.com/Pointcept/SegmentAnything3D)

- **A Unified Framework for 3D Scene Understanding**  
  [[Paper]](https://arxiv.org/abs/2407.03263)
  [[Project-Page]](https://dk-liang.github.io/UniSeg3D/)
  [[Code]](https://github.com/dk-liang/UniSeg3D)

- **OpenIns3D: Snap and Lookup for 3D Open-vocabulary Instance Segmentation**  
  [[Paper]](https://arxiv.org/abs/2309.00616)
  [[Project-Page]](https://zheninghuang.github.io/OpenIns3D/)
  [[Code]](https://github.com/Pointcept/OpenIns3D)

- **SAM-Guided Masked Token Prediction for 3D Scene Understanding**  
  [[Paper]](https://arxiv.org/abs/2410.12158)

- **Open-Vocabulary SAM3D: Towards Training-free Open-Vocabulary 3D Scene Understanding**  
  [[Paper]](https://arxiv.org/abs/2405.15580)
  [[Project-Page]](https://hithqd.github.io/projects/OV-SAM3D/)
  [[Code]](https://github.com/HanchenTai/OV-SAM3D)

- **SAM2Point: Segment Any 3D as Videos in Zero-shot and Promptable Manners**  
  [[Paper]](https://arxiv.org/abs/2408.16768)
  [[Project-Page]](https://sam2point.github.io/)
  [[Code]](https://github.com/ZiyuGuo99/SAM2Point)

- **GrabS: Generative Embodied Agent for 3D Object Segmentation without Scene Supervision**  
  [[Paper]](https://arxiv.org/abs/2504.11754)
  [[Code]](https://github.com/vLAR-group/GrabS)

- **Multimodality Helps Few-shot 3D Point Cloud Semantic Segmentation**  
  [[Paper]](https://arxiv.org/abs/2410.22489)
  [[Code]](https://github.com/ZhaochongAn/Multimodality-3D-Few-Shot)

- **UniPLV: Towards Label-Efficient Open-World 3D Scene Understanding by Regional Visual Language Supervision**  
  [[Paper]](https://arxiv.org/abs/2412.18131)

- **Cross-Modal and Uncertainty-Aware Agglomeration for Open-Vocabulary 3D Scene Understanding**  
  [[Paper]](https://arxiv.org/abs/2503.16707)
  [[Project-Page]](https://tyroneli.github.io/CUA_O3D/)
  [[Code]](https://github.com/TyroneLi/CUA_O3D)

- **Pre-training with 3D Synthetic Data: Learning 3D Point Cloud Instance Segmentation from 3D Synthetic Scenes**  
  [[Paper]](https://arxiv.org/abs/2503.24229)

- **DINO in the Room: Leveraging 2D Foundation Models for 3D Segmentation**  
  [[Paper]](https://arxiv.org/abs/2503.18944)
  [[Project-Page]](https://vision.rwth-aachen.de/DITR)
  [[Code]](https://github.com/VisualComputingInstitute/DITR)

###### 3.1.4.4.3 Occupancy Prediction

- **EmbodiedScan: A Holistic Multi-Modal 3D Perception Suite Towards Embodied AI**  
  [[Paper]](https://arxiv.org/abs/2312.16170)
  [[Project-Page]](https://tai-wang.github.io/embodiedscan/)
  [[Code]](https://github.com/OpenRobotLab/EmbodiedScan)

- **ZOPP: A Framework of Zero-shot Offboard Panoptic Perception for Autonomous Driving**  
  [[Paper]](https://arxiv.org/abs/2411.05311)
  [[Code]](https://github.com/PJLab-ADG/ZOPP)

- **OccMamba: Semantic Occupancy Prediction with State Space Models**  
  [[Paper]](https://arxiv.org/abs/2408.09859)
  [[Code]](https://github.com/USTCLH/OccMamba)

###### 3.1.4.4.4 Visual Grounding

- **OVIR-3D: Open-Vocabulary 3D Instance Retrieval Without Training on 3D Data**  
  [[Paper]](https://arxiv.org/abs/2311.02873)
  [[Code]](https://github.com/shiyoung77/OVIR-3D)

- **Task-oriented Sequential Grounding and Navigation in 3D Scenes**  
  [[Paper]](https://arxiv.org/abs/2408.04034)
  [[Project-Page]](https://sg-3d.github.io/)

- **SeeGround: See and Ground for Zero-Shot Open-Vocabulary 3D Visual Grounding**  
  [[Paper]](https://arxiv.org/abs/2412.04383)
  [[Project-Page]](https://seeground.github.io/)
  [[Code]](https://github.com/iris0329/SeeGround)

- **AugRefer: Advancing 3D Visual Grounding via Cross-Modal Augmentation and Spatial Relation-based Referring**  
  [[Paper]](https://arxiv.org/abs/2501.09428)

- **TSP3D: Text-guided Sparse Voxel Pruning for Efficient 3D Visual Grounding**  
  [[Paper]](https://arxiv.org/abs/2502.10392)
  [[Code]](https://github.com/GWxuan/TSP3D)

- **Unifying 2D and 3D Vision-Language Understanding**  
  [[Paper]](https://arxiv.org/abs/2503.10745)
  [[Project-Page]](https://univlg.github.io/)
  [[Code]](https://github.com/facebookresearch/univlg)

- **Intent3D: 3D Object Detection in RGB-D Scans Based on Human Intention**  
  [[Paper]](https://arxiv.org/abs/2405.18295)
  [[Project-Page]](https://weitaikang.github.io/Intent3D-webpage/)
  [[Code]](https://github.com/WeitaiKang/Intent3D)

- **MLLM-For3D: Adapting Multimodal Large Language Model for 3D Reasoning Segmentation**  
  [[Paper]](https://arxiv.org/abs/2503.18135)

### 3.2 Machine Learning

- **Locate 3D: Real-World Object Localization via Self-Supervised Learning in 3D**  
  *a model for localizing objects in 3D scenes from referring expressions like "the small coffee table between the sofa and the lamp."*  
  [[Paper]](https://arxiv.org/abs/2504.14151)
  [[Code]](https://github.com/facebookresearch/locate-3d)

### 3.3 deep learning

- **Compositional 4D Dynamic Scenes Understanding with Physics Priors for Video Question Answering**  
  *a Neural-Symbolic VideoQA model integrating Physics prior for 4D dynamic properties with explicit scene representation of videos.*  
  [[Paper]](https://arxiv.org/abs/2406.00622)
  [[Project-Page]](https://xingruiwang.github.io/projects/DynSuperCLEVR/)
  [[Code]](https://github.com/XingruiWang/NS-4DPhysics)

- **Can Transformers Capture Spatial Relations between Objects?**  
  *approaches exploiting the long-range attention capabilities of transformers for spatial relationships task*  
  [[Paper]](https://arxiv.org/abs/2403.00729)
  [[Project-Page]](https://sites.google.com/view/spatial-relation)
  [[Code]](https://github.com/AlvinWen428/spatial-relation-benchmark)

- **3D-Aware Visual Question Answering about Parts, Poses and Occlusions**  
  *a 3D-aware VQA model that marries two powerful ideas: probabilistic neural symbolic program execution for reasoning and deep neural networks with 3D generative representations of objects for robust visual recognition.*  
  [[Paper]](https://arxiv.org/abs/2310.17914)
  [[Code]](https://github.com/XingruiWang/3D-Aware-VQA)

- **Pow3R: Empowering Unconstrained 3D Reconstruction with Camera and Scene Priors**  
  [[Paper]](https://arxiv.org/abs/2503.17316)
  [[Project-Page]](https://europe.naverlabs.com/research/publications/pow3r-empowering-unconstrained-3d-reconstruction-with-camera-and-scene-priors/)

- **Easi3R: Estimating Disentangled Motion from DUSt3R Without Training**  
  [[Paper]](https://arxiv.org/abs/2503.24391)
  [[Project-Page]](https://easi3r.github.io/)
  [[Code]](https://github.com/Inception3D/Easi3R) 

- **Regist3R: Incremental Registration with Stereo Foundation Model**  
  [[Paper]](https://arxiv.org/abs/2504.12356)

- **POMATO: Marrying Pointmap Matching with Temporal Motion for Dynamic 3D Reconstruction**  
  [[Paper]](https://arxiv.org/abs/2504.05692)
  [[Code]](https://github.com/wyddmw/POMATO)

#### 3.3.1 Geometry Based

- **MVSAnywhere: Zero-Shot Multi-View Stereo**  
  [[Paper]](https://arxiv.org/abs/2503.22430)
  [[Project-Page]](https://nianticlabs.github.io/mvsanywhere/)
  [[Code]](https://github.com/nianticlabs/mvsanywhere)

- **VGGT: Visual Geometry Grounded Transformer**  
  [[Paper]](https://arxiv.org/abs/2503.11651)
  [[Code]](https://github.com/facebookresearch/vggt)

- **UniK3D: Universal Camera Monocular 3D Estimation**  
  [[Paper]](https://arxiv.org/abs/2503.16591)
  [[Project-Page]](https://lpiccinelli-eth.github.io/pub/unik3d/)
  [[Code]](https://github.com/lpiccinelli-eth/unik3d)  

- **St4RTrack: Simultaneous 4D Reconstruction and Tracking in the World**  
  [[Paper]](https://arxiv.org/abs/2504.13152)
  [[Project-Page]](https://st4rtrack.github.io/)

- **Mono3R: Exploiting Monocular Cues for Geometric 3D Reconstruction**  
  [[Paper]](https://arxiv.org/abs/2504.13419)

### 3.4 LLM

#### 3.4.1 Spatial Reasoning

- **Spatial-R1: Enhancing MLLMs in Video Spatial Reasoning**  
  *a targeted approach involving two key contributions: the curation of SR, a new video spatial reasoning dataset from ScanNet with automatically generated QA pairs across seven task types, and the application of Task-Specific Group Relative Policy Optimization (GRPO) for fine-tuning.*  
  [[Paper]](https://arxiv.org/abs/2504.01805)
  [[Code]](https://github.com/OuyangKun10/Spatial-R1)

- **Improved Visual-Spatial Reasoning via R1-Zero-Like Training**  
  *incorporate GRPO training for improved visual-spatial reasoning, using the carefully curated VSI-100k dataset, following DeepSeek-R1-Zero.*  
  [[Paper]](https://arxiv.org/abs/2504.00883)
  [[Code]](https://github.com/zhijie-group/R1-Zero-VSI)

- **From Flatland to Space: Teaching Vision-Language Models to Perceive and Reason in 3D**  
  *a novel 2D spatial data generation and annotation pipeline built upon scene data with 3D ground-truth, enables the creation of a diverse set of spatial tasks, ranging from basic perception tasks to more complex reasoning tasks.*  
  [[Paper]](https://arxiv.org/abs/2503.22976)
  [[Project-Page]](https://fudan-zvg.github.io/spar/)
  [[Code]](https://github.com/fudan-zvg/spar)

- **SoFar: Language-Grounded Orientation Bridges Spatial Reasoning and Object Manipulation**  
  *introduce the concept of semantic orientation for the capability to precisely understand object orientations, which defines object orientations using natural language in a reference-frame-free manner.*  
  [[Paper]](https://arxiv.org/abs/2502.13143)
  [[Project-Page]](https://qizekun.github.io/sofar/)
  [[Code]](https://github.com/qizekun/SoFar)

- **Spatial457: A Diagnostic Benchmark for 6D Spatial Reasoning of Large Multimodal Models**  
  *Spatial457, a scalable and unbiased synthetic dataset designed with 4 key capability for spatial reasoning: multi-object recognition, 2D location, 3D location, and 3D orientation.*  
  [[Paper]](https://arxiv.org/abs/2502.08636)
  [[Project-Page]](https://xingruiwang.github.io/projects/Spatial457/)
  [[Code]](https://github.com/XingruiWang/Spatial457)

- **SpatialCOT: Advancing Spatial Reasoning through Coordinate Alignment and Chain-of-Thought for Embodied Task Planning**  
  *a novel approach named SpatialCoT comprises two stages: spatial coordinate bi-directional alignment, which aligns vision-language inputs with spatial coordinates, and chain-of-thought spatial grounding, which harnesses the reasoning capabilities of language models for advanced spatial reasoning.*  
  [[Paper]](https://arxiv.org/abs/2501.10074)
  [[Project-Page]](https://spatialcot.github.io/)

- **Imagine while Reasoning in Space: Multimodal Visualization-of-Thought**  
  *Multimodal Visualization-of-Thought (MVoT) enables visual thinking in MLLMs by generating image visualizations of their reasoning traces.*  
  [[Paper]](https://arxiv.org/abs/2501.07542)
  [[Code]](https://github.com/chengzu-li/MVoT)

- **Perception Tokens Enhance Visual Reasoning in Multimodal Language Models**  
  *AURORA, a training method that augments MLMs with perception tokens for improved reasoning over visual inputs.*  
  [[Paper]](https://arxiv.org/abs/2412.03548)
  [[Project-Page]](https://aurora-perception.github.io/)
  [[Code]](https://github.com/mahtabbigverdi/Aurora-perception)

- **Dspy-based Neural-Symbolic Pipeline to Enhance Spatial Reasoning in LLMs**  
  *a novel neural-symbolic framework that enhances LLMs' spatial reasoning abilities through iterative feedback between LLMs and Answer Set Programming (ASP).*  
  [[Paper]](https://arxiv.org/abs/2411.18564)

- **Sparkle: Mastering Basic Spatial Capabilities in Vision Language Models Elicits Generalization to Spatial Reasoning**  
  *a framework that uses synthetic data generation to provide targeted supervision for vision language models (VLMs) in three basic spatial capabilities, creating an instruction dataset for each capability.*  
  [[Paper]](https://arxiv.org/abs/2410.16162)

- **SpaRC and SpaRP: Spatial Reasoning Characterization and Path Generation for Understanding Spatial Reasoning Capability of Large Language Models**  
  *a novel Spatial Reasoning Characterization (SpaRC) framework and Spatial Reasoning Paths (SpaRP) datasets, to enable an in-depth understanding of the spatial relations and compositions as well as the usefulness of spatial reasoning chains.*  
  [[Paper]](https://arxiv.org/abs/2406.04566)
  [[Code]](https://github.com/UKPLab/acl2024-sparc-and-sparp)

- **Beyond Lines and Circles: Unveiling the Geometric Reasoning Gap in Large Language Models**  
  *a framework that formulates an LLMs-based multi-agents system that enhances their existing reasoning potential by conducting an internal dialogue.*  
  [[Paper]](https://arxiv.org/abs/2402.03877)
  [[Code]](https://github.com/SpyrosMouselinos/GeometryAgents)

- **Advancing Spatial Reasoning in Large Language Models: An In-Depth Evaluation and Enhancement Using the StepGame Benchmark**  
  *provide a flawless solution to the benchmark by combining template-to-relation mapping with logic-based reasoning.*  
  [[Paper]](https://arxiv.org/abs/2401.03991)
  [[Code]](https://github.com/Fangjun-Li/SpatialLM-StepGame)

- **Visual Spatial Reasoning**  
  *Visual Spatial Reasoning (VSR), a dataset containing more than 10k natural text-image pairs with 66 types of spatial relations in English*  
  [[Paper]](https://arxiv.org/abs/2205.00363)
  [[Code]](https://github.com/cambridgeltl/visual-spatial-reasoning)

- **SpatialLLM: A Compound 3D-Informed Design towards Spatially-Intelligent Large Multimodal Models**  
  [[Paper]](https://arxiv.org/abs/2505.00788)

#### 3.4.2 Recognition

- **Video-3D LLM: Learning Position-Aware Video Representation for 3D Scene Understanding**  
  *a novel generalist model for 3D scene understanding by treating 3D scenes as dynamic videos and incorporating 3D position encoding into these representations*  
  [[Paper]](https://arxiv.org/abs/2412.00493)
  [[Code]](https://github.com/LaVi-Lab/Video-3D-LLM)

- **Proximity QA: Unleashing the Power of Multi-Modal Large Language Models for Spatial Proximity Analysis**  
  *Framework operates in two phases: the first phase focuses on guiding the models to understand the relative depth of objects, and the second phase further encourages the models to infer the proximity relationships between objects based on their depth perceptions.*  
  [[Paper]](https://arxiv.org/abs/2401.17862)
  [[Code]](https://github.com/NorthSummer/ProximityQA)

- **3DAxiesPrompts: Unleashing the 3D Spatial Task Capabilities of GPT-4V**  
  *By presenting images infused with the 3DAP visual prompt as inputs, we empower GPT-4V to ascertain the spatial positioning information of the given 3D target image with a high degree of precision.*  
  [[Paper]](https://arxiv.org/abs//2312.09738)

- **Improving Vision-and-Language Reasoning via Spatial Relations Modeling**  
  *construct the spatial relation graph based on the given visual scenario.*  
  [[Paper]](https://arxiv.org/abs/2311.05298)

#### 3.4.3 Reinforcement Learning

- **Dream to Control: Learning Behaviors by Latent Imagination**  
  *Dreamer, a reinforcement learning agent that solves long-horizon tasks from images purely by latent imagination.*  
  [[Paper]](https://arxiv.org/abs/1912.01603)
  [[Project-Page]](https://danijar.com/project/dreamer/)
  [[Code]](https://github.com/danijar/dreamer)

## 4. Application

### 4.1 Robotics

- **GR00T N1: An Open Foundation Model for Generalist Humanoid Robots**  
  *GR00T N1 is a Vision-Language-Action (VLA) model with a dual-system architecture. The vision-language module (System 2) interprets the environment through vision and language instructions.*  
  [[Paper]](https://arxiv.org/abs/2503.14734)
  [[Project-Page]](https://developer.nvidia.com/isaac/gr00t)

- **Gemini Robotics: Bringing AI into the Physical World**  
  *Gemini Robotics-ER (Embodied Reasoning) extends Gemini's multimodal reasoning capabilities into the physical world, with enhanced spatial and temporal understanding.*  
  [[Paper]](https://arxiv.org/abs/2503.20020)

- **VL-Nav: Real-time Vision-Language Navigation with Spatial Reasoning**  
  *method integrates pixel-wise vision-language features with curiosity-driven exploration.*  
  [[Paper]](https://arxiv.org/abs/2502.00931)
  [[Project-Page]](https://sairlab.org/vlnav/)

- **SpatialVLA: Exploring Spatial Representations for Visual-Language-Action Model**  
  *introduce Ego3D Position Encoding to inject 3D information into the input observations of the visual-language-action model, and propose Adaptive Action Grids to represent spatial robot movement actions with adaptive discretized action grids, facilitating learning generalizable and transferrable spatial action knowledge for cross-robot control.*  
  [[Paper]](https://arxiv.org/abs/2501.15830)
  [[Project-Page]](https://spatialvla.github.io/)
  [[Code]](https://github.com/SpatialVLA/SpatialVLA)

- **Emma-X: An Embodied Multimodal Action Model with Grounded Chain of Thought and Look-ahead Spatial Reasoning**  
  *the Embodied Multimodal Action Model with Grounded Chain of Thought and Look-ahead Spatial Reasoning*  
  [[Paper]](https://arxiv.org/abs/2412.11974)
  [[Code]](https://github.com/declare-lab/Emma-X)

- **RoboSpatial: Teaching Spatial Understanding to 2D and 3D Vision-Language Models for Robotics**  
  *RoboSpatial, a large-scale dataset for spatial understanding in robotics. It consists of real indoor and tabletop scenes, captured as 3D scans and egocentric images, and annotated with rich spatial information relevant to robotics.*  
  [[Paper]](https://arxiv.org/abs/2411.16537)
  [[Project-Page]](https://chanh.ee/RoboSpatial/)
  [[Code]](https://github.com/NVlabs/RoboSpatial)

- **π_0: A Vision-Language-Action Flow Model for General Robot Control**  
  *a general-purpose robot foundation model that is believed this is a first step toward our long-term goal of developing artificial physical intelligence*  
  [[Paper]](https://arxiv.org/abs/2410.24164)
  [[Project-Page]](https://www.physicalintelligence.company/blog/pi0)

- **Latent Action Pretraining from Videos**  
  *an unsupervised method for pretraining Vision-Language-Action (VLA) models without ground-truth robot action labels.*  
  [[Paper]](https://arxiv.org/abs/2410.11758)
  [[Project-Page]](https://latentactionpretraining.github.io/)
  [[Code]](https://github.com/LatentActionPretraining/LAPA)

- **Poliformer: Scaling On-Policy RL with Transformers Results in Masterful Navigators**  
  *PoliFormer (Policy Transformer), an RGB-only indoor navigation agent trained end-to-end with reinforcement learning at scale that generalizes to the real-world without adaptation despite being trained purely in simulation.*  
  [[Paper]](https://arxiv.org/abs/2406.20083)
  [[Project-Page]](https://poliformer.allen.ai/)
  [[Code]](https://github.com/allenai/poliformer)

- **RoboPoint: A Vision-Language Model for Spatial Affordance Prediction for Robotics**  
  *an automatic synthetic data generation pipeline that instruction-tunes VLMs to robotic domains and needs*  
  [[Paper]](https://arxiv.org/abs/2406.10721)
  [[Project-Page]](https://robo-point.github.io/)
  [[Code]](https://github.com/wentaoyuan/RoboPoint)

- **OpenVLA: An Open-Source Vision-Language-Action Model**  
  *OpenVLA builds on a Llama 2 language model combined with a visual encoder that fuses pretrained features from DINOv2 and SigLIP.*  
  [[Paper]](https://arxiv.org/abs/2406.09246)
  [[Project-Page]](https://openvla.github.io/)
  [[Code]](https://github.com/openvla/openvla)

- **Holistic Autonomous Driving Understanding by Bird's-Eye-View Injected Multi-Modal Large Models**  
  *an end-to-end method for efficiently deriving instruction-aware Bird's-Eye-View (BEV) features, language-aligned for large language models, which integrates multi-view, spatial awareness, and temporal semantics*  
  [[Paper]](https://arxiv.org/abs/2401.00988)
  [[Code]](https://github.com/xmed-lab/NuInstruct)

- **SPOC: Imitating Shortest Paths in Simulation Enables Effective Navigation and Manipulation in the Real World**  
  *how that imitating shortest-path planners in simulation produces agents that, given a language instruction, can proficiently navigate, explore, and manipulate objects in both simulation and in the real world using only RGB sensors (no depth map or GPS coordinates).*  
  [[Paper]](https://arxiv.org/abs/2312.02976)
  [[Project-Page]](https://spoc-robot.github.io/)
  [[Code]](https://github.com/allenai/spoc-robot-training)

- **Open X-Embodiment: Robotic Learning Datasets and RT-X Models**  
  *models output robot actions represented with respect to the robot gripper frame. The robot action is a 7-dimensional vector consisting of x, y, z, roll, pitch, yaw, and gripper opening or the rates of these quantities.*  
  [[Paper]](https://arxiv.org/abs/2310.08864)
  [[Project-Page]](https://robotics-transformer-x.github.io/)
  [[Code]](https://github.com/google-deepmind/open_x_embodiment)

### 4.2 GIScience/Geo AI

### 4.3 Medicine

### 4.4 AR/VR/XR

### 4.5 Beyond AI

### 4.6 Integrated apps

#### 4.6.1 World Model

- **Improving Transformer World Models for Data-Efficient RL**  
  *an approach to model-based RL that achieves a new state of the art performance on the challenging Craftax-classic benchmark, an open-world 2D survival game that requires agents to exhibit a wide range of general abilities*  
  [[Paper]](https://arxiv.org/abs/2502.01591)
  [[Code]](https://github.com/lucidrains/improving-transformers-world-model-for-rl?tab=readme-ov-file)

- **3D-VLA: A 3D Vision-Language-Action Generative World Model**  
  *propose 3D-VLA by introducing a new family of embodied foundation models that seamlessly link 3D perception, reasoning, and action through a generative world model.*  
  [[Paper]](https://arxiv.org/abs/2403.09631)
  [[Project-Page]](https://vis-www.cs.umass.edu/3dvla/)

#### 4.6.2 Others

- **Spatial Speech Translation: Translating Across Space With Binaural Hearables**  
  *a novel concept for hearables that translate speakers in the wearer's environment, while maintaining the direction and unique voice characteristics of each speaker in the binaural output.*  
  [[Paper]](https://arxiv.org/abs/2504.18715)
  [[Code]](https://github.com/chentuochao/Spatial-Speech-Translation)

## Other

- **Fei-Fei Li: With Spatial Intelligence, AI will Understand the Real World**  
  [[Vedio-Page]](https://radical.vc/fei-fei-li-with-spatial-intelligence-ai-will-understand-the-real-world/)

## Reference Repository

- [Awesome-Spatial-Reasoning](https://github.com/yyyybq/Awesome-Spatial-Reasoning)
- [awesome-spatial-reasoning](https://github.com/arijitray1993/awesome-spatial-reasoning)
- [awesome-3d-spatial-reasoning](https://github.com/wufeim/awesome-3d-spatial-reasoning)
- [Awesome-Spatial-Intelligence (for Robotics)](https://github.com/lif314/Awesome-Spatial-Intelligence)
- [Awesome-Visual-Spatial-Intelligence](https://github.com/bobochow/Awesome-Visual-Spatial-Intelligence)
- [Awesome-VLA-Robotics](https://github.com/Jiaaqiliu/Awesome-VLA-Robotics)
