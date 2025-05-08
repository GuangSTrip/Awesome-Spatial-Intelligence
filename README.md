<h1 align="center">Awesome-Spatial-Intelligence</h1>

## Introduction

Spatial Intelligence is becoming increasingly important in the field of Artificial Intelligence. This repository aims to provide a comprehensive and systematic collection of research related to Spatial Intelligence.

Any suggestion is welcome, please feel free to raise an issue. ^_^

## Table of Contents

- [Introduction](#introduction)
- [Table of Contents](#table-of-contents)
- [1. Spatial Intelligence in various areas/tasks](#1-spatial-intelligence-in-various-areastasks)
  - [1.1 NLP](#11-nlp)
  - [1.2 CV](#12-cv)
  - [1.3 Multi-modal](#13-multi-modal)
  - [1.4 Others (radar/GPS etc.)](#14-others-radargps-etc)
- [2. Datasets and Benchmarks](#2-datasets-and-benchmarks)
- [3. Spatial Intelligence Methods](#3-spatial-intelligence-methods)
  - [3.1 old/mathematical/rule-based](#31-oldmathematicalrule-based)
  - [3.2 Machine Learning](#32-machine-learning)
  - [3.3 deep learning](#33-deep-learning)
  - [3.4 LLM](#34-llm)
- [4. Application](#4-application)
  - [4.1 Robotics](#41-robotics)
  - [4.2 GIScience/Geo AI](#42-gisciencegeo-ai)
  - [4.3 Medicine](#43-medicine)
  - [4.4 AR/VR/XR](#44-arvrxr)
  - [4.5 Beyond AI](#45-beyond-ai)
  - [4.6 Integrated apps](#46-integrated-apps)
- [Reference Repository](#reference-repository)

## 1. Spatial Intelligence in various areas/tasks

### 1.1 NLP

### 1.2 CV

**<div style="text-align: center;">Understanding</div>**

- **SpatialSense: An Adversarially Crowdsourced Benchmark for Spatial Relation Recognition**  
  *a dataset specializing in spatial relation recognition which captures a broad spectrum of such challenges, allowing for proper benchmarking of computer vision techniques.*  
  [[Paper]](https://arxiv.org/abs/1908.02660)
  [[Code]](https://github.com/princeton-vl/SpatialSense)

**<div style="text-align: center;">3D Reconstruction</div>**

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
  **  
  [[Paper]](https://arxiv.org/abs/2404.03658)
  [[Project-Page]](https://ruili3.github.io/kyn/)
  [[Code]](https://github.com/ruili3/Know-Your-Neighbors)

- **DUSt3R: Geometric 3D Vision Made Easy**  
  *DUSt3R, a radically novel paradigm for Dense and Unconstrained Stereo 3D Reconstruction of arbitrary image collections, i.e. operating without prior information about camera calibration nor viewpoint poses.*  
  [[Paper]](https://arxiv.org/abs/2312.14132)
  [[Project-Page]](https://dust3r.europe.naverlabs.com/)
  [[Code]](https://github.com/naver/dust3r)

---
**<div style="text-align: center;">Scene Generation</div>**

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

---
**<div style="text-align: center;">Spatial Simulation</div>**

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

### 1.3 Multi-modal

**<div style="text-align: center;">Understanding</div>**

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

### 3.2 Machine Learning

### 3.3 deep learning

- **Can Transformers Capture Spatial Relations between Objects?**  
  *approaches exploiting the long-range attention capabilities of transformers for spatial relationships task*  
  [[Paper]](https://arxiv.org/abs/2403.00729)
  [[Project-Page]](https://sites.google.com/view/spatial-relation)
  [[Code]](https://github.com/AlvinWen428/spatial-relation-benchmark)

### 3.4 LLM

**<div style="text-align: center;">Reasoning</div>**

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
  *provide a flawless solution to the benchmark by combining template-to-relation mapping with logic-based reasoning. *  
  [[Paper]](https://arxiv.org/abs/2401.03991)
  [[Code]](https://github.com/Fangjun-Li/SpatialLM-StepGame)

**<div style="text-align: center;">Recognition</div>**

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

---
**<div style="text-align: center;">Reinforcement Learning</div>**

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

**<div style="text-align: center;">World Model</div>**

- **Improving Transformer World Models for Data-Efficient RL**  
  *an approach to model-based RL that achieves a new state of the art performance on the challenging Craftax-classic benchmark, an open-world 2D survival game that requires agents to exhibit a wide range of general abilities*  
  [[Paper]](https://arxiv.org/abs/2502.01591)
  [[Code]](https://github.com/lucidrains/improving-transformers-world-model-for-rl?tab=readme-ov-file)

- **3D-VLA: A 3D Vision-Language-Action Generative World Model**  
  *propose 3D-VLA by introducing a new family of embodied foundation models that seamlessly link 3D perception, reasoning, and action through a generative world model.*  
  [[Paper]](https://arxiv.org/abs/2403.09631)
  [[Project-Page]](https://vis-www.cs.umass.edu/3dvla/)

## Reference Repository

- [Awesome-Spatial-Reasoning](https://github.com/yyyybq/Awesome-Spatial-Reasoning)
- [awesome-spatial-reasoning](https://github.com/arijitray1993/awesome-spatial-reasoning)
- [awesome-3d-spatial-reasoning](https://github.com/wufeim/awesome-3d-spatial-reasoning)
- [Awesome-Spatial-Intelligence (for Robotics)](https://github.com/lif314/Awesome-Spatial-Intelligence)
- [Awesome-Visual-Spatial-Intelligence](https://github.com/bobochow/Awesome-Visual-Spatial-Intelligence)
- [Awesome-VLA-Robotics](https://github.com/Jiaaqiliu/Awesome-VLA-Robotics)





- ****  
  **  
  [[Paper]]()
  [[Project-Page]]()
  [[Code]]()

- ****  
  **  
  [[Paper]]()
  [[Project-Page]]()
  [[Code]]()


- ****  
  **  
  [[Paper]]()
  [[Project-Page]]()
  [[Code]]()


- ****  
  **  
  [[Paper]]()
  [[Project-Page]]()
  [[Code]]()


- ****  
  **  
  [[Paper]]()
  [[Project-Page]]()
  [[Code]]()


