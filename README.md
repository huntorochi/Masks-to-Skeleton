# Masks-to-Skeleton
Masks-to-Skeleton: Multi-view Mask-Based Tree Skeleton Extraction with 3D Gaussian Splatting
## Xinpeng Liu<sup>1</sup>, Kanyu Xu<sup>1</sup>, Risa Shinoda<sup>1</sup>, Hiroaki Santo<sup>1</sup>, Fumio Okura<sup>1</sup>  <br> <sup>1</sup> The University of Osaka<br>
![Paper](https://www.mdpi.com/1424-8220/25/14/4354)

![Teaser image](assets/teaser.pdf)

Abstract: *Accurately reconstructing tree skeletons from multi-view images is challenging. While most existing works use skeletonization from 3D point clouds, thin branches with low-texture contrast often involve multi-view stereo (MVS) to produce noisy and fragmented point clouds, which break branch connectivity. Leveraging the recent development in accurate mask extraction from images, we introduce a mask-guided graph optimization framework that estimates a 3D skeleton directly from multi-view segmentation masks, bypassing the reliance on point cloud quality. In our method, a skeleton is modeled as a graph whose nodes store positions and radii while its adjacency matrix encodes branch connectivity. We use 3D Gaussian splatting (3DGS) to render silhouettes of the graph and directly optimize the nodes and the adjacency matrix to fit given multi-view silhouettes in a differentiable manner. Furthermore, we use a minimum spanning tree (MST) algorithm during the optimization loop to regularize the graph to a tree structure. Experiments on synthetic and real-world plants show consistent improvements in completeness and structural accuracy over existing point-cloud-based and heuristic baseline methods.*

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@Article{liuHoGS,
        AUTHOR = {Liu, Xinpeng and Xu, Kanyu and Shinoda, Risa and Santo, Hiroaki and Okura, Fumio},
        TITLE = {Masks-to-Skeleton: Multi-View Mask-Based Tree Skeleton Extraction with 3D Gaussian Splatting},
        JOURNAL = {Sensors},
        VOLUME = {25},
        YEAR = {2025},
        NUMBER = {14},
        ARTICLE-NUMBER = {4354},
        URL = {https://www.mdpi.com/1424-8220/25/14/4354},
        PubMedID = {40732481},
        ISSN = {1424-8220},
        DOI = {10.3390/s25144354}
}</code></pre>
  </div>
</section>

## Cloning the Repository

The repository contains submodules, thus please check it out with 
```shell
# SSH
git clone git@github.com:huntorochi/Masks-to-Skeleton.git
```
or
```shell
# HTTPS
git clone https://github.com/huntorochi/Masks-to-Skeleton.git
```

## Overview
This repository contains an **early-release** implementation of *Masks-to-Skeleton*, a multi-view, mask-guided tree skeleton extraction pipeline based on **3D Gaussian Splatting (3DGS)** and **MST-regularized graph optimization**.

## News
- **(Early Release)** This code is released quickly to provide an early baseline version. See **Notes / Limitations** below.

## Dataset

### Download
Please download the dataset from [**Google Drive**](https://drive.google.com/file/d/1ThWaL7e_z-xTZ6IFICSN9UIGXr9LmonM/view?usp=sharing). After downloading, place and unzip the dataset under:
Masks-to-Skeleton/
├─ github_mdpi/
│ ├─ dataset/
