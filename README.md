# MatSpray

<p align="left">
  <strong>
    Fusing 2D Material World Knowledge on 3D Geometry
  </strong>
</p>

⚠️ This is work in progress please don't publish yet (will release Mon. 22 / 00:00 GMT)

<video width="100%" controls autoplay muted loop>
  <source src="https://github.com/user-attachments/assets/0ff4195d-a5dc-4d4f-9ff7-9ea093a46754" type="video/mp4">
</video>

*If the video doesn't load, [click here to view it directly](videos/trailer.mp4)*

<p align="center">
    <span> 🌐  <a href="https://matspray.jdihlmann.com/"> Project Page </a> </span>&nbsp;&nbsp;&nbsp;
    <span> 📄  <a href="http://arxiv.org/abs/"> Paper (Arxiv) </a> </span>&nbsp;&nbsp;&nbsp;
  <span>  📦  <a href="https://drive.google.com/drive/folders/1znN_KllBKllIY_1PLZUHbnfHsB6KNifR?usp=sharing"> Materials </a> </span>&nbsp;&nbsp;&nbsp;
  <span>  ✍🏻
     <a href="https://github.com/cgtuebingen/MatSpray?tab=readme-ov-file#citation"> Citation </a> </span>&nbsp;&nbsp;&nbsp;
</p>

# About

We propose MatSpray, a framework for fusing 2D material world knowledge from diffusion models into 3D geometry to obtain relightable assets with spatially varying physically based rendering (PBR) materials. Our method leverages pretrained 2D diffusion-based material predictors to generate per-view material maps (base color, roughness, metallic) and integrates them into a 3D Gaussian Splatting representation via Gaussian ray tracing. To enhance multi-view consistency and physical accuracy, we introduce a lightweight Neural Merger that refines material estimates using a softmax-based restriction. Our approach enables high-quality relightable 3D reconstruction that outperforms existing methods in both quantitative metrics and visual realism, while being approximately 3.5× faster than state-of-the-art inverse rendering approaches.

# Code

Paper is currently under review, we will release the code shortly in a cleaned version. Up until then if you have any questions regarding the project or need material to compare against fast, feel free to contact us.

# Citation

You can find our paper on [arXiv](https://arxiv.org/), please consider citing, if you find this work useful:

```
@inproceedings{langsteiner2025matspray,
author = {Langsteiner, Philipp and Dihlmann, Jan-Niklas and Lensch, Hendrik P.A.},
title = {MatSpray: Fusing 2D Material World Knowledge on 3D Geometry},
booktitle = {arXiv preprint},
year = {2025}
}
```
