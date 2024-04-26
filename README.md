# EENG30009 Individual Research Project 3

---

## Table of Contents

- [Title](#title)
- [Guidance](#guidance)
- [Enviornment](#environment)
- [Abstract](#abstract)
- [Contribution](#contribution)

---

## Title

A performance analysis of Single-image Super-resolution by using ESRGANs (SRGANs) up-sampling in bit-rate reduction scenarios

---

## Guidance

This project is Python-based and MATLAB-based. All reference and guidance are quoted in the scripts. Some scripts do not have reference or guidance, because they are draft scripts which are used for a simple demonstration purpose. All scripts and experimental results will remain open source. Please feel free to download or clone this repository. Anyone is welcome to use or modify these codes and scripts for academic research purpose.

---

## Environment

Programming environment: WSL Linux subsystem of Windows

Python version: 3.7.9

Python setup is based on 'pyenv' (Simple Python version management)

https://github.com/pyenv/pyenv

---

## Abstract

Maintaining high-quality images when reducing image file size is one of the research directions in image processing and computer vision nowadays. Reducing file size means that the static bit-rate or data rate is decreased and the information is lost, which usually leads to distortion and degradation in original images. These distortions are often lossy and cannot be recovered. To address this issue, the image down-sampling/re-sampling + ESRGANs (SRGANs) up-sampling (+ Bilinear up-sampling) system is introduced, and this project examines the performance of this system and compares it with JPEG image compression. The main limitation of this system is that it cannot reduce image file size and maintain high quality simultaneously, but it might be a possible technique to replace JPEG image compression under specific situations. This system down-scales images by image down-sampling/re-sampling methods to reduce file size and static bit-rate and then up-samples images for recovery by the ESRGANs (SRGANs) model. There is a latent layer of Bilinear up-sampling to fix tiny pixel loss, which is to restore the small mismatch between the original image and the distorted image in image down-sampling/re-sampling + ESRGANs (SRGANs) up-sampling system. There are multiple image quality assessment (IQA) metrics used to analyse the performance of this system and JPEG image compression, including objective full-reference metrics, objective no-reference metrics, and no-reference metrics. After experiments, the Pixel area relation re-sampling + ESRGANs (SRGANs) up-sampling (+ Bilinear up-sampling) system is a good method to replace extreme JPEG image compression under specific circumstances.

---

## Contribution

Supervisor:

Dr Dimitris Agrafiotis d.agrafiotis@bristol.ac.uk

Contributor:

Yumu Xie po21744@bristol.ac.uk

EENG30009 Individual Research Project 3

School of Electrical, Electronic and Mechanical Engineering

University of Bristol

---
