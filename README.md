# UCM-NetV2 Improvement

This repository presents an **improved version of UCM-NetV2**, titled **BNN-UCM-NetV2**, featuring enhanced efficiency and accuracy for skin lesion segmentation, especially optimized for mobile devices.

---

## 🧠 Overview

This project builds upon the original **UCM-NetV2** model and improves its architecture and training pipeline for better performance.  
While some code components such as dataset generation and loading are inspired by [UltraLight-VM-UNet](https://github.com/wurenkai/UltraLight-VM-UNet), the core model and methodology here represent a significant advancement over UCM-NetV2.

A fine-tuning notebook is included to facilitate testing and adaptation on customized datasets.

---

## 📦 Dataset Preparation

The model is trained and tested on the **ISIC2017** and **ISIC2018** datasets.

Download the datasets here:

- [ISIC2017](https://cuny547-my.sharepoint.com/:u:/g/personal/cyuan1_gradcenter_cuny_edu/EcNlo_WWGVJBtAf_F6AFrrABmSAJi9J3TlHwrnsW8ccOPw?e=NRScI8)
- [ISIC2018](https://cuny547-my.sharepoint.com/:u:/g/personal/cyuan1_gradcenter_cuny_edu/EZr0xZCCRJZBv9HW4dYWr94BlHmD_9dg6k1QbnzMmH6Asw?e=RVDise)

---

## 🔁 Merging ISIC2017 and ISIC2018

To test the improved model, **merge** ISIC2017 and ISIC2018 datasets into unified `.npy` files containing images and masks.

A script `merge_datasets.py` is provided to help you combine these datasets into the following format:

