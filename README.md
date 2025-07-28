# UCM-NetV2

This is the official code repository for:

**"UCM-NetV2 and BNN-UCM-NetV2: Efficient and Accurate Deep Learning Models for Skin Lesion Segmentation on Mobile Devices"**

---

## 🧠 Overview

**UCM-NetV2** is a lightweight, accurate segmentation model designed to run efficiently on mobile and edge devices.  
It is an **improved version** of [UltraLight-VM-UNet](https://github.com/wurenkai/UltraLight-VM-UNet), featuring enhanced performance, better generalization, and added support for **fine-tuning** via `.npy` datasets.

This repository includes:
- UCM-NetV2 architecture implementation
- Scripts for training and testing
- A **Colab-ready fine-tuning notebook**
- Utilities to merge datasets into `.npy` format

---

## 📦 Dataset Preparation

We use both **ISIC2017** and **ISIC2018** datasets.  
You can download them from the following links:

- 📥 [ISIC2017 Download](https://cuny547-my.sharepoint.com/:u:/g/personal/cyuan1_gradcenter_cuny_edu/EcNlo_WWGVJBtAf_F6AFrrABmSAJi9J3TlHwrnsW8ccOPw?e=NRScI8)
- 📥 [ISIC2018 Download](https://cuny547-my.sharepoint.com/:u:/g/personal/cyuan1_gradcenter_cuny_edu/EZr0xZCCRJZBv9HW4dYWr94BlHmD_9dg6k1QbnzMmH6Asw?e=RVDise)

---

## 🔁 Merge Datasets into `.npy`

To fine-tune or test UCM-NetV2 using the provided notebook, you'll need to **merge ISIC2017 and ISIC2018** into single `.npy` files.

You can use our merging script:

```bash
python merge_datasets.py
