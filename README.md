# CDMT：Cross-Dependence Multi-task Transformer For Neonatal Hip Bone Intelligent Diagnosis

## Introduction 

This project contains the open-source code and dataset for the paper titled "CDMT：Cross-Dependence Multi-task Transformer For Neonatal Hip Bone Intelligent Diagnosis"

## Dataset 
The provided dataset includes 1,000 labeled images, designed to support hip joint multi-task learning and related research. The dataset is inside the HioJoint.rar file in the 'data' directory. Please extract the file in that directory.

## Requirements
* python=3.8
* pytorch=2.4.1
* mmsegmentation v1.2.0
* mmpose v1.3.2

Note: mmsegmentation and mmpose are only required for the baselines of bone segmentation and landmark localization, respectively.

## Project Structure

* CDMT : code for multi-task learning
* mmpose: code for landmark localization baselines
* mmsegmentation: code for bone segmentation baselines
* data: hip joint dataset
  * images: Ultrasound images of hip joint
  * annatations: labels for landmark
  * labels: labels for bone segmentation




