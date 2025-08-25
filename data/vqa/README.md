---
dataset_info:
  features:
  - name: UID
    dtype: string
  - name: Question
    dtype: string
  - name: Answer
    dtype: string
  - name: Type
    dtype: string
  - name: PatientID
    dtype: string
  - name: Age
    dtype: int64
  - name: HeartSize
    dtype: int64
  - name: PulmonaryCongestion
    dtype: int64
  - name: PleuralEffusion_Right
    dtype: int64
  - name: PleuralEffusion_Left
    dtype: int64
  - name: PulmonaryOpacities_Right
    dtype: int64
  - name: PulmonaryOpacities_Left
    dtype: int64
  - name: Atelectasis_Right
    dtype: int64
  - name: Atelectasis_Left
    dtype: int64
  - name: Split
    dtype: string
  - name: PhysicianID
    dtype: string
  - name: StudyDate
    dtype: string
  - name: Sex
    dtype: string
  - name: Image
    dtype: image
  splits:
  - name: train
    num_bytes: 5622656901.0
    num_examples: 20288
  - name: val
    num_bytes: 1462315894.0
    num_examples: 5120
  - name: test
    num_bytes: 1783934753.0
    num_examples: 6592
  download_size: 363809891
  dataset_size: 8868907548.0
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: val
    path: data/val-*
  - split: test
    path: data/test-*
---
