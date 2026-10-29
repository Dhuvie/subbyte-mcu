# Dataset Documentation and Licensing

This repository evaluates mixed-precision quantization across five diverse TinyML benchmarks spanning computer vision, speech recognition, and sensor time-series classification.

---

## Benchmark Datasets

### 1. CIFAR-10
- **Domain:** Computer Vision (10-class object recognition)
- **Input Dimensions:** $32 \times 32 \times 3$ RGB images
- **Source:** Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009.
- **URL:** [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)
- **License:** MIT / Open Access (University of Toronto)

### 2. CIFAR-100
- **Domain:** Computer Vision (100-class fine-grained recognition)
- **Input Dimensions:** $32 \times 32 \times 3$ RGB images
- **Source:** Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009.
- **URL:** [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)
- **License:** MIT / Open Access (University of Toronto)

### 3. Fashion-MNIST
- **Domain:** Computer Vision (10-class grayscale fashion item recognition)
- **Input Dimensions:** $28 \times 28 \times 1$ grayscale images
- **Source:** Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms, Han Xiao, Kashif Rasul, Roland Vollgraf, 2017.
- **URL:** [https://github.com/zalandoresearch/fashion-mnist](https://github.com/zalandoresearch/fashion-mnist)
- **License:** MIT License (Zalando SE)

### 4. Google Speech Commands v2
- **Domain:** Audio Keyword Spotting (12 classes: 10 command keywords + silence + unknown)
- **Input Dimensions:** $49 \times 10$ Mel-frequency cepstral coefficients (MFCCs) computed from 1-second 16\,kHz audio clips
- **Source:** Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition, Pete Warden, 2018.
- **URL:** [https://arxiv.org/abs/1804.03209](https://arxiv.org/abs/1804.03209)
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)

### 5. UCI Human Activity Recognition (HAR)
- **Domain:** Sensor Time-Series Classification (6 physical activities)
- **Input Dimensions:** 128 time steps $\times$ 9 inertial measurement signals (body acceleration, total acceleration, angular velocity)
- **Source:** A Public Domain Dataset for Human Activity Recognition Using Smartphones, Davide Anguita et al., ESANN 2013.
- **URL:** [https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)

---

## Test-Set Quarantine Protocol

To ensure absolute methodological validity, all datasets are strictly split into:
- 80% Training Split
- 10% Validation Split (used exclusively for layer sensitivity profiling and knapsack optimization)
- 10% Test Split (quarantined programmatically via `subbyte.training.guard.GLOBAL_GUARD`)

The test split cannot be accessed during sensitivity sweeps, heuristic ranking, or knapsack allocation. It is evaluated strictly once on the final packed `.qpack` artifact.
