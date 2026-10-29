# Trained Model Checkpoints Manifest

This document records the cryptographic SHA-256 hashes, file sizes, and architecture metadata for all pre-trained FP32 baseline checkpoints provided with this repository.

---

## 1. Summary

- **Total Checkpoints:** 28 files (25 evaluation baseline models across 5 architectures x 5 seeds, plus 3 auxiliary topology checkpoints).
- **Total Storage Size:** 19.65 MB (strictly beneath the 25 MB release asset threshold).
- **Format:** PyTorch state dictionary archives (`.pt`), containing network weights and execution time metadata.
- **Availability:** Fully included in the public release bundle under `results/checkpoints/`.

---

## 2. Cryptographic Checksums (SHA-256)

| File Name | Size (Bytes) | SHA-256 Checksum |
| :--- | :---: | :--- |
| `fp32_cifar100_mobilenet_small_seed42.pt` | 703,778 | `4330fe2ecb7468c82165596a482e393fc87ed454c1fa363c5a70d20b895311ff` |
| `fp32_cifar100_resnet8_2x_seed42.pt` | 1,304,654 | `439dd22d32f153c7c4ed169493ee55f1ead5198c264dd9724673b2f0802d7060` |
| `fp32_cifar10_noaug_separable_cnn_seed42.pt` | 2,059,476 | `659c3838b95ce859e0337393f45c06b283a421b990883bda083c7ea7301a3b6e` |
| `fp32_cifar10_resnet8_seed42.pt` | 332,702 | `ed148d774a4f3f12b75df2fa09f60c9b10076e2166d9fb49e9c7f57fe161c1ed` |
| `fp32_cifar10_resnet8_seed43.pt` | 332,702 | `473ea6c31ec775d220679e253a481022cf774c1b266a5f73f53c1ad46999c937` |
| `fp32_cifar10_resnet8_seed44.pt` | 332,702 | `9234e33ab151fff84e5a3ddeca9b774223fbe09951b13f7bef7ba9cd4950c578` |
| `fp32_cifar10_resnet8_seed45.pt` | 332,702 | `0c12c330ac036007b8e32d49a72190435c2c0190f8187bde2dcbaf5949319e98` |
| `fp32_cifar10_resnet8_seed46.pt` | 332,702 | `c440e62347f4587dc32ed8040357a6d5e03c56b29ae186766e7b83857ee214dc` |
| `fp32_cifar10_separable_cnn_seed42.pt` | 2,059,242 | `20114d2641dc08cdc66733f31993846f16cb6cb495560ff429eaba8f381a887c` |
| `fp32_cifar10_separable_cnn_seed43.pt` | 2,059,242 | `97df5c4195a532b0e4f493526243990b29281f9e8b9c114e55d05f894d832537` |
| `fp32_cifar10_separable_cnn_seed44.pt` | 2,059,242 | `ccac65c431c1d35415167aa11389f3891246cc3c2eb532ecf71545b1d5ce1c26` |
| `fp32_cifar10_separable_cnn_seed45.pt` | 2,059,242 | `59517a8b582b4c61f62a4d2e876062be65351bfa3fcc3b9f0ee479688812ca27` |
| `fp32_cifar10_separable_cnn_seed46.pt` | 2,059,242 | `d0159dd2b272f6e9942499a9aaec9ece6859cadc6955782235d5083652b0b085` |
| `fp32_fmnist_fmnist_cnn_seed42.pt` | 251,472 | `1747beb4c21e583ad7fbb214780b53d9c3b06eeef3685245522f2203bd9a642d` |
| `fp32_fmnist_fmnist_cnn_seed43.pt` | 251,472 | `556214366aa6e2a2bae0c0e932837250e4fc9cb00c1229a6374189dca9d364ca` |
| `fp32_fmnist_fmnist_cnn_seed44.pt` | 251,472 | `a7a0f4dd585525529c5fbae00c241e29fd7cae81e82726b8d18170d7c6b4d2ab` |
| `fp32_fmnist_fmnist_cnn_seed45.pt` | 251,472 | `edab9d0705cc30ef2919bfaea1020d292a69a713f9293a043cf58d64af95eb55` |
| `fp32_fmnist_fmnist_cnn_seed46.pt` | 251,472 | `f3e15e3e6956134c432010de43f687abe1aa581424ece5a0764d08835f3ebc65` |
| `fp32_har_cnn1d_seed42.pt` | 549,888 | `bea3151e61e1c9d853a97dfb730ca1fece817977745532e093541d7d78a1b51d` |
| `fp32_har_cnn1d_seed43.pt` | 549,888 | `0e21027e9de54c1a2f9af9d11cc31fa3d24c13da4344e330d01faa42ee02138f` |
| `fp32_har_cnn1d_seed44.pt` | 549,888 | `5b956bb79b72dec81a81eff2b3287ab124df796ff0b8b022e76e1639a89d1661` |
| `fp32_har_cnn1d_seed45.pt` | 549,888 | `82addf9864f944db0bca0286154a19fc3de182c7b281ec4354d8c6b40456809f` |
| `fp32_har_cnn1d_seed46.pt` | 549,888 | `a6768c776b1fbdc1cce0163f99bec6c1b7014cd303b5ac492151f4afbcebeb83` |
| `fp32_speech_dscnn_seed42.pt` | 114,794 | `afd855f835701a5ba026ae17203d67be5e54323403fef27a2c725084c0e2009f` |
| `fp32_speech_dscnn_seed43.pt` | 114,794 | `a430c25f0e2cd34fd28f218271b5eedd4186946a13878f6f6389f4f10d0ee0d7` |
| `fp32_speech_dscnn_seed44.pt` | 114,794 | `6e15cdba569e5726399051ffdc0a6d880bfd494c9b6e330293366b56454e76c3` |
| `fp32_speech_dscnn_seed45.pt` | 114,794 | `a16ec0bd2d661a3ef2ff229b56b48baae8727865775e7358ac0fc4b253eb712a` |
| `fp32_speech_dscnn_seed46.pt` | 114,794 | `8bddd79cf7ba365c277fde1788724cbfe7c404ce3096c5f1e683a46f2ae0e632` |
