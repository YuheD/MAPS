# Official Implementation for MAPS
[MAPS: A Noise-Robust Progressive Learning Approach for Source-Free Domain Adaptive Keypoint Detection](https://arxiv.org/abs/2302.04589)

This implementation is based on [UDA_PoseEstimation](https://github.com/VisionLearningGroup/UDA_PoseEstimation).
### Framework:  

1. train on the source domain;
2. Construct the proxy source domain and train on target dataset.

<img src="figs/net.png" width="800"/>

### Dataset:

- Please put the hand datasets H3D and RHD under the folder './hand_data/', put the human datasets LSP and SURREAL under the folder './human_data'

### Training:
1. ##### Hand dataset
	```python
    # train source model
    python hand_src.py 
    # train target model
    python hand_tgt_proxy.py
	```
2. ##### Human dataset
	```python
    # train source model
    python human_src.py
    # train target model
    python human_tgt_proxy.py
	```


### Citation

If you find this code useful for your research, please cite our paper
```
@article{ding2023maps,
  title={MAPS: A Noise-Robust Progressive Learning Approach for Source-Free Domain Adaptive Keypoint Detection},
  author={Ding, Yuhe and Liang, Jian and Jiang, Bo and Zheng, Aihua and He, Ran},
  journal={arXiv preprint arXiv:2302.04589},
  year={2023}
}
