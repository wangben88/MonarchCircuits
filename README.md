# Scaling Probabilistic Circuits via Monarch Matrices

This is the official implementation of the ICML 2025 paper [Scaling Probabilistic Circuits via Monarch Matrices](https://www.arxiv.org/pdf/2506.12383).

To reproduce the experiments on the image datasets, first [download](https://www.image-net.org/) the ImageNet32/64 npz files to `../datasets`. Then, run e.g. the following command:

```python experiments/run_image_exp.py -ly monarch -hs 1024 -ds imagenet32```

Code for reproducing the Text8 and LM1B text dataset results will be added to this repository soon.

## Citation
    @inproceedings{ZhangICML25,
        title     = {Scaling Probabilistic Circuits via Monarch Matrices},
        author    = {Zhang, Honghua and Dang, Meihua and Wang, Benjie and Ermon, Stefano and Peng, Nanyun and Van den Broeck, Guy},
        booktitle = {Proceedings of the 42th International Conference on Machine Learning (ICML)},
        month     = {jul},
        year      = {2025},
    }