# RSISC_SSL_Paradigm

Code for the paper:"Remote Sensing Image Scene Classification With Self-Supervised Paradigm Under Limited Labeled Samples" by Chao Tao, Ji Qi, Weipeng Lu, Hao Wang and Haifeng Li.


## Dependencies
```
python3
pytorch >= 1.1
gdal >= 3.0
```

## Running
Once the data set is prepared, set the dir of your dataset at `config\dataset_cfg\opt_AID.py`, `config\dataset_cfg\opt_NR.py` and `config\dataset_cfg\opt_EuroSAT_MS.py` and so on.

Do SSL pretraining use `main_cls_ss_train.py`.
Examples:
```shell
# 910 - SimCLR_org on NR dataset
python main_cls_ss_train.py --mode 910 \
  --ds nr --ts 1 \
  --netnum 020000 \
  --lr 1e-4 --lr_policy cosine --mepoch 400 --bs 256 \
  --loss NTXentLoss --optimizer adam \
  --pin-memory \
  --aug 3 --aug_p S5 \
  --mp-distributed --dist-url 'tcp://localhost:10010'

# 910 - SimCLR_org on EuroSAT dataset (with all 13 bands)
python main_cls_ss_train.py --mode 910 \
  --ds euroms --ts 1 \
  --netnum 020000 \
  --lr 1e-4 --lr_policy cosine --mepoch 400 --bs 256 \
  --loss NTXentLoss --optimizer adam \
  --pin-memory \
  --aug 3 --aug_p S5 \
  --mp-distributed --dist-url 'tcp://localhost:10010'
```

Do fine-tuning / training / evaluating use `main_cls.py`
```shell

# finetune the pretrained model (with expnum is 20186093531_pretrain) on AID dataset using 5/class and 20/class samples
for train_scales in 5 20
do
    python main_cls.py --ds aid --ts $train_scales --mode 41 \
		--lr 4e-4 --lr_policy cosine --mepoch 200 \
		--wd 0 --hos 9 \
		--bs 64 --optimizer adam \
		--netnum 020000 --expnum 20186093531_pretrain --cepoch 400  # SimCLR 910
done
```

## Pretrained models
Uploading...


## Citing this work
[1]	C. Tao, J. Qi, W. Lu, H. Wang, and H. Li, ‘Remote Sensing Image Scene Classification With Self-Supervised Paradigm Under Limited Labeled Samples’, IEEE Geoscience and Remote Sensing Letters, pp. 1–5, 2020, doi: 10.1109/lgrs.2020.3038420.

@article{tao_remote_2020,
	title = {Remote Sensing Image Scene Classification With Self-Supervised Paradigm Under Limited Labeled Samples},
	copyright = {All rights reserved},
	issn = {1558-0571},
	doi = {10.1109/lgrs.2020.3038420},
	journal = {IEEE Geoscience and Remote Sensing Letters},
	author = {Tao, Chao and Qi, Ji and Lu, Weipeng and Wang, Hao and Li, Haifeng},
	year = {2020},
	pages = {1--5},
}
