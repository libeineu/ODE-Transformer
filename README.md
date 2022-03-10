# ODE Transformer: An Ordinary Differential Equation-Inspired Model for Sequence Generation
This code is based on Fairseq v0.6.2
## Requirements and Installation
- PyTorch version >= 1.2.0
- python version >= 3.6  

## Prepare Data
### For Machine Translation 

#### 1、Download [WMT14' En-De](https://drive.google.com/uc?export=download&id=0B_bZck-ksdkpM25jRUN2X2UxMm8) and [WMT14' En-Fr](https://github.com/pytorch/fairseq/blob/master/examples/translation/prepare-wmt14en2fr.sh)

#### 2、Preprocessed dataset

### For Abstractive Summarization Task

#### 1、Download [CNN dataset](https://drive.google.com/uc?export=download&id=0BwmD_VLjROrfTHk4NFg2SndKcjQ) and [Daily Mail dataset](https://drive.google.com/uc?export=download&id=0BwmD_VLjROrfM1BxdkxVaTY2bWs)


#### 2、Generate binary dataset ```data-bin/cnndm```

```bash preprocess_cnndaily_bin.sh path/to/cnndm_raw_data```

### For Grammatical Error Correction Task  

  #### 1、Download [FCE v2.1 dataset](https://www.cl.cam.ac.uk/research/nl/bea2019st/data/fce_v2.1.bea19.tar.gz)、[Lang-8 Corpus of Learner English dataset](https://docs.google.com/forms/d/e/1FAIpQLSflRX3h5QYxegivjHN7SJ194OxZ4XN_7Rt0cNpR2YbmNV-7Ag/viewform)、[NUCLE dataset](https://sterling8.d2.comp.nus.edu.sg/nucle_download/nucle.php)、[W&I+LOCNESS v2.1 dataset](https://www.cl.cam.ac.uk/research/nl/bea2019st/data/wi+locness_v2.1.bea19.tar.gz)

  #### 2、Get CONLL14 test set  

  ```bash prepare_conll14_test_data.sh```

  #### 3、Preprocessed dataset  

  ```bash preprocess_gec.sh```

  #### 4、Generate binary dataset  ```data-bin/BEA```

  ```bash preprocess_gec_bin.sh```

## Train
### For WMT'14 En-De Task  

#### Train a RK2-block  $\textrm{learnable}\, \gamma_i$ model (6-layer Big model)

```bash train_wmt_en_de.sh```

```
python3 -u train.py data-bin/$data_dir
  --distributed-world-size 8 -s src -t tgt
  --arch transformer_ode_t2t_wmt_en_de_big
  --share-all-embeddings
  --optimizer adam --clip-norm 0.0
  --adam-betas '(0.9, 0.997)'
  --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 16000
  --lr 0.002 --min-lr 1e-09
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1
  --max-tokens 4096
  --update-freq 4
  --max-epoch 20
  --dropout 0.3 --attention-dropout 0.1 -- relu-dropout 0.1
  --no-progress-bar
  --log-interval 100
  --ddp-backend no_c10d
  --seed 1 
  --save-dir $save_dir
  --keep-last-epochs 10
```



### For WMT'14 En-Fr Task

#### Train a RK2-block  $\textrm{learnable}\, \gamma_i$ model

```bash train_wmt_en_fr.sh```

```
python3 -u train.py data-bin/$data_dir
  --distributed-world-size 8 -s src -t tgt
  --arch transformer_ode_t2t_wmt_en_de_big
  --share-all-embeddings
  --optimizer adam --clip-norm 0.0
  --adam-betas '(0.9, 0.997)'
  --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 16000
  --lr 0.002 --min-lr 1e-09
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1
  --max-tokens 4096
  --update-freq 8
  --max-epoch 20
  --dropout 0.1 --attention-dropout 0.1 -- relu-dropout 0.1
  --no-progress-bar
  --log-interval 100
  --ddp-backend no_c10d
  --seed 1 
  --save-dir $save_dir
  --keep-last-epochs 10
```



### For Abstractive Summarization Task  

#### Train a RK2-block  $\textrm{learnable}\, \gamma_i$ model

```bash train_cnn_daily.sh```

```
python3 -u train.py data-bin/$data_dir
  --distributed-world-size 8 -s src -t tgt
  --arch transformer_ode_t2t_wmt_en_de
  --share-all-embeddings
  --optimizer adam --clip-norm 0.0
  --adam-betas '(0.9, 0.997)'
  --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 8000
  --lr 0.002 --min-lr 1e-09
  --weight-decay 0.0001
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1
  --max-tokens 4096
  --update-freq 4
  --max-epoch 20
  --dropout 0.1 --attention-dropout 0.1 -- relu-dropout 0.1
  --truncate-source  --skip-invalid-size-inputs-valid-test --max-source-positions 500
  --no-progress-bar
  --log-interval 100
  --ddp-backend no_c10d
  --seed 1 
  --save-dir $save_dir
  --keep-last-epochs 10
```

### For Grammatical Error Correction Task  

#### Train a RK2-block $\textrm{learnable}\, \gamma_i$ model 
```bash train_gec.sh```

```
python3 -u train.py data-bin/$data_dir
  --distributed-world-size 8 -s src -t tgt
  --arch transformer_ode_t2t_wmt_en_de
  --share-all-embeddings
  --optimizer adam --clip-norm 0.0
  --adam-betas '(0.9, 0.98)'
  --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000
  --lr 0.0015 --min-lr 1e-09
  --weight-decay 0.0001
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1
  --max-tokens 4096
  --update-freq 2
  --max-epoch 55
  --dropout 0.2 --attention-dropout 0.1 -- relu-dropout 0.1
  --no-progress-bar
  --log-interval 100
  --ddp-backend no_c10d
  --seed 1 
  --save-dir $save_dir
  --keep-last-epochs 10
  --tensorboard-logdir $save_dir"
```

## Evaluation
### For WMT'14 En-De Task

We measure the performance through multi-bleu and sacrebleu

```
python3 generate.py \
data-bin/wmt-en2de \
--path $model_dir/$checkpoint \
--gen-subset test \
--batch-size 64 \
--beam 4 \
--lenpen 0.6 \
--output hypo.txt \
--quiet \
--remove-bpe
```



### For WMT'14 En-Fr Task

We measure the performance through multi-bleu and sacrebleu

```
python3 generate.py \
data-bin/wmt-en2fr \
--path $model_dir/$checkpoint \
--gen-subset test \
--batch-size 64 \
--beam 4 \
--lenpen 0.6 \
--output hypo.txt \
--quiet \
--remove-bpe
```



### For Abstractive Summarization Task

We use pyrouge as the scoring script. 

```
python3 generate.py \
data-bin/$data_dir \
--path $model_dir/$checkpoint \
--gen-subset test \
--truncate-source \
--batch-size 32 \
--lenpen 2.0 \
--min-len 55 \
--max-len-b 140 \
--max-source-positions 500 \
--beam 4 \
--no-repeat-ngram-size 3 \
--remove-bpe

python3 get_rouge.py --decodes_filename cnndm.test.target.tok --targets_filename $model_dir/hypo.sorted.tok
```

### For Grammatical Error Correction Task
We use m2scorer as the scoring script. 

```
python3 generate.py \
data-bin/$data_dir \
--path $model_dir/$checkpoint \
--gen-subset test \
--batch-size 64 \
--beam 4 \
--lenpen 2.0 \
--output hypo.txt \
--quiet \
--remove-bpe

path/to/m2scorer path/to/model_output path/to/conll14st-test.m2
```


## Results
### Machine Translation

| Model                            | Layer | En-De | En-Fr |
| -------------------------------- | ----- | ----- | ----- |
| Residual-block (baseline)        | 6-6   | 29.21 | 42.89 |
| RK2-block (learnable $\gamma_i$) | 6-6   | 30.53 | 43.59 |
| Residual-block (baseline)        | 12-6  | 29.91 | 43.22 |
| RK2-block (learnable $\gamma_i$) | 12-6  | 30.76 | 44.11 |

### Abstractive Summarization Task

| Model                             | RG-1 | RG-2 | RG-L |
| --------------------------------- | ---- | ---- | ---- |
| Residual-block                    | 40.47 | 17.73 | 37.29 |
| RK2-block ((learnable $\gamma_i$) | 41.58 | 18.57 | 38.41 |
| RK4-block                         | 41.83 | 18.84 | 38.68 |

### Grammatical Error Correction Task

|   Model  |  Prec.  |  Recall | F_0.5 |
|  ----  |  ----  | ----  | ---- |
| Residual-block  | 67.97 | 32.17 |55.61 |
| RK2-block ((learnable $\gamma_i$) | 68.21 | 35.30 |57.49 |
| RK4-block | 66.20  | 38.13 |57.71 |

