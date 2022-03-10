#!/bin/bash

set -x

## paths to training and development datasets
src_ext=src
trg_ext=trg
train_data_prefix=path/to/bea/train
dev_data_prefix=path/to/bea/dev
dev_data_m2=path/to/bea/dev.all.m2
test_root=path/to/conll14_tok.src
# path to subword nmt
SUBWORD_NMT=path/to/subword-nmt
SCRIPTS_DIR=gec_scripts

######################
# subword segmentation
mkdir -p data/bpe_model
bpe_operations=30000
cat $train_data_prefix.tok.$trg_ext | $SUBWORD_NMT/learn_bpe.py -s $bpe_operations > data/bpe_model/train.bpe.model
mkdir -p data/bea_processed/
$SUBWORD_NMT/apply_bpe.py -c data/bpe_model/train.bpe.model < $train_data_prefix.tok.$src_ext > data/bea_processed/train.all.src
$SUBWORD_NMT/apply_bpe.py -c data/bpe_model/train.bpe.model < $train_data_prefix.tok.$trg_ext > data/bea_processed/train.all.trg
$SUBWORD_NMT/apply_bpe.py -c data/bpe_model/train.bpe.model < $dev_data_prefix.tok.$src_ext > data/bea_processed/dev.src
$SUBWORD_NMT/apply_bpe.py -c data/bpe_model/train.bpe.model < $dev_data_prefix.tok.$trg_ext > data/bea_processed/dev.trg
$SUBWORD_NMT/apply_bpe.py -c data/bpe_model/train.bpe.model < $test_root > data/bea_processed/test.src
$SUBWORD_NMT/apply_bpe.py -c data/bpe_model/train.bpe.model < $test_root > data/bea_processed/test.trg
cp $dev_data_m2 data/bea_processed/dev.m2
cp $dev_data_prefix.all.tok.$src_ext data/bea_processed/dev.input.txt

##########################
#  getting annotated sentence pairs only
python $SCRIPTS_DIR/get_diff.py  processed/train.all src trg > processed/train.annotated.src-trg
cut -f1  processed/train.annotated.src-trg > processed/train.src
cut -f2  processed/train.annotated.src-trg > processed/train.trg


#########################
python3 preprocess.py --source-lang src --target-lang trg --trainpref data/bea_processed/train --validpref data/bea_processed/dev --testpref  data/bea_processed/test --destdir data-bin/BEA/ --joined-dictionary

