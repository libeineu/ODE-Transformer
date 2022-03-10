src=source
tgt=target
TEXT=${1}
output=data-bin/cnndm


python3 preprocess.py \
    -s $src \
    -t $tgt \
    --trainpref ${TEXT}cnndm.train \
    --testpref ${TEXT}cnndm.test \
    --validpref ${TEXT}cnndm.dev \
    --destdir $output --workers 60 \
    --joined-dictionary



