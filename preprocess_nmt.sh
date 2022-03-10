src=en
tgt=de
TEXT=../wmt-en2de
tag=wmt-en2de
output=data-bin/$tag


python3 preprocess.py --source-lang $src --target-lang $tgt --trainpref $TEXT/train  --validpref $TEXT/valid --testpref $TEXT/test --destdir $output --workers 32
