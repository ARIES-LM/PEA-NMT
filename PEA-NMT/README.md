1.Train

python rnnsearch.py train --corpus zh.txt en.txt zh.pos en.pos
    --vocab zh.vocab.pkl en.vocab.pkl zh.posvocab.pkl en.posvocab.pkl 
    --model pea --embdim 500 500 150 300 --decoder GruCond
    --hidden 1000 1000 1000 100 500 400 500 400 --deephid 500
    --alpha 5e-4 --norm 1.0 --batch 80 --seed 1234
    --freq 1000 --vfreq 2500 --sfreq 50 --validation nist02.src
    --references nist02.ref0 nist02.ref1 nist02.ref2 nist02.ref3
    --ext_val_script scripts/validate-nist.sh
    --optimizer adam --shuffle 1

2.Decode

python rnnsearch.py translate --model pea.best.pkl --normalize <src 1>translation 2>log.out
