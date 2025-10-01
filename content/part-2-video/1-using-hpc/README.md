# 3
```
python -m venv .venv
```

# 5
To go into interactive node with course gpu, use
```
02516sh
```
for [regular gpu](https://www.hpc.dtu.dk/?page_id=2759) use either of:
```
voltash
sxm2sh
a100sh
```

# 6
```sh
#!/bin/bash
#BSUB -J DLCV-project2
#BSUB -q c02516
#BSUB -N

#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=1GB]"

# Max wall clock time is 12 hours
#BSUB -W 1:00 

#BSUB -o Output_%J.out
#BSUB -e Output_%J.err

#BSUB -gpu "num1:mode=exclusive_process"


source .venv/bin/activate
python project2.py
```

And run
```
#bsub -app c02516_1g.10gb < submit.sh
```