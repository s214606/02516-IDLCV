# 1 - SSH

Use `ssh-keygen` to make a private key and public key and name the file `hpc` or anything you like.

Make a config to remember in `~/.ssh/config`:
```bash
Host hpc
    User sxxxxxx
    IdentityFile ~/.ssh/hpc
    Hostname login1.hpc.dtu.dk
```

To connect to the HPC, use `ssh hpc` if you have set up the config, otherwise `ssh sxxxxxx@login1.hpc.dtu.dk`

Remember to:
    - add your public ssh key `hpc.pub` to `~/.ssh/authorized_keys`
    - go to compute node `linuxsh` or interactive gpu `02516sh` before submitting jobs or running scripts.



# 2 - Copy a file
Here's three options including the one Dimitrios presented:

    - [FileZilla](https://filezilla-project.org/) which is a free FTP tool
    - Copy file over VScode but I can't recommend big files or you'll hate yourself
    - SCP: `scp DTU_logo.png hpc:02516-IDLCV/content/part-2-video/1-using-hpc/DTU_logo.png`

# 3 - Virtual Environment
Docs: https://www.hpc.dtu.dk/?page_id=3678

Use `module avail` to see available modules and pick yours e.g

1. Load Python module`module load python3/3.13.5`
2. Create virtual environment `python3 -m venv .venv`
3. Activate the virtual environment, run `source .venv/bin/activate`.

Alternatively, use [uv](https://docs.astral.sh/uv/getting-started/installation/)



# 4 - Simple script
```

```

# 5 Run script interactively using GPU
To go into interactive node with course gpu, use `02516sh`.

For [regular gpu](https://www.hpc.dtu.dk/?page_id=2759) use either of:
```
voltash
sxm2sh
a100sh
```

After activating it, check you are using it interactively with `nvidia-smi`.

To run the script use `python3 simple_script.py`



# 6 - Submit a batch job
Make a job script like so:

```sh
#!/bin/bash
#BSUB -J DLCV-part-2-exercise-1
#BSUB -q c02516
#BSUB -gpu "num1:mode=exclusive_process"

#BSUB -N

#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=1GB]"

# Max wall clock time is 12 hours
#BSUB -W 1:00 

#BSUB -o Output_%J.out
#BSUB -e Output_%J.err

source .venv/bin/activate
python simple_script.py
```

And run `bsub -app c02516_1g.10gb < jobscript.sh`
