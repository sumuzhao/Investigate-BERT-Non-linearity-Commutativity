## Non-linearity 

In this repository, we use linear approximators to quantify the degree of non-linearity of different components in BERT. 
For easy usage, we use bash scripts to run different experiments.
You need to set the parameters inside the scripts before running experiments.  

First, you need to enter the **non-linearity** folder. 
```bash
cd non-linearity/
```

### Train linear/non-linear approximators

Approximators are defined in approximator.py where you can define your own approximators. 

Main parameters:
- approximate_part
- approximator_setting

```bash
bash run_approximator.sh
```

### Extracting hidden embeddings

```bash
bash run_track.sh
```

### Replacing

Main parameters:
- approximator_checkpoint
- approximator_setting
- layers, single-replacing use `$j`, multi-replacing use `${multi_layer// /,}`

```bash
bash run_replace.sh
```

### Removing

Main parameters:
- remove_part
- freeze_part
- layers, single-replacing use `$j`, multi-replacing use `${multi_layer// /,}`

```bash
bash run_remove.sh
```

### Freezing

Main parameters:
- freeze_part
- layers, single-replacing use `$j`, multi-replacing use `${multi_layer// /,}`

```bash
bash run_freeze.sh
```
