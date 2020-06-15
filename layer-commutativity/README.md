## Layer Commutativity

In this repository, we study the interactions between layers in BERT by swapping or shuffling layers. 
For easy usage, we use bash scripts to run different experiments.
You need to set the parameters inside the scripts before running experiments.  

First, you need to enter the **layer-commutativity** folder. 
```bash
cd layer-commutativity/
```

### Swapping

Approximators are defined in approximator.py where you can define your own approximators. 

Main parameters:
- layers, select the corresponding swapped pairs. 

```bash
bash run_swap.sh
```

### Shuffling

In **shuffle_exhaustive_search**, there are randomly generated layer orders for exhaustive search. 
You can generate new layer orders. 
```bash
python generate_shuffling_order.py
```

Main parameters:
- shuffle_setting
- freeze_part
- layers, single-replacing use `$j`, multi-replacing use `${multi_layer// /,}`

```bash
bash run_shuffle.sh
```
