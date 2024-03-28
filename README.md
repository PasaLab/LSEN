# Mining Long Short-Term Evolutionary Patterns for Temporal Knowledge Graph Reasoning

This is the code associated with the submission "Mining Long Short-Term Evolutionary Patterns
for Temporal Knowledge Graph Reasoning" under review at ICPR 2024. 

## Datasets

All the processed datasets we used in the paper can be downloaded at [Baidu Yun](https://pan.baidu.com/s/1Yx3n1tUvQeviKY1OttYP8Q?pwd=6cha)(password:6cha). Put datasets in the folder 'data' to run experimments.

## Run scripts

To run LSEN on YAGO for link prediction task:

```{bash}
python main.py -d YAGO --history-len 1 --lambdax 1.0 --graph-layer 6
```

To run LSEN on WIKI for link prediction task:

```{bash}
python main.py -d WIKI --history-len 1 --lambdax 1.0 --graph-layer 6
```

To run LSEN on ICEWS18 for link prediction task:

```{bash}
python main.py -d ICEWS18 --history-len 3 --lambdax 2.0 --graph-layer 2
```

To run LSEN on ICEWS14 for link prediction task:

```{bash}
python main.py -d ICEWS14 --history-len 3 --lambdax 2.0 --graph-layer 2 --use-valid false --max-epochs 15
```

To run LSEN on GDELT for link prediction task:

```bash
python main.py -d GDELT --history-len 3 --lambdax 2.0 --graph-layer 2
```

## Optional arguments

```bash
--dataset			the dataset to use (YAGO, WIKI, ICEWS18, ICEWS14, or GDELT)
--device			the device to use
--batch-size 		batch size
--max-epochs 		maximum epochs
--valid-epochs		validation epochs
--alpha 			alpha for nceloss
--lambdax 			the hyperparameter lambda
--history-len		the time window size
--mode				offline or online setting
--graph-layer		number of GNN layers

--embedding-dim		embedding dimension of entities and relations
--lr 				learning rate
--weight-decay 		weight decay ratio
--dropout 			dropout rate
--grad-norm 		norm to clip gradient to
--filtering 		filtering setup
--only-eva   		whether only evaluation on test set
--use-valid 		whether using validation set
--model-dir 		model directory
--save-dir 			save directory
--eva-dir 			saved dir of the testing model
```

