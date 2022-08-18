## stateful-premise-selection-with-transformers


Sort of reverse engineering the code of the transformer/rnn paper to generate some problems and analyse the output/performance.
Unclear how this code actually operaties. Data seems to have been premade by another script somewhere?



runnable scripts:
- predictions_to_dependencies.py
- jaccard_and_coverage.py
- also predict.py  but it fails on importing local modules
- run_prover.py
- train.py



# Basics


Run `train.py` to train a model on the parameters given in config. The evaluation loop is super slow,
so best to not use it during training. The model is saved to saved_model_path and the losses are logged
in results.pkl


