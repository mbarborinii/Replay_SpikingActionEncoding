# Replay_SpikingActionEncoding
SpikingNeuralNetwork for regression and autoregression of repeated dynamics. Plug and play with already-trained weights and parameters

Use:
- choose experiment to run in the Experiments folder (only one available now)
- run with Run_experiment (ipynb file or normal python script)
- loss, outputs and targets are saved in Results>...>models>test_results.npz, while the rasterplots are saved in results_raster.npz. Note that the networs uses 8 smaller runtime steps in between iterations, therefore len(raster) in time = 8 len(output) in time