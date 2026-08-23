"""Expert Network — machines pooling SSDs and RAM to run MoE models none of
them could run alone. Nodes serve expert partitions; a driver runs attention
locally and gathers active experts over the network."""
