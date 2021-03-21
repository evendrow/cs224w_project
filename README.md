# CS224W Final Project: DropEdge and Guided DropEdge on ogbn-proteins

This repository contains scripts for training MLP and GNN (GraphSage and GCN) models with support for DropEdge, Single-iteration DropEdge, and Guided DropEdge as described in our project report. Parameters for training, including dropping probabilities for DropEdge.

## Training & Evaluation

```
# Run with default config
python gnn.py

# Run GraphSage with DropEdge
python gnn.py --use_sage --dropedge=0.2 --everyepoch

# Run GraphSage with Guided DropEdge
python gnn.py --use_sage --dropedge=0.2 --everyepoch --guideddrop
```
