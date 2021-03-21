import argparse

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from logger import Logger

import torch_sparse
from torch_sparse import SparseTensor

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, normalize=False))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, normalize=False))
        self.convs.append(
            GCNConv(hidden_channels, out_channels, normalize=False))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x

def drop_edges_with_mask(train_adj, mask):
    row, col, val = train_adj.coo()
    sparse_size = train_adj.sparse_sizes()
    row = row[mask]
    col = col[mask]
    if val is not None:
        val = val[mask]
    new_adj = torch_sparse.SparseTensor(row=row, col=col, value=val, sparse_sizes=sparse_size)
    return new_adj

def add_edges_with_prob(adj, prob=0.05):
    row, col, _ = adj.coo()
    sparse_size = adj.sparse_sizes()
    
    # new edges
    num_new = int(adj.numel()*prob)
    new_row = torch.randint(0, sparse_size[0], (num_new,)).cuda()
    new_col = torch.randint(0, sparse_size[1], (num_new,)).cuda()
    # make sure we forward and backward edges, since graph is undirected
    all_rows = torch.cat((new_row, new_col))
    all_cols= torch.cat((new_col, new_row))
     
    row = torch.cat((row, all_rows))
    col = torch.cat((col, all_cols))
    new_adj = SparseTensor(row=row, col=col, sparse_sizes=sparse_size)
    return new_adj

def get_guided_drop_mask(edge_preds, drop_prob=0.1, alpha=0.5):
    random_preds = torch.rand(edge_preds.shape)
    guided_preds = alpha*random_preds + (1-alpha)*drop_prob*edge_preds
    mask = guided_preds > drop_prob
    return mask

def train(model, data, train_idx, optimizer, train_adj=None):
    model.train()
    criterion = torch.nn.BCEWithLogitsLoss()

    if train_adj is None:
        train_adj = data.adj_t

    optimizer.zero_grad()
    out = model(data.x, train_adj)[train_idx]
    loss = criterion(out, data.y[train_idx].to(torch.float))
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator):
    model.eval()

    y_pred = model(data.x, data.adj_t)

    train_rocauc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['rocauc']
    valid_rocauc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['rocauc']
    test_rocauc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['rocauc']

    return train_rocauc, valid_rocauc, test_rocauc


def main():
    parser = argparse.ArgumentParser(description='OGBN-Proteins (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--dropedge', type=float, default=0.0)
    parser.add_argument('--addedge', type=float, default=0.0)
    parser.add_argument('--everyepoch', action='store_true')
    parser.add_argument('--guidedrop', action='store_true')
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygNodePropPredDataset(name='ogbn-proteins',
                                     transform=T.ToSparseTensor())
    data = dataset[0]

    # Move edge features to node features.
    data.x = data.adj_t.mean(dim=1)
    data.adj_t.set_value_(None)

    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)

    if args.use_sage:
        model = SAGE(data.num_features, args.hidden_channels, 112,
                     args.num_layers, args.dropout).to(device)
    else:
        model = GCN(data.num_features, args.hidden_channels, 112,
                    args.num_layers, args.dropout).to(device)

        # Pre-compute GCN normalization.
        adj_t = data.adj_t.set_diag()
        deg = adj_t.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
        data.adj_t = adj_t

    data = data.to(device)

    # Guided edge dropping
    if args.guidedrop:
        edge_preds = torch.load('edge_preds.pt')

    # DropEdge
    torch.random.manual_seed(1776)
    dropedge_p = args.dropedge
    mask = torch.rand(data.adj_t.numel()) > dropedge_p 
    if args.guidedrop:
        mask = get_guided_drop_mask(edge_preds, drop_prob=dropedge_p)

    evaluator = Evaluator(name='ogbn-proteins')
    logger = Logger(args.runs, args)

    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            train_adj = data.adj_t
            if args.everyepoch:
                mask = torch.rand(data.adj_t.numel()) > dropedge_p 
                if args.guidedrop:
                    mask = get_guided_drop_mask(edge_preds, drop_prob=dropedge_p)
            if dropedge_p > 0:
                train_adj = drop_edges_with_mask(data.adj_t, mask)
            if args.addedge > 0:
                train_adj = add_edges_with_prob(train_adj, prob=args.addedge)

            loss = train(model, data, train_idx, optimizer, train_adj=train_adj)

            if epoch % args.eval_steps == 0:
                result = test(model, data, split_idx, evaluator)
                logger.add_result(run, result)

                if epoch % args.log_steps == 0:
                    train_rocauc, valid_rocauc, test_rocauc = result
                    print(f'Run: {run + 1:02d}, '
                          f'Epoch: {epoch:02d}, '
                          f'Loss: {loss:.4f}, '
                          f'Train: {100 * train_rocauc:.2f}%, '
                          f'Valid: {100 * valid_rocauc:.2f}% '
                          f'Test: {100 * test_rocauc:.2f}%')

        print('saving model...')
        torch.save(model.state_dict(), 'gnn_state_dict.pt')
        torch.save(model, 'gnn_model.pt')

        logger.print_statistics(run)
    logger.print_statistics()

if __name__ == "__main__":
    main()
