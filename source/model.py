import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn

from source.utils import *
from source.transform import EdgeNormalization


class AssemblyGNN(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(dict(hparams))

    def training_step(self, train_batch, batch_idx):
        _, loss = self._get_pred_loss(train_batch, mode="train")

        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        logit, loss = self._get_pred_loss(val_batch, mode="val")
        acc, _, _ = self._get_accuracy(logit, val_batch)

        self.log('val_loss', loss, batch_size=self.hparams.batch_size)
        self.log('val_acc', acc, batch_size=self.hparams.batch_size)

        return loss

    def test_step(self, test_batch, batch_idx):
        logit, _ = self._get_pred_loss(test_batch, mode="test")
        acc, y_hat, y = self._get_accuracy(logit, test_batch)
        recall, precision, fpr, miss = confusion(y_hat, y, test_batch)

        self.log('test_acc', acc, batch_size=self.hparams.batch_size)
        self.log('recall', recall, batch_size=self.hparams.batch_size)
        self.log('precision', precision, batch_size=self.hparams.batch_size)
        self.log('fpr', fpr, batch_size=self.hparams.batch_size)
        self.log('miss', miss, batch_size=self.hparams.batch_size)

    def configure_optimizers(self):
        if self.hparams.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        elif self.hparams.optimizer == "RMSprop":
            optimizer = torch.optim.RMSprop(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer


class AssemblyNodeLabelGNN(AssemblyGNN):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.save_hyperparameters(dict(hparams))

        self.criterion = GraphBCELoss() if self.hparams.loss_function == "BCE" else GraphCELoss()
        
        self_edge_fill_value = torch.tensor([0, 1], dtype=torch.float) if self.hparams.edge_dim == 2 else 0
        self.surface_gnn = nn.ModuleList([
            pyg_nn.GATv2Conv(in_channels=(self.hparams.surface_input_dim, self.hparams.surface_input_dim),
                      out_channels=self.hparams.hidden_channels, add_self_loops=True, 
                      fill_value=self_edge_fill_value, 
                      edge_dim=self.hparams.edge_dim,
                      negative_slope=self.hparams.relu_slope,
                      dropout=self.hparams.dropout) 
        if self.hparams.architecture == "GAT" else \
            pyg_nn.GraphConv(in_channels=self.hparams.surface_input_dim,
                      out_channels=self.hparams.hidden_channels)
        ])

        for _ in range(self.hparams.surface_gnn_layers - 1):
            self.surface_gnn.append(pyg_nn.GATv2Conv(in_channels=(self.hparams.hidden_channels, self.hparams.hidden_channels),
                                              out_channels=self.hparams.hidden_channels, 
                                              add_self_loops=True, 
                                              fill_value=self_edge_fill_value,
                                              edge_dim=self.hparams.edge_dim,
                                              negative_slope=self.hparams.relu_slope,
                                              dropout=self.hparams.dropout) 
                        if self.hparams.architecture == "GAT" else \
                                    pyg_nn.GraphConv(in_channels=self.hparams.hidden_channels,
                                            out_channels=self.hparams.hidden_channels)
            )

        self.surface_norm= nn.ModuleList(
            [pyg_nn.InstanceNorm(in_channels=self.hparams.hidden_channels, affine=self.hparams.instance_norm_affine, track_running_stats=self.hparams.instance_norm_running_stats) 
                                    for _ in range(self.hparams.surface_gnn_layers)]
        )

        self.part_gnn = pyg_nn.HeteroConv({
                ('surface', 'to', 'part'): pyg_nn.GATv2Conv(in_channels=(self.hparams.hidden_channels, self.hparams.part_input_dim),
                                                     out_channels=self.hparams.hidden_channels, add_self_loops=False,
                                                     edge_dim=None, negative_slope=self.hparams.relu_slope) 
                                        if self.hparams.architecture == "GAT" else \
                                           pyg_nn.GraphConv(in_channels=(self.hparams.hidden_channels, self.hparams.part_input_dim),
                                                    out_channels=self.hparams.hidden_channels)
                                        
        }, aggr='mean')

        self.part_norm = pyg_nn.InstanceNorm(in_channels=self.hparams.hidden_channels, affine=self.hparams.instance_norm_affine,  track_running_stats=self.hparams.instance_norm_running_stats) 

        self.fc = nn.Sequential()
        fc_out_channels = self.hparams.hidden_channels
        for i in range(self.hparams.fc_layers - 1):
            self.fc.add_module("fc_%d" % i, pyg_nn.Linear(fc_out_channels, self.hparams.fc_layer_size))
            self.fc.add_module("tanh_%d" % i, nn.Tanh())

            fc_out_channels //= 2
        self.fc.add_module("fc_last", pyg_nn.Linear(self.hparams.hidden_channels, 1))

    def forward(self, batch):
        x_dict, edge_index_dict, edge_weight_dict = batch.x_dict, batch.edge_index_dict, batch.edge_weight_dict
        surface_batch = batch['surface'].batch
        part_batch = batch['part'].batch
        
        for gnn, norm in zip(self.surface_gnn, self.surface_norm):
            x_dict['surface'] = gnn(x_dict['surface'], edge_index_dict[('surface', 'dist', 'surface')], edge_weight_dict[('surface', 'dist', 'surface')])
            x_dict['surface'] = norm(x_dict['surface'], surface_batch)
            x_dict['surface'] = x_dict['surface'].tanh()

        x_dict = self.part_gnn(x_dict, edge_index_dict, edge_weight_dict)
        x_dict['part'] = self.part_norm(x_dict['part'], part_batch)
        x_dict['part'] = x_dict['part'].tanh()

        return x_dict['part']

    def _get_pred_loss(self, batch, mode):
        #logit = self.forward(batch)
        part_embedding = self.forward(batch)
        logit = self.fc(part_embedding)

        bce_loss = self.criterion(src=logit, index=batch['part'].batch, target=batch['part'].y)
        assembled_loss = assembled_regularization(logit, batch)

        d = self.hparams.reg_delta
        loss = (1 - d) * bce_loss + d * assembled_loss

        return logit, loss

    def _get_accuracy(self, logit, batch):
        probs = logit.sigmoid()
        y_hat = (probs.detach() > self.hparams.threshold).float()
        acc = (y_hat == batch['part'].y).float().sum() / batch['part'].y.shape[0]

        return acc, y_hat, batch['part'].y


class AssemblyNodeLabelWithDistanceGNN(AssemblyNodeLabelGNN):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.aux_criterion = GraphMSELoss()

        # defined here since we need this only when aux loss computed
        self.target_edge_normalization = EdgeNormalization(edge=('part', 'dist', 'part'), min=0, max=679)

        self.part_dist_fc = pyg_nn.Linear(self.hparams.hidden_channels, self.hparams.part_dist_fc_channels)

    def forward(self, batch):
        x_dict, edge_index_dict, edge_weight_dict = batch.x_dict, batch.edge_index_dict, batch.edge_weight_dict
        surface_batch = batch['surface'].batch
        part_batch = batch['part'].batch
        
        for gnn, norm in zip(self.surface_gnn, self.surface_norm):
            x_dict['surface'] = gnn(x_dict['surface'], edge_index_dict[('surface', 'dist', 'surface')], edge_weight_dict[('surface', 'dist', 'surface')])
            x_dict['surface'] = norm(x_dict['surface'], surface_batch)
            x_dict['surface'] = x_dict['surface'].tanh()
            

        x_dict = self.part_gnn(x_dict, edge_index_dict, edge_weight_dict)
        x_dict['part'] = self.part_norm(x_dict['part'], part_batch)
        x_dict['part'] = x_dict['part'].tanh()

        return x_dict['part']

    def _aux_loss(self, batch, part_position_embedding):
        src, dst = batch.edge_index_dict[('part', 'dist', 'part')]

        predicted_distance = torch.norm(part_position_embedding[src] - part_position_embedding[dst], p=2, dim=-1)

        batch = self.target_edge_normalization(batch)
        target_distance = batch.edge_weight_dict[('part', 'dist', 'part')]
        loss = self.aux_criterion(src=predicted_distance, index=batch['part'].batch, target=target_distance)

        return loss
    
    def _get_pred_loss(self, batch, mode):
        part_embedding = self.forward(batch)

        part_logit = self.fc(part_embedding)
        part_position = self.part_dist_fc(part_embedding)

        bce_loss = self.criterion(src=part_logit, index=batch['part'].batch, target=batch['part'].y)
        assembled_loss = assembled_regularization(part_logit, batch)

        aux_loss = self._aux_loss(batch, part_position)
        
        if mode in ["val", "test"]:
            self.log('%s_bce_loss' % mode, bce_loss, batch_size=self.hparams.batch_size)
            self.log('%s_assembled_loss' % mode, assembled_loss, batch_size=self.hparams.batch_size)
            self.log('%s_aux_loss' % mode, aux_loss, batch_size=self.hparams.batch_size)

        d = self.hparams.reg_delta
        loss = (1 - d) * bce_loss + d * assembled_loss + self.hparams.aux_loss_delta * aux_loss

        return part_logit, loss
