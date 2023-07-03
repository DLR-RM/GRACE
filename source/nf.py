import normflows as nf
import larsflow as lf
import pytorch_lightning as pl
import torch
import torch_geometric.nn as pyg_nn
import numpy as np
import sklearn.metrics as sk_metrics

class NFModel(pl.LightningModule):
    def __init__(self, gnn, hparams=None):
        super().__init__()
        self.save_hyperparameters(dict(hparams))

        self.gnn = gnn
        self.dim = self.hparams.input_dim

        if self.hparams.nf_base_dist == "gaussian":
            # Define 2D Gaussian base distribution
            base = nf.distributions.base.DiagGaussian(self.dim, trainable=True)
        else:
            a = nf.nets.MLP([self.dim, self.dim // 2, self.dim // 2, 1], output_fn="sigmoid")
            base = lf.distributions.ResampledGaussian(self.dim, a, 100, 0.1, trainable=False)

        # Define list of flows
        flows = []
        for _ in range(self.hparams.flow_layers):
            # Last layer is initialized by zeros making training more stable
            param_map = nf.nets.MLP([self.dim // 2, self.hparams.nf_layer_size, self.hparams.nf_layer_size, self.dim], 
                                    init_zeros=True)
            # Add flow layer
            flows.append(nf.flows.AffineCouplingBlock(param_map, scale_map=self.hparams.nf_scale_map))

            # Swap dimensions
            flows.append(nf.flows.Permute(self.dim, mode='swap'))

            if self.hparams.nf_act_norm:
                flows.append(nf.flows.ActNorm(self.dim))
            
        # Construct flow model
        self.nf = nf.NormalizingFlow(base, flows)
    
    def configure_optimizers(self):    
        optimizer = torch.optim.Adam(self.nf.parameters(), lr=self.hparams.learning_rate)
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        embedding = self._embed(train_batch)
        nll = -1 * self.nf.log_prob(embedding)

        if self.hparams.nf_supervised_loss:
            y = self._get_feasibility(train_batch)
            y = torch.Tensor([-1 if x == 0 else 1 for x in y])

            # min feas. loss, max infeas. loss
            nll = y * nll

            # remove infeasible samples which are already far away
            feas_nll = nll[y == 1]
            infeas_nll = nll[y == -1]
            infeas_nll = infeas_nll[infeas_nll < self.hparams.nf_c_threshold]
            nll = torch.cat((feas_nll, infeas_nll), 0)
            
        bpd = self._calc_bpd(nll)

        mean_nll = torch.mean(nll)
        mean_bpd = torch.mean(bpd)

        self.log('nf_train_nll', mean_nll, batch_size=self.hparams.batch_size)
        self.log('nf_train_bpd', mean_bpd, batch_size=self.hparams.batch_size)

        return mean_bpd

    def validation_step(self, val_batch, batch_idx):
        nll, bpd, log_prob = self._get_loss(val_batch, return_prob=True)

        self.log('nf_val_nll', nll, batch_size=self.hparams.batch_size)
        self.log('nf_val_bpd', bpd, batch_size=self.hparams.batch_size)

        y = self._get_feasibility(val_batch)
        y_hat = log_prob.cpu().tolist()

        return y, y_hat

    def test_step(self, batch, batch_idx):
        nll, bpd = self._get_loss(batch)
        
        self.log('test_nll', nll, batch_size=self.hparams.batch_size)
        self.log('test_bpd', bpd, batch_size=self.hparams.batch_size)

    def validation_epoch_end(self, outputs):
        y_true, y_score = list(), list()
        for (y, y_hat) in outputs:
            y_true.append(y)
            y_score.append(y_hat)
        
        y_true = np.concatenate(y_true)
        y_score = np.concatenate(y_score)

        if self.hparams.nf_supervised_loss:
            self._log_prob_prints(y_true, y_score)
        
        y_score = np.nan_to_num(y_score)
        fpr, tpr, _ = sk_metrics.roc_curve(y_true, y_score)
        auc_score = sk_metrics.auc(fpr, tpr)

        print("AUC: ", auc_score)

        self.log("nf_val_auc", auc_score)

    def _get_loss(self, batch, return_prob=False):
        embedding = self._embed(batch)
        log_prob = self.nf.log_prob(embedding)

        nll = -1 * torch.mean(log_prob)
        bpd = self._calc_bpd(nll)

        if return_prob:         
            return nll, bpd, log_prob
        else:
            return nll, bpd

    def _calc_bpd(self, nll):
        return (nll * np.log2(np.exp(1)) / np.prod(self.dim))
    
    def _embed(self, batch):
        self.gnn.freeze()
        part_embeddings = self.gnn(batch)
        graph_embedding = pyg_nn.global_mean_pool(part_embeddings, batch=batch['part'].batch)
        
        return graph_embedding
    
    def _get_feasibility(self, batch):
        y = list()
        for i in range(len(batch['part'].y)):
            seqs = batch['part'].y[i]
            if len(seqs) == 1 and seqs[0][0] == 'dummy':
                y.append(0)
            else:
                y.append(1)
        
        return y

    def _log_prob_prints(self, y_true, y_score):
        y_true = torch.tensor(y_true, dtype=torch.bool)
        y_score = -1 * torch.tensor(y_score)

        feas_prob = y_score[y_true == True].mean().item()
        infeas_prob = y_score[y_true == False].mean().item()

        print("Nll feas/infeas:", feas_prob, infeas_prob)

    # Inference
    def forward(self, batch):
        embedding = self._embed(batch)
        prob = self.nf.log_prob(embedding)

        return prob

    # Ablation studies
    def bpd(self, batch):
        embedding = self._embed(batch)
        nll = -1 * self.nf.log_prob(embedding)

        return (nll * np.log2(np.exp(1)) / np.prod(self.dim))


    def ll_terms(self, batch):
        x = self._embed(batch)
        flows = self.nf.flows

        log_det_sum = torch.zeros(len(x), dtype=x.dtype, device=x.device)
        z = x
        for i in range(len(flows) - 1, -1, -1):
            z, log_det = flows[i].inverse(z)
            log_det_sum += log_det        
        base_prob = self.nf.q0.log_prob(z)

        return base_prob, log_det_sum

    def target_embedding(self, batch):
        graph_embedding = self._embed(batch)

        z = graph_embedding
        flows = self.nf.flows
        
        for i in range(len(flows) - 1, -1, -1):
            z, _ = flows[i].inverse(z)

        return z
