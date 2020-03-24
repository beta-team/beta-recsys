import sys
sys.path.append("../")
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse import vstack
from scipy import sparse
import torch
import torch.nn as nn
from models.gmf import GMF
from models.mlp import MLP
from models.torch_engine import Engine

class NGCF(torch.nn.Module):
    def __init__(self, config):
        super(NeuMF, self).__init__()
        self.config = config
        self.userNum = config["num_users"]
        self.itemNum = config["num_items"]
        self.embedSize = config["emb_dim"]
        self.uEmbd = nn.Embedding(self.userNum,self.embedSize)
        self.iEmbd = nn.Embedding(self.itemNum,self.embedSize)
        self.GNNlayers = torch.nn.ModuleList()
        self.leakyRelu = nn.LeakyReLU()
        self.selfLoop = self.getSparseEye(self.userNum+self.itemNum)
        self.transForm1 = nn.Linear(in_features=layers[-1]*(len(layers))*2,out_features=64)
        self.transForm2 = nn.Linear(in_features=64,out_features=32)
        self.transForm3 = nn.Linear(in_features=32,out_features=1)

        for idx, (in_size, out_size) in enumerate(
            zip(config["layers"][:-1], config["layers"][1:])
        ):
            self.GNNlayers.append(torch.nn.Linear(in_size, out_size))

    def getSparseEye(self,num):
        i = torch.LongTensor([[k for k in range(0,num)],[j for j in range(0,num)]])
        val = torch.FloatTensor([1]*num)
        return torch.sparse.FloatTensor(i,val)

    def getFeatureMat(self):
        uidx = torch.LongTensor([i for i in range(self.userNum)])
        iidx = torch.LongTensor([i for i in range(self.itemNum)])
        if self.useCuda == True:
            uidx = uidx.cuda()
            iidx = iidx.cuda()

        userEmbd = self.uEmbd(uidx)
        itemEmbd = self.iEmbd(iidx)
        features = torch.cat([userEmbd,itemEmbd],dim=0)
        return features

    def forward(self,userIdx,itemIdx):

        itemIdx = itemIdx + self.userNum
        userIdx = list(userIdx.cpu().data)
        itemIdx = list(itemIdx.cpu().data)
        # gcf data propagation
        features = self.getFeatureMat()
        finalEmbd = features.clone()
        for gnn in self.GNNlayers:
            features = gnn(self.LaplacianMat,self.selfLoop,features)
            features = nn.ReLU()(features)
            finalEmbd = torch.cat([finalEmbd,features.clone()],dim=1)

        userEmbd = finalEmbd[userIdx]
        itemEmbd = finalEmbd[itemIdx]
        embd = torch.cat([userEmbd,itemEmbd],dim=1)

        embd = nn.ReLU()(self.transForm1(embd))
        embd = self.transForm2(embd)
        embd = self.transForm3(embd)
        prediction = embd.flatten()

        return prediction


class NGCFEngine(Engine):
    """Engine for training & evaluating GMF model"""

    def __init__(self, config, gmf_config=None, mlp_config=None):
        self.model = NeuMF(config)
        self.gmf_config = gmf_config
        self.mlp_config = mlp_config
        super(NeuMFEngine, self).__init__(config)
        print(self.model)
        if gmf_config != None and mlp_config != None:
            self.load_pretrain_weights()
            
    def buildLaplacianMat(self,rt):
        rt_item = rt['itemId'] + self.userNum
        uiMat = coo_matrix((rt['rating'], (rt['userId'], rt['itemId'])))

        uiMat_upperPart = coo_matrix((rt['rating'], (rt['userId'], rt_item)))
        uiMat = uiMat.transpose()
        uiMat.resize((self.itemNum, self.userNum + self.itemNum))

        A = sparse.vstack([uiMat_upperPart,uiMat])
        selfLoop = sparse.eye(self.userNum+self.itemNum)
        sumArr = (A>0).sum(axis=1)
        diag = list(np.array(sumArr.flatten())[0])
        diag = np.power(diag,-0.5)
        D = sparse.diags(diag)
        L = D * A * D
        L = sparse.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor([row,col])
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i,data)
        return SparseL

    def train_single_batch(self, users, items, ratings):
        assert hasattr(self, "model"), "Please specify the exact model !"
        users, items, ratings = (
            users.to(self.device),
            items.to(self.device),
            ratings.to(self.device),
        )
        self.optimizer.zero_grad()
        ratings_pred = self.model(users, items)
        loss = self.loss(ratings_pred.view(-1), ratings)
        loss.backward()
        self.optimizer.step()
        loss = loss.item()
        return loss

    def train_an_epoch(self, train_loader, epoch_id):
        assert hasattr(self, "model"), "Please specify the exact model !"
        self.model.train()
        total_loss = 0
        for batch_id, batch in enumerate(train_loader):
            assert isinstance(batch[0], torch.LongTensor)
            user, item, rating = batch[0], batch[1], batch[2]
            rating = rating.float()
            loss = self.train_single_batch(user, item, rating)
            total_loss += loss
        print("[Training Epoch {}], Loss {}".format(epoch_id, loss))
        self.writer.add_scalar("model/loss", total_loss, epoch_id)

    def load_pretrain_weights(self):
        """Loading weights from trained MLP model & GMF model"""
        mlp_model = MLP(self.mlp_config)

        self.resume_checkpoint(
            self.config["checkpoint_dir"] + self.config["pretrain_mlp"], mlp_model
        )

        self.model.embedding_user_mlp.weight.data = mlp_model.embedding_user.weight.data
        self.model.embedding_item_mlp.weight.data = mlp_model.embedding_item.weight.data
        #         for idx in range(len(self.fc_layers)):
        #             self.model.fc_layers[idx].weight.data = mlp_model.fc_layers[idx].weight.data

        gmf_model = GMF(self.gmf_config)
        self.resume_checkpoint(
            self.config["checkpoint_dir"] + self.config["pretrain_gmf"], gmf_model
        )
        self.model.embedding_user_mf.weight.data = gmf_model.embedding_user.weight.data
        self.model.embedding_item_mf.weight.data = gmf_model.embedding_item.weight.data

        self.model.affine_output.weight.data = 0.5 * torch.cat(
            [mlp_model.affine_output.weight.data, gmf_model.affine_output.weight.data],
            dim=-1,
        )
        self.model.affine_output.bias.data = 0.5 * (
            mlp_model.affine_output.bias.data + gmf_model.affine_output.bias.data
        )