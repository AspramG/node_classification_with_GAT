import argparse
from os import path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from dgl import batch
from dgl.data.ppi import LegacyPPIDataset
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch import GATConv
from sklearn.metrics import f1_score
from torch import nn, optim
from torch.utils.data import DataLoader


MODEL_STATE_FILE = path.join(path.dirname(path.abspath(__file__)), "model_state.pth")


class BasicGraphModel(nn.Module):

    def __init__(self, g, num_layers,input_dimension,hidden_l_dim,heads,feat_drop,att_drop,negative_slope,residual,activation,num_classes):
        super().__init__()

        self.g = g
        self.activation=activation
        self.input_dimension=input_dimension
        self.num_layers=num_layers
        self.hidden_l_dim=hidden_l_dim
        self.num_classes=num_classes
        self.heads=heads
        self.feat_drop=feat_drop
        self.att_drop=att_drop
        self.negative_slope=negative_slope
        self.residual=residual
        self.layers=nn.ModuleList()
        
        #input layer
       # self.layers.append(GATConv(input_dimension,hidden_l_dim[0]*heads[0] + input_dimension,heads[0],feat_drop,att_drop,negative_slope,False,self.activation))
        #hidden
        for layer in range(num_layers-1):
            
            if(layer==0):
                l_inp=input_dimension
            else:
                l_inp=l_out 
            l_out=hidden_l_dim[layer]*heads[layer]+l_inp
            self.layers.append(GATConv(l_inp,hidden_l_dim[layer],heads[layer],feat_drop,att_drop,negative_slope,residual,self.activation))
        #output layer
        self.layers.append(GATConv(l_out,num_classes,heads[-1],feat_drop,att_drop,negative_slope,residual,None)) #last element -1 is the last
        
        

    def forward(self, inputs):
        
        for q in range(self.num_layers-1):
            outputs = self.layers[q](self.g, inputs).flatten(1) #since GATconv returns a tensor in wich the first dimension is the corresponding value for each head (n_head,output size), i need to flatten it out (and remembering to adjust the output shape)
            inputs = torch.cat([inputs, outputs], dim=1) #since output dimension is hidden_dim*heads+input, i have to take in account the increase in dimensionof outputs , concatenating the inputs 
            
        last = self.layers[-1](self.g, inputs).mean(1)

        return last

def main(args):

    # load dataset and create dataloader
    train_dataset, test_dataset = LegacyPPIDataset(mode="train"), LegacyPPIDataset(mode="test")
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    n_features, n_classes = train_dataset.features.shape[1], train_dataset.labels.shape[1]

    # create the model, loss function and optimizer
    device = torch.device("cpu" if args.gpu < 0 else "cuda:" + str(args.gpu))

    ########### Replace this model with your own GNN implemented class ################################

    model = BasicGraphModel(g=train_dataset.graph, num_layers=4, input_dimension=n_features,
                            hidden_l_dim=[240,260,120],heads=[4,3,4,6],feat_drop=0,att_drop=0,negative_slope=0.2,residual=True,activation=F.elu,num_classes=n_classes).to(device)
   
    
    ###################################################################################################

    loss_fcn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # train
    if args.mode == "train":
        train(model, loss_fcn, device, optimizer, train_dataloader, test_dataset)
        torch.save(model.state_dict(), MODEL_STATE_FILE)

    # import model from file
    model.load_state_dict(torch.load(MODEL_STATE_FILE))

    # test the model
    test(model, loss_fcn, device, test_dataloader)

    return model

def train(model, loss_fcn, device, optimizer, train_dataloader, test_dataset):

    f1_score_list = []
    epoch_list = []

    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch, data in enumerate(train_dataloader):
            subgraph, features, labels = data
            subgraph = subgraph.to(device)
            features = features.to(device)
            labels = labels.to(device)
            model.g = subgraph
            for layer in model.layers:
                layer.g = subgraph
            logits = model(features.float())
            loss = loss_fcn(logits, labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        loss_data = np.array(losses).mean()
        print("Epoch {:05d} | Loss: {:.4f}".format(epoch + 1, loss_data))

        if epoch % 5 == 0:
            scores = []
            for batch, test_data in enumerate(test_dataset):
                subgraph, features, labels = test_data
                subgraph = subgraph.clone().to(device)
                features = features.clone().detach().to(device)
                labels = labels.clone().detach().to(device)
                score, _ = evaluate(features.float(), model, subgraph, labels.float(), loss_fcn)
                scores.append(score)
                f1_score_list.append(score)
                epoch_list.append(epoch)
            print("F1-Score: {:.4f} ".format(np.array(scores).mean()))

    #plot_f1_score(epoch_list, f1_score_list)

    
def test(model, loss_fcn, device, test_dataloader):
   
    test_scores = []
    for batch, test_data in enumerate(test_dataloader):
        subgraph, features, labels = test_data
        subgraph = subgraph.to(device)
        features = features.to(device)
        labels = labels.to(device)
        test_scores.append(evaluate(features, model, subgraph, labels.float(), loss_fcn)[0])
    mean_scores = np.array(test_scores).mean()
    print("F1-Score: {:.4f}".format(np.array(test_scores).mean()))
    
    return mean_scores

def evaluate(features, model, subgraph, labels, loss_fcn):
    with torch.no_grad():
        model.eval()
        model.g = subgraph
        for layer in model.layers:
            layer.g = subgraph
        output = model(features.float())
        loss_data = loss_fcn(output, labels.float())
        predict = np.where(output.data.cpu().numpy() >= 0.5, 1, 0)
        score = f1_score(labels.data.cpu().numpy(), predict, average="micro")
        return score, loss_data.item()

def collate_fn(sample) :
    # concatenate graph, features and labels w.r.t batch size
    graphs, features, labels = map(list, zip(*sample))
    graph = batch(graphs)
    features = torch.from_numpy(np.concatenate(features))
    labels = torch.from_numpy(np.concatenate(labels))
    return graph, features, labels

def plot_f1_score(epoch_list, f1_score_list) :
    plt.plot(epoch_list, f1_score_list)
    plt.title("Evolution of f1 score w.r.t epochs")
    fig = plt.gcf()
    fig.savefig('fig1.pdf')
    
    plt.show()

if __name__ == "__main__":

    # PARSER TO ADD OPTIONS
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",  choices=["train", "test"], default="train")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU to use. Set -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    # READ MAIN
    main(args)
