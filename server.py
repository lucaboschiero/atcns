from __future__ import print_function

from copy import deepcopy
import pandas as pd
from rules.correlations import C
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import torch
import torch.nn.functional as F
import numpy as np

from utils import utils
from utils.backdoor_semantic_utils import SemanticBackdoor_Utils
from utils.backdoor_utils import Backdoor_Utils
import time
from utils.logger import get_logger

# Get the logger
logger = get_logger()

class Server():
    def __init__(self, model, dataLoader, criterion=F.nll_loss, device='cpu'):
        self.clients = []
        self.model = model
        self.dataLoader = dataLoader
        self.device = device
        self.emptyStates = None
        self.init_stateChange()
        self.Delta = None
        self.iter = 0
        self.AR = self.FedAvg
        self.func = torch.mean
        self.isSaveChanges = False
        self.savePath = './AggData'
        self.criterion = criterion
        self.path_to_aggNet = ""

    def init_stateChange(self):
        states = deepcopy(self.model.state_dict())
        for param, values in states.items():
            values *= 0
        self.emptyStates = states

    def attach(self, c):
        self.clients.append(c)

    def distribute(self):
        for c in self.clients:
            c.setModelParameter(self.model.state_dict())

    def test(self):
        #print("[Server] Start testing")
        logger.info("[Server] Start testing")
        self.model.to(self.device)          #Moves the model to the appropriate device (CPU or GPU) for testing.
        self.model.eval()                   #Puts the model in evaluation mode (disables dropout, batch normalization updates, etc.).
        
        #variable initialization
        test_loss = 0
        correct = 0
        count = 0
        c = 0
        f1 = 0
        conf = np.zeros([10,10])

        #Disables gradient calculations to save memory and speed up testing (no backpropagation needed) and iterates over the test dataset using the dataLoader
        with torch.no_grad():
            for data, target in self.dataLoader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)              #Feeds the input data through the global model to get predictions.

                test_loss += self.criterion(output, target, reduction='sum').item()  # sum up batch loss between output and target
                
                if output.dim() == 1:
                    pred = torch.round(torch.sigmoid(output))    #If the output is one-dimensional, apply the sigmoid function and round the result to get a binary prediction (0 or 1).
                else:
                    pred = output.argmax(dim=1, keepdim=True)  # If the output is a logits tensor with multiple dimensions, use argmax to select the class with the highest probability.

                correct += pred.eq(target.view_as(pred)).sum().item()    #number of correct predictions
                count += pred.shape[0]                                   #number of predictions
                conf += confusion_matrix(target.cpu(),pred.cpu(), labels = [i for i in range(10)])    #calculate global confusion matrix
                f1 += f1_score(target.cpu(), pred.cpu(), average = 'weighted')*count          #calculate global f1 score   
                c+=count

        test_loss /= count                             # Computes the average test loss.
        accuracy = 100. * correct / count              # Calculates the accuracy percentage.

        #print results
        logger.info(conf.astype(int))                        # Prints the confusion matrix with integer values.
        logger.info(f1/c)                                    # Prints the final weighted F1-score.
        self.model.cpu()                               # avoid occupying gpu when idle
        logger.info(
            '[Server] Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, count, accuracy))
        return test_loss, accuracy

    def test_backdoor(self):
        logger.info("[Server] Start testing backdoor\n")
        self.model.to(self.device)
        self.model.eval()
        test_loss = 0
        correct = 0
        utils = Backdoor_Utils()
        with torch.no_grad():
            for data, target in self.dataLoader:
                
                #corrupting dataset
                data, target = utils.get_poison_batch(data, target, backdoor_fraction=1,
                                                      backdoor_label=utils.backdoor_label, evaluation=True)
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.dataLoader.dataset)
        accuracy = 100. * correct / len(self.dataLoader.dataset)

        self.model.cpu()  ## avoid occupying gpu when idle
        logger.info(
            '[Server] Test set (Backdoored): Average loss: {:.4f}, Success rate: {}/{} ({:.0f}%)\n'.format(test_loss, correct,
                                                                                                    len(
                                                                                                        self.dataLoader.dataset),
                                                                                                    accuracy))
        return test_loss, accuracy

    def test_semanticBackdoor(self):
        print("[Server] Start testing semantic backdoor")

        self.model.to(self.device)
        self.model.eval()
        test_loss = 0
        correct = 0
        utils = SemanticBackdoor_Utils()
        with torch.no_grad():
            for data, target in self.dataLoader:

                #corrupting dataset
                data, target = utils.get_poison_batch(data, target, backdoor_fraction=1,
                                                      backdoor_label=utils.backdoor_label, evaluation=True)
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.dataLoader.dataset)
        accuracy = 100. * correct / len(self.dataLoader.dataset)

        self.model.cpu()  ## avoid occupying gpu when idle
        print(
            '[Server] Test set (Semantic Backdoored): Average loss: {:.4f}, Success rate: {}/{} ({:.0f}%)\n'.format(test_loss,
                                                                                                             correct,
                                                                                                             len(
                                                                                                                 self.dataLoader.dataset),
                                                                                                             accuracy))
        return test_loss, accuracy, data, pred

    #train the server model by getting all the clients info (ie weight update) and updating the global model, redistributing it until convergence
    def train(self, group, epoch):
        selectedClients = [self.clients[i] for i in group]       # get all clients
        for c in selectedClients:             
            c.train()                   # launch training for each client
            c.update()                  # update clients models

        if self.isSaveChanges:
            self.saveChanges(selectedClients)
        

        attackers = 0
        if epoch < 3:
            Delta = self.FedAvg(selectedClients)
        else:
            tic = time.perf_counter()
            if (self.AR == self.FedAvg or self.AR == self.FedMedian or self.AR == self.geometricMedian or self.AR == self.krum or self.AR == self.mkrum):
                Delta = self.AR(selectedClients)      # back to the server, get clients weight update, aggregated accorting to the aggregation rule
            else: 
                Delta, attackers = self.AR(selectedClients)

            toc = time.perf_counter()
            #print(f"[Server] The aggregation takes {toc - tic:0.6f} seconds.\n")
            logger.info(f"[Server] The aggregation takes {toc - tic:0.6f} seconds.")

        for param in self.model.state_dict():
            self.model.state_dict()[param] += Delta[param]         # update model parameters
        self.iter += 1
        return attackers

    def saveChanges(self, clients):

        Delta = deepcopy(self.emptyStates)
        deltas = [c.getDelta() for c in clients]
        
        n = len([c for c in clients])
        vecs = [utils.net2vec(delta) for delta in deltas]
        vecs = [vec for vec in vecs if torch.isfinite(vec).all().item()]
        input = torch.stack(vecs, 1).unsqueeze(0)
        #a = C(input,n)
        #pd.DataFrame(a).to_csv(str(self.savePath)+"corr"+str(self.iter)+".csv",header = None, index = None)      
        #print("Correlations Saved")
        param_trainable = utils.getTrainableParameters(self.model)

        param_nontrainable = [param for param in Delta.keys() if param not in param_trainable]
        for param in param_nontrainable:
            del Delta[param]
        print(f"[Server] Saving the model weight of the trainable paramters:\n {Delta.keys()}")
        for param in param_trainable:
            ##stacking the weight in the innerest dimension
            param_stack = torch.stack([delta[param] for delta in deltas], -1)
            shaped = param_stack.view(-1, len(clients))
            Delta[param] = shaped

        saveAsPCA = True
        saveOriginal = False
        if saveAsPCA:
            from utils import convert_pca
            proj_vec = convert_pca._convertWithPCA(Delta)
            savepath = f'{self.savePath}/pca_{self.iter}.pt'
            torch.save(proj_vec, savepath)
            print(f'[Server] The PCA projections of the update vectors have been saved to {savepath} (with shape {proj_vec.shape})')
#             return
        if saveOriginal:
            savepath = f'{self.savePath}/{self.iter}.pt'

            torch.save(Delta, savepath)
            print(f'[Server] Update vectors have been saved to {savepath}')
        #correlations = pd.read_csv("/content/drive/MyDrive/FL/corr.csv",header = None, sep = ",") 
        #pd.DataFrame(correlations).to_csv(str(savepath)+"corr.csv", header = None, sep = ',', index = "None")

    ## Aggregation functions ##

    def set_AR(self, ar):
        if ar == 'fedavg':
            self.AR = self.FedAvg
        elif ar == 'median':
            self.AR = self.FedMedian
        elif ar == 'gm':
            self.AR = self.geometricMedian
        elif ar == 'krum':
            self.AR = self.krum
        elif ar == 'mkrum':
            self.AR = self.mkrum
        elif ar == 'foolsgold':
            self.AR = self.foolsGold
        elif ar == 'contra' :
            self.AR = self.contra
        elif ar == 'residualbase':
            self.AR = self.residualBase
        elif ar == 'attention':
            self.AR = self.net_attention
        elif ar == 'mlp':
            self.AR = self.net_mlp
        elif ar == 'mst' :
            self.AR = self.mst
        elif ar == 'density' :
            self.AR = self.k_densest
        elif ar == 'cc' :
            self.AR = self.central
        elif ar == 'threshold' :
            self.AR = self.thresholding
        elif ar == 'pca' :
            self.AR = self.pca    
        elif ar == 'kmeans' :
            self.AR = self.k_means   
        else:
            raise ValueError("Not a valid aggregation rule or aggregation rule not implemented")

    def FedAvg(self, clients):
        out = self.FedFuncWholeNet(clients, lambda arr: torch.mean(arr, dim=-1, keepdim=True))  # arr is the imput to this lambda function, that computes the mean of the tensor arr
        return out

    def FedMedian(self, clients):
        out = self.FedFuncWholeNet(clients, lambda arr: torch.median(arr, dim=-1, keepdim=True)[0])
        return out

    def geometricMedian(self, clients):
        from rules.geometricMedian import Net
        self.Net = Net
        out = self.FedFuncWholeNet(clients, lambda arr: Net().cpu()(arr.cpu()))
        return out

    def krum(self, clients):
        from rules.multiKrum import Net
        self.Net = Net
        out = self.FedFuncWholeNet(clients, lambda arr: Net('krum').cpu()(arr.cpu()))
        return out

    def mkrum(self, clients):
        from rules.multiKrum import Net
        self.Net = Net
        out = self.FedFuncWholeNet(clients, lambda arr: Net('mkrum').cpu()(arr.cpu()))
        return out

    def foolsGold(self, clients):
        from rules.foolsGold import Net
        self.Net = Net
        out,attackers = self.FedFuncWholeNetAttackers(clients, lambda arr: Net().cpu()(arr.cpu()))
        return out,attackers
        
    def contra(self, clients):
        from rules.contra import Net
        self.Net = Net
        out,attackers = self.FedFuncWholeNetAttackers(clients, lambda arr: Net().cpu()(arr.cpu()))
        return out,attackers    

    def residualBase(self, clients):
        from rules.residualBase import Net
        out = self.FedFuncWholeStateDict(clients, Net().main)
        return out

    def net_attention(self, clients):
        from aaa.attention import Net

        net = Net()
        net.path_to_net = self.path_to_aggNet

        out = self.FedFuncWholeStateDict(clients, lambda arr: net.main(arr, self.model))
        return out

    def net_mlp(self, clients):
        from aaa.mlp import Net

        net = Net()
        net.path_to_net = self.path_to_aggNet

        out = self.FedFuncWholeStateDict(clients, lambda arr: net.main(arr, self.model))
        return out

        ## Helper functions, act as adaptor from aggregation function to the federated learning system##
        
    def mst(self, clients) :
        from rules.mst import Net         #import net from rules/mst.py
        self.Net = Net                    # non ho capito perchè questo (cioè perchè assegna net a self.Net?)
        out,attackers = self.FedFuncWholeNetAttackers(clients, lambda arr: Net().cpu()(arr.cpu()))
        return out,attackers
    
    def k_densest(self, clients) :
        from rules.density import Net
        self.Net = Net
        out,attackers = self.FedFuncWholeNetAttackers(clients, lambda arr: Net().cpu()(arr.cpu()))
        return out,attackers
    
    def k_means(self,clients):
        from rules.kmeans import Net
        self.Net = Net
        out,attackers = self.FedFuncWholeNetAttackers(clients, lambda arr: Net().cpu()(arr.cpu()))
        return out, attackers

    def FedFuncWholeNet(self, clients, func):
        '''
        Returns the weight update of the server model by applying the aggregation rule to the weight updates of the clients.
        The aggregation rule views the update vectors as stacked vectors (1 by d by n).
        '''
        Delta = deepcopy(self.emptyStates)           # Delta is an empty copy of the whole net structure, used to contain the updated weights after aggregation
        deltas = [c.getDelta() for c in clients]     # gets the weight updates of each client
        vecs = [utils.net2vec(delta) for delta in deltas]              # transforms all deltas into vectors of dimension 1
        vecs = [vec for vec in vecs if torch.isfinite(vec).all().item()]      # filters values to keep only finite ones

        # Print the L2 norm for each client's update vector
        for i, vec in enumerate(vecs):
            update_norm = torch.norm(vec, p=2).item()  # Compute L2 norm
            logger.info(f"Client {i} Update Norm: {update_norm}")  # Print the norm

        result = func(torch.stack(vecs, 1).unsqueeze(0))  # input as 1 by d by n, apply the aggregation function
        result = result.view(-1)
        utils.vec2net(result, Delta)     # converts back the resulting vectors to the network structure
        return Delta                     # quando viene aggiornato Delta???
    
    def FedFuncWholeNetAttackers(self, clients, func):
        '''
        The aggregation rule views the update vectors as stacked vectors (1 by d by n).
        '''
        Delta = deepcopy(self.emptyStates)
        deltas = [c.getDelta() for c in clients]
        vecs = [utils.net2vec(delta) for delta in deltas]
        vecs = [vec for vec in vecs if torch.isfinite(vec).all().item()]

        # Print the L2 norm for each client's update vector
        for i, vec in enumerate(vecs):
            update_norm = torch.norm(vec, p=2).item()  # Compute L2 norm
            logger.info(f"Client {i} Update Norm: {update_norm}")  # Print the norm

        result, attackers = func(torch.stack(vecs, 1).unsqueeze(0))  # input as 1 by d by n
        result = result.view(-1)
        utils.vec2net(result, Delta)
        return Delta, attackers

    def FedFuncWholeStateDict(self, clients, func):
        '''
        Returns the weight update of the server model by applying the aggregation rule to the weight updates of the clients, for more complex scenario.
        The aggregation rule views the update vectors as a set of state dict.
        '''
        Delta = deepcopy(self.emptyStates)
        deltas = [c.getDelta() for c in clients]
        # sanity check, remove update vectors with nan/inf values
        deltas = [delta for delta in deltas if torch.isfinite(utils.net2vec(delta)).all().item()]   # removes all infinite/null deltas

        resultDelta = func(deltas)       # apply the aggregation rule (used in )

        Delta.update(resultDelta)
        return Delta
