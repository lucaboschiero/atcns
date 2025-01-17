from __future__ import print_function

import torch
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter

from clients_attackers import *
from server import Server

from utils.logger import get_logger

# Get the logger
logger = get_logger()

def main(args):
    print('#####################')
    print('#####################')
    print('#####################')
    print(f'Aggregation Rule:\t{args.AR}\nData distribution:\t{args.loader_type}\nAttacks:\t{args.attacks} ')
    print('#####################')
    print('#####################')
    print('#####################')

    torch.manual_seed(args.seed)

    device = args.device           #define the device (cpu or cuda) to be used

    attacks = args.attacks         #define the attack name(?)

    writer = SummaryWriter(f'./logs/{args.experiment_name}')       
    

    #depending on the dataset, gets a list of training data loaders, one for each client, and 1 test data loader for the server
    if args.dataset == 'mnist':
        from tasks import mnist
        trainData = mnist.train_dataloader(args.num_clients, loader_type=args.loader_type, path=args.loader_path,
                                           store=False)
        testData = mnist.test_dataloader(args.test_batch_size)
        Net = mnist.Net
        criterion = F.cross_entropy
    elif args.dataset == 'fmnist':
        from tasks import fmnist
        trainData = fmnist.train_dataloader(args.num_clients, loader_type=args.loader_type, path=args.loader_path,
                                           store=False)
        testData = fmnist.test_dataloader(args.test_batch_size)
        Net = fmnist.Net   
        criterion = F.cross_entropy
    elif args.dataset == 'cifar':
        from tasks import cifar
        trainData = cifar.train_dataloader(args.num_clients, loader_type=args.loader_type, path=args.loader_path,
                                           store=False)
        testData = cifar.test_dataloader(args.test_batch_size)
        Net = cifar.Net
        criterion = F.cross_entropy
    elif args.dataset == 'cifar100':
        from tasks import cifar100
        trainData = cifar100.train_dataloader(args.num_clients, loader_type=args.loader_type, path=args.loader_path,
                                              store=False)
        testData = cifar100.test_dataloader(args.test_batch_size)
        Net = cifar100.Net
        criterion = F.cross_entropy
    elif args.dataset == 'imdb':
        from tasks import imdb
        trainData = imdb.train_dataloader(args.num_clients, loader_type=args.loader_type, path=args.loader_path,
                                          store=False)
        testData = imdb.test_dataloader(args.test_batch_size)
        Net = imdb.Net
        criterion = F.cross_entropy

    # create server instance
    model0 = Net()     #create a mnist/cifar/cifar100/fminst object according to the dataset
    server = Server(model0, testData, criterion, device)          #create a server instance, passing the model, testData...
    server.set_AR(args.AR)             #set the aggregation rule (The aggregation rule determines how the updates from different clients are combined.)
    server.path_to_aggNet = args.path_to_aggNet

    if args.save_model_weights:
        server.isSaveChanges = True
        server.savePath = f'./AggData/{args.loader_type}/{args.dataset}/{args.attacks}/{args.AR}'
        from pathlib import Path
        Path(server.savePath).mkdir(parents=True, exist_ok=True)

        '''
        The following code marks specific clients as malicious by setting their label to 0.
        honest clients are labeled as 1, malicious clients are labeled as 0
        '''
        label = torch.ones(args.num_clients)
        for i in args.attacker_list_labelFlipping:
            label[i] = 0
        for i in args.attacker_list_labelFlippingDirectional:
            label[i] = 0
        for i in args.attacker_list_omniscient:
            label[i] = 0
        for i in args.attacker_list_backdoor:
            label[i] = 0
        for i in args.attacker_list_semanticBackdoor:
            label[i] = 0
        for i in args.attacker_list_multilabelFlipping:
            label[i] = 0

        torch.save(label, f'{server.savePath}/label.pt')        #Saves the label tensor (which marks honest and malicious clients) to the file

    # create clients instance
    attacker_list_labelFlipping = args.attacker_list_labelFlipping
    attacker_list_omniscient = args.attacker_list_omniscient
    attacker_list_backdoor = args.attacker_list_backdoor
    attacker_list_labelFlippingDirectional = args.attacker_list_labelFlippingDirectional
    attacker_list_semanticBackdoor = args.attacker_list_semanticBackdoor
    attacker_list_multilabelFlipping = args.attacker_list_multilabelFlipping

    for i in range(args.num_clients):
        model = Net()
        if args.optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.optimizer == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr)

        if i in attacker_list_labelFlipping:
            client_i = Attacker_LabelFlipping01swap(i, model, trainData[i], optimizer, criterion, device,
                                                    args.inner_epochs)
        elif i in attacker_list_labelFlippingDirectional:
            client_i = Attacker_LabelFlipping1to7(i, model, trainData[i], optimizer, criterion, device,
                                                  args.inner_epochs)
        elif i in attacker_list_multilabelFlipping:
            client_i = Attacker_LabelFlipping59to71(i, model, trainData[i], optimizer, criterion, device,
                                                  args.inner_epochs)                                          
        elif i in attacker_list_omniscient:
            client_i = Attacker_Omniscient(i, model, trainData[i], optimizer, criterion, device, args.omniscient_scale,
                                           args.inner_epochs)
        elif i in attacker_list_backdoor:
            client_i = Attacker_Backdoor(i, model, trainData[i], optimizer, criterion, device, args.inner_epochs)
            
            if 'RANDOM' in args.attacks.upper():                    #if the user chooses a random trigger for backdoor attack
                client_i.utils.setRandomTrigger(seed=args.attacks)
                print(client_i.utils.trigger_position)
                print(f'Client {i} is using a random backdoor with seed \"{args.attacks}\"')
            if 'CUSTOM' in args.attacks.upper():                   #if the user chooses a custom trigger for backdoor attack
                client_i.utils.setTrigger(*args.backdoor_trigger)
                print(client_i.utils.trigger_position)
                print(f'Client {i} is using a backdoor with hyperparameter \"{args.backdoor_trigger}\"')
            
        elif i in attacker_list_semanticBackdoor:
            client_i = Attacker_SemanticBackdoor(i, model, trainData[i], optimizer, criterion, device,
                                                 args.inner_epochs)
        else:
            client_i = Client(i, model, trainData[i], optimizer, criterion, device, args.inner_epochs)   #good clients

        server.attach(client_i)         #add clients objects to the server

    loss, accuracy = server.test()           #test to get accuracy result before training
    steps = 0
    writer.add_scalar('test/loss', loss, steps)
    writer.add_scalar('test/accuracy', accuracy, steps)

    if 'BACKDOOR' in args.attacks.upper():
        if 'SEMANTIC' in args.attacks.upper():
            loss, accuracy, bdata, bpred = server.test_semanticBackdoor()     # test for semantic backdoor attack
        else:
            loss, accuracy = server.test_backdoor()           # test for backdoor attack

        writer.add_scalar('test/loss_backdoor', loss, steps)
        writer.add_scalar('test/backdoor_success_rate', accuracy, steps)

    for j in range(args.epochs + 5):                      # for each epoch
        steps = j + 1

        #print('\n\n########EPOCH %d ########' % j)
        #print('###Model distribution###\n')
        logger.info('########EPOCH %d ########' % j)
        logger.info('###Model distribution###')
        
        server.distribute()                            #distribute the model to all clients
        #         group=Random().sample(range(5),1)
        group = range(args.num_clients)                #integer from 0 to num_clients
        attackers = server.train(group, j)                #train the clients
        #         server.train_concurrent(group)

        loss, accuracy = server.test()               # launch again testing

        writer.add_scalar('test/loss', loss, steps)
        writer.add_scalar('test/accuracy', accuracy, steps)
        
        #detected = {"Client_"+str(client.cid) : 1 if client.cid in attackers else 0 for client in server.clients}
        #writer.add_scalars("test/detected_clients", detected, steps)
        
        if 'BACKDOOR' in args.attacks.upper():
            if 'SEMANTIC' in args.attacks.upper():
                loss, accuracy, bdata, bpred = server.test_semanticBackdoor()       # launch again testing for semantic backdoor attacks
            else:
                loss, accuracy = server.test_backdoor()                              # launch again testing for backdoor attacks

            writer.add_scalar('test/loss_backdoor', loss, steps)
            writer.add_scalar('test/backdoor_success_rate', accuracy, steps)

    writer.close()
