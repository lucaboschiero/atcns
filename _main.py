from __future__ import print_function

import torch
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
import pandas as pd

from clients_attackers import *
from server import Server
import os

from utils.logger import get_logger

# Get the logger
logger = get_logger()

# Function to initialize the log table if it doesn't exist
def initialize_log_table(filepath, columns):
    if not os.path.exists(filepath):
        # Create an empty DataFrame with specified columns
        log_table = pd.DataFrame(columns=columns)
        # Save to CSV
        log_table.to_csv(filepath, index=False)
        print(f"Log table initialized at {filepath}")
    else:
        print(f"Log table already exists at {filepath}")

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

        label_flipping_type = 'NA'      # initialized as Not Assigned
        label = torch.ones(args.num_clients)
        for i in args.attacker_list_labelFlipping:
            label[i] = 0
            label_flipping_type = 'SF'
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
            if label_flipping_type == 'SF':
                label_flipping_type = 'SMF'
            elif label_flipping_type == 'NA':
                label_flipping_type = 'MF'

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

    loss, accuracy, labelflipping_asr = server.test(label_flipping_type)           #test to get accuracy result before training
    steps = 0
    writer.add_scalar('test/loss', loss, steps)
    writer.add_scalar('test/accuracy', accuracy, steps)

    asr_backdoor = 0
    asr_labelflipping = 0

    if 'BACKDOOR' in args.attacks.upper():
        if 'SEMANTIC' in args.attacks.upper():
            loss, accuracy, bdata, bpred = server.test_semanticBackdoor()     # test for semantic backdoor attack
        else:
            loss, accuracy, backdoor_asr = server.test_backdoor()           # test for backdoor attack

        writer.add_scalar('test/loss_backdoor', loss, steps)
        writer.add_scalar('test/backdoor_success_rate', accuracy, steps)

    list_attackers = [index for index, value in enumerate(label) if value == 0]
    ED_epoch = '*'
    false_positives_vec = []
    attackers = 0
    b = True  # To disable the defence mechanisms when all attacker identified correctly
    detection_time_vec = []
    number_of_attacker_type = 0

    remaining_clients = [] # Benign clients correct identified with detection mechanism

    for j in range(args.epochs):                      # for each epoch
        steps = j + 1

        #print('\n\n########EPOCH %d ########' % j)
        #print('###Model distribution###\n')
        logger.info('########EPOCH %d ########' % j)
        logger.info('###Model distribution###')
        
        server.distribute()                            #distribute the model to all clients
        #         group=Random().sample(range(5),1)
        group = range(args.num_clients)                #integer from 0 to num_clients
        if ED_epoch != '*':
            if b:
                server.set_AR('fedavg')
                b = False # Defense mechanism disabled
                remaining_clients = [i for i in group if i not in attackers]
            group = remaining_clients

        attackers, detection_time = server.train(group, j)                #train the clients
        #         server.train_concurrent(group)
        print("ATTACKERS: ", attackers)
        print("LIST ATTACKERS: ", list_attackers)
        print("ED: ", ED_epoch)
        if not isinstance(attackers, int):

            print("Detection time: ", detection_time)

            detection_time_vec.append(detection_time)
            if len(attackers) > 0:
                false_positive = (len([i for i in attackers if label[i]==1])/ len(attackers)) *100
                false_positives_vec.append(false_positive)
                #print("FALSE POSITIVE: ", false_positive)
        if attackers == list_attackers:
            if ED_epoch == '*':
                ED_epoch = j

        loss, Testaccuracy, labelflipping_asr = server.test(label_flipping_type)               # launch again testing

        asr_labelflipping = labelflipping_asr


        writer.add_scalar('test/loss', loss, steps)
        writer.add_scalar('test/accuracy', accuracy, steps)
        
        #detected = {"Client_"+str(client.cid) : 1 if client.cid in attackers else 0 for client in server.clients}
        #writer.add_scalars("test/detected_clients", detected, steps)
        
        if 'BACKDOOR' in args.attacks.upper():
            if 'SEMANTIC' in args.attacks.upper():
                loss, accuracy, bdata, bpred = server.test_semanticBackdoor()       # launch again testing for semantic backdoor attacks
            else:
                loss, accuracy, backdoor_asr = server.test_backdoor()               # launch again testing for backdoor attacks
                print("ASR backdoor: ", backdoor_asr)                           
                
                asr_backdoor = backdoor_asr

            writer.add_scalar('test/loss_backdoor', loss, steps)
            writer.add_scalar('test/backdoor_success_rate', accuracy, steps)



    writer.close()


    # Compute percentage of Label Flipping attacker
    total = 0
    s2 = ""


    if len(attacker_list_multilabelFlipping) > 0 and len(attacker_list_labelFlipping) > 0 and len(attacker_list_backdoor) > 0:
        s2 = "3attackers"
        total_str = ""
        number_of_attacker_type = 3
    else:
        if len(attacker_list_labelFlipping) > 0:
            s2 = "SF"
            if len(attacker_list_backdoor) > 0:
                total = (len(attacker_list_labelFlipping) / (len(attacker_list_labelFlipping) + len(attacker_list_backdoor)))
                #print("TOTAL : ", total)
                total_str = f"{total:.2f}".replace('.', ',')
                number_of_attacker_type = 2
        else:
            if len(attacker_list_multilabelFlipping) > 0:
                s2 = "MF"
            if len(attacker_list_backdoor) > 0:
                total = (len(attacker_list_multilabelFlipping) / (len(attacker_list_multilabelFlipping) + len(attacker_list_backdoor)))
                #print("TOTAL : ", total)
                total_str = f"{total:.2f}".replace('.', ',')
                number_of_attacker_type = 2

    #Compute the average detection time
    avg_det_time = f"{(sum(detection_time_vec) / len(detection_time_vec)) :.2f}"
    print("Average detection time: ", avg_det_time)
    with open(f"./logs/detection_time.txt", "w") as f:
        f.write(avg_det_time)

    # Compute the percentage of attacker
    n_attackers = sum(1 for i in label if i == 0)
    percentageOfAttackers = (n_attackers / args.num_clients) * 100

    # Table for accuracy
    # Initialize the filepath
    filepath = f"./logs/{args.dataset.capitalize()}/Accuracy/{total_str}{s2}.csv"
    # Initialize the log table
    initialize_log_table(filepath, ["% of attackers", "mstold", "density", "foolsgold", "mst", "kmeans"])
    add_or_update_row(filepath=filepath, attackers_percentage=percentageOfAttackers, column_name=args.AR, value=Testaccuracy)

    # Table for early detection
    # Initialize the filepath
    filepath = f"./logs/{args.dataset.capitalize()}/EarlyDetection/{total_str}{s2}.csv"
    # Initialize the log table
    initialize_log_table(filepath, ["% of attackers", "mstold", "density", "foolsgold", "mst", "kmeans"])
    add_or_update_row(filepath=filepath, attackers_percentage=percentageOfAttackers, column_name=args.AR, value=ED_epoch)

    #Table for false positives
    # Initialize the filepath
    filepath = f"./logs/{args.dataset.capitalize()}/FP/{total_str}{s2}.csv"
    # Initialize the log table
    initialize_log_table(filepath, ["% of attackers", "mstold", "density", "foolsgold", "mst", "kmeans"])
    FPmean = f"{(sum(false_positives_vec) / len(false_positives_vec)) :.2f}"
    print("False positive mean: ", FPmean)
    add_or_update_row(filepath=filepath, attackers_percentage=percentageOfAttackers, column_name=args.AR, value=FPmean)

    #Table for ASR
    # Initialize the filepath
    filepath = f"./logs/{args.dataset.capitalize()}/ASR/{total_str}{s2}.csv"
    # Initialize the log table
    initialize_log_table(filepath, ["% of attackers", "mstold", "density", "foolsgold", "mst", "kmeans"])
    ASR_total = f"{((float(asr_labelflipping) + float(asr_backdoor)) / number_of_attacker_type):.3f}"
    print("ASR total: ", ASR_total)
    add_or_update_row(filepath=filepath, attackers_percentage=percentageOfAttackers, column_name=args.AR, value=ASR_total)


def add_or_update_row(filepath, attackers_percentage, column_name, value):
    # Load the existing log table
    log_table = pd.read_csv(filepath)
    # Check if a row with the given % of attackers already exists
    row_index = log_table.index[log_table["% of attackers"] == attackers_percentage].tolist()
    
    if row_index:
        # Row exists; update the specified column
        log_table.at[row_index[0], column_name] = value
        print(f"Updated row: % of attackers = {attackers_percentage}, {column_name} = {value}")
    else:
        # Row does not exist; create a new one
        new_row = {
            "% of attackers": attackers_percentage,
            "mstold": value if column_name == "mstold" else None,
            "density": value if column_name == "density" else None,
            "foolsgold": value if column_name == "foolsgold" else None,
            "mst": value if column_name == "mst" else None,
            "kmeans": value if column_name == "kmeans" else None,
        }
        # Replace append with concat
        new_row_df = pd.DataFrame([new_row])  # Convert the new row to a DataFrame
        log_table = pd.concat([log_table, new_row_df], ignore_index=True)
        print(f"Added new row: % of attackers = {attackers_percentage}, {column_name} = {value}")
    
    # Save back to the CSV
    log_table.to_csv(filepath, index=False)