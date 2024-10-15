import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold, StratifiedKFold
import torch.optim as optim
from torch.utils.data import SubsetRandomSampler
from src.utils.load_data import CustomTensorDataset, TabularDataset
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from src.utils import *
import monai
from src.models import load_model
import optuna

def valid_step(model, criterion, val_loader):
    model.eval()
    soft = nn.Softmax(dim=1)
    avg_loss = 0.0
    avg_acc = 0.0
    y_pred = []
    prediction = []
    y_label = []
    with torch.no_grad():
        for i, (x_imgs, labels) in enumerate(val_loader):
            # forward pass
            x_imgs, labels = x_imgs.to(args.device), labels.to(args.device)
            outputs = model(x_imgs)
            prediction += (soft(outputs)).tolist()
            y_label += labels.tolist()
            #labels = labels.unsqueeze(1)
            loss = criterion(outputs, labels)
            # gather statistics
            avg_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            y_pred += preds.tolist()
            avg_acc += torch.sum(preds == labels.data).item()

    pred_proba = np.array(prediction)[:,1]
    auc = roc_auc_score(np.array(y_label), pred_proba)
   
    return {'loss': avg_loss / len(val_loader), 'accuracy': avg_acc / len(y_label), 'auc': auc, 
            'labels': y_label, 'pred': y_pred, 'predicted_proba': pred_proba}



def train_step(model, criterion, optimizer, train_loader):
    model.train()
    soft = nn.Softmax(dim=1)
    prediction = []
    avg_loss = 0.0
    avg_acc = 0.0
    y_label = []
    for i, (x_imgs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        # forward pass
        #labels = labels.astype(np.int64) ##### trial
        x_imgs, labels = x_imgs.to(args.device), labels.to(args.device)
        probs = model(x_imgs)
        prediction += (soft(probs)).tolist()
        y_label += labels.tolist()
        #labels = labels.unsqueeze(1)
        loss = criterion(probs, labels)
        # back-prop
        loss.backward()
        optimizer.step()
        # gather statistics
        avg_loss += loss.item()
        _, preds = torch.max(probs, 1)
        avg_acc += torch.sum(preds == labels.data).item()

    pred_proba = np.array(prediction)[:,1]
    auc = roc_auc_score(np.array(y_label), pred_proba)

    return {'loss': avg_loss / len(train_loader), 'accuracy': avg_acc / len(y_label), 'auc':auc}


def plot_losses(n_epoch, tr_losses, val_losses, params, stage='val', value='loss'):
    plt.figure()
    # plotting the line 1 points 
    plt.plot(n_epoch, tr_losses, label = "train")
    # plotting the line 2 points 
    if stage == 'val':
        plt.plot(n_epoch, val_losses, label = "valid")
        title= ' VAL \n' + str(params)
        plt.title(title)
    else:
        plt.plot(n_epoch, val_losses, label = "test")
        # Set a title of the current axes.
        title= ' TEST \n' + str(params)
        plt.title(title)

    plt.xlabel('epochs')
    # Set the y axis label of the current axis.
    if value =='loss':
        plt.ylabel('loss')
    else:
        plt.ylabel('accuracy')
    # show a legend on the plot
    plt.legend()
    return plt, title




def train_model(trial, train_data, image_details, transform, loader_kwargs, seed):
    # define search space for tuning hyperparameters  
    if args.model == 'CNN_opt':
        rows, columns, size = image_details
        params = { 
                'learning_rate': trial.suggest_categorical('learning_rate',[1e-3, 1e-4]),
                'weight_decay': trial.suggest_categorical('weight_decay', [1e-8]),
                'optimizer': trial.suggest_categorical("optimizer", ["Adam"]), 
                'n_kernels': trial.suggest_categorical('n_kernels', [2, 4]),
                'n_layers': trial.suggest_categorical('n_layers', [2, 3]),
                'dropout': trial.suggest_categorical('dropout', [0.4, 0.5]),
                }
    elif args.model == 'MLP_opt':
        params = { 
                'learning_rate': trial.suggest_categorical('learning_rate',[1e-3, 1e-4]),
                'weight_decay': trial.suggest_categorical('weight_decay', [1e-8]),
                'optimizer': trial.suggest_categorical("optimizer", ["Adam"]), 
                'n_layers': trial.suggest_categorical('n_layers', [1, 2, 3]),
                'input_dim': trial.suggest_categorical('input_dim',[64, 128]),
                'dropout': trial.suggest_categorical('dropout', [0.4, 0.5]),
                }


    # Check duplication and skip if it's detected.
    for t in trial.study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        if 'best_epoch' in t.params.keys():
            del t.params["best_epoch"]
        if t.params == trial.params:
            raise optuna.exceptions.TrialPruned('Duplicate parameter set')
           
            #return t.value  # Return the previous value without re-evaluating it.
        
    criterion = nn.CrossEntropyLoss()

    # define splits for the k-fold
    splits= StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

    # scores of a configuration of hyperparameters on each fold
    scores_tr = []
    scores_val = []
    scores_val_auc = []
    input_dim = train_data[0][0].shape[0]
    for fold, (train_idx, val_idx) in enumerate(splits.split(train_data[0], train_data[1])):
        fix_random_seed(seed=seed)
        model = load_model(params, input_dim, args.model)

        # define optimizer
        if params['optimizer'] in ["Adam", "AdamW", "Adagrad"]:
            optimizer = getattr(optim, params['optimizer'])(model.parameters(), lr= params['learning_rate'], weight_decay=params['weight_decay'])
        else:
            optimizer = getattr(optim, params['optimizer'])(model.parameters(), lr= params['learning_rate'])

    
        if args.img_type == 'digits':
            train_dataset = CustomTensorDataset(data=train_data, rows=rows, columns=columns, size=size, transform=transform, seed=seed)
            val_dataset = CustomTensorDataset(data=train_data, rows=rows, columns=columns, size=size, transform=None, seed=seed)
        elif args.img_type == 'tabular':
            train_dataset = TabularDataset(data=train_data)
            val_dataset = TabularDataset(data=train_data)

        np.random.shuffle(train_idx)
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = val_idx
        train_loader= monai.data.DataLoader(train_dataset, **loader_kwargs, sampler= train_sampler)
        val_loader  = monai.data.DataLoader(val_dataset, **loader_kwargs, sampler= valid_sampler)


        max_epochs = 500
        # best epoch is chosen in based on the first fols metric
        if fold == 0:
            epochs = max_epochs
        # best epoch for the other folds 
        else:
            epochs = best_epoch

        patience = 50 #15
        count_patience = 0
        best_loss = np.inf
        best_auc = 0
        for epoch in range(1, epochs+1):
            train_stats = train_step(model, criterion, optimizer, train_loader)
            valid_stats = valid_step(model, criterion, val_loader)
            if fold == 0:
                if epoch >=4: #100
                    if valid_stats['auc'] > best_auc:
                        best_auc = valid_stats['auc']
                        best_train_acc = train_stats['accuracy']
                        best_val_acc = valid_stats['accuracy']
                        #best_val_auc = valid_stats['auc']
                        best_epoch = epoch
                        count_patience = 0

                    else:
                        count_patience += 1
                    
                    if count_patience == patience:
                        break
            else:

                best_train_acc = train_stats['accuracy']
                best_val_acc = valid_stats['accuracy']

                best_auc = valid_stats['auc']

        trial.suggest_int("best_epoch", best_epoch , best_epoch)
        scores_tr.append(best_train_acc)
        scores_val.append(best_val_acc)
        scores_val_auc.append(best_auc)

    print()
    print(f'Train acc mean over folds: {np.mean(scores_tr)}, Train acc std {np.std(scores_tr)}')
    print(f'Val acc mean over folds: {np.mean(scores_val)}, Val acc std {np.std(scores_val)}')

    print()
    print(f'Val auc mean over folds: {np.mean(scores_val_auc)}, Val auc std {np.std(scores_val_auc)}')


    return np.mean(scores_val_auc)


if __name__ == "__main__":
    pass
