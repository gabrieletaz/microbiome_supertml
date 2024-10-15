from datetime import datetime
import torch
#from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from src.utils import args
from src.models import load_model
from src.utils.load_data import load_data, CustomTensorDataset, show_batch, TabularDataset
import pandas as pd
from src.utils import *
from src.opt.training_pipeline import train_model, train_step, valid_step, plot_losses
from clearml import Task
import optuna
import torch.optim as optim
from clearml import Task, Logger
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import monai

def report_image(logger, name, title, iteration):
    # report PIL Image object
    Logger.current_logger().report_image(
        "image", 
        title, 
        iteration=iteration, 
        image=name
    )

def full_run(task, seed, iteration):
    # Dataset and Dataloader settings
    train_data, test_data, image_details, transform = load_data(seed=seed)
    rows, columns, size = image_details

    kwargs = {} if args.device=='cpu' else {'num_workers': 1}
    loader_kwargs = {'batch_size':args.batch_size, **kwargs}

    set_determinism(seed=seed)
    if args.img_type == 'digits': 
        train_dataset = CustomTensorDataset(data=train_data, rows=rows, columns=columns, size=size, transform=transform, seed=seed)
        test_dataset  = CustomTensorDataset(data=test_data, rows=rows, columns=columns, size=size, transform=None, seed=seed)
    elif args.img_type == 'tabular':
        train_dataset = TabularDataset(data=train_data)
        test_dataset  = TabularDataset(data=test_data)

    # load train and test for the 5 runs
    train_loader = monai.data.DataLoader(train_dataset, **loader_kwargs)
    test_loader = monai.data.DataLoader(test_dataset, **loader_kwargs)

    if args.img_type == 'digits':
        # save a sample image to load it on clearML
        img_train = show_batch(train_loader)
        img_test = show_batch(test_loader)
        # load the test image on ClearML to check if it's correct
        logger = task.get_logger()
        report_image(logger, name=img_train, title='train', iteration=iteration)
        report_image(logger, name=img_test, title='test', iteration=iteration)

    objective = lambda trial: train_model(trial, train_data, image_details, transform, loader_kwargs, seed=seed)

    # RUN loop for hyper-parameter optimization
    sampler = optuna.samplers.RandomSampler(seed=1) #RandomSampler
    study = optuna.create_study(direction="maximize", sampler=sampler)
    unique_trials = 16
    while unique_trials > len(set(str(t.params) for t in study.trials)):
        study.optimize(objective, n_trials=1)

    # TRAIN and TEST best configuration of hyperparameters on the entrire train and test sets
    best_params = study.best_trial.params
    task.connect(best_params, name=f'best_params_{iteration}')

    fix_random_seed(seed=seed)
    input_dim = train_data[0][0].shape[0]
    best_model = load_model(best_params, input_dim, args.model)
    criterion = nn.CrossEntropyLoss()
    if best_params['optimizer'] in  ["Adagrad", "Adam", "AdamW", "SGD"]:
        best_optimizer = getattr(optim, best_params['optimizer'])(best_model.parameters(), lr= best_params['learning_rate'], weight_decay=best_params['weight_decay'])
    else:
        best_optimizer = getattr(optim, best_params['optimizer'])(best_model.parameters(), lr= best_params['learning_rate'])
    

    n_epoch = []
    tr_losses = []
    tr_acc = []
    val_losses = []
    val_acc= []
    val_auc = []
    for epoch in range(best_params['best_epoch']):
        train_stats = train_step(best_model, criterion, best_optimizer, train_loader)
        test_rmse = valid_step(best_model,criterion, test_loader)
        n_epoch.append(epoch)
        tr_losses.append(train_stats["loss"])
        tr_acc.append(train_stats["accuracy"])
        val_losses.append(test_rmse["loss"])
        val_acc.append(test_rmse["accuracy"])
        val_auc.append(test_rmse['auc'])

    plt, title = plot_losses(n_epoch, tr_losses, val_losses, best_params, stage='test')
    task.logger.report_matplotlib_figure(title='Losses', 
                                        series= "losses: " + title, 
                                        iteration=iteration, 
                                        figure=plt)

    
    # LOG confusion matrix
    confusion = confusion_matrix(test_rmse['labels'], test_rmse['pred'], labels=[0,1])
    Logger.current_logger().report_matrix(
        "Confusion_matrix",
        f"CF{iteration}",
        iteration=iteration,
        matrix=confusion,
        xaxis="Predicted",
        yaxis="True Label",
        yaxis_reversed=True)

    print(f'Seed_{seed}: AUC = {val_auc[-1]}')
    print(f'Seed_{seed}: ACCURACY = {val_acc[-1]}')

    
    results = {'prediction': test_rmse['pred'],
               'true': test_rmse['labels'],
               'predict_proba': test_rmse['predicted_proba'],
    }  

    # Create a DataFrame
    res = pd.DataFrame(results)

    Logger.current_logger().report_table(
        "Results single run", 
        f"run_{iteration}", 
        iteration=iteration, 
        table_plot=res
    )



    
    return val_acc[-1], val_auc[-1]

def main():
    task = Task.init(project_name=f'project_name', task_name=args.exp_name)
    task.connect(args)

    accuracies = []
    auc = []
    
    for i in range(5):
        test_acc, test_auc= full_run(task, seed=i, iteration=i)
        accuracies.append(test_acc)
        auc.append(test_auc)
        


    #log hyperparameters
    task.connect(args)

    data = {
    'run1': [accuracies[0], auc[0]],
    'run2': [accuracies[1], auc[1]],
    'run3': [accuracies[2], auc[2]],
    'run4': [accuracies[3], auc[3]],
    'run5': [accuracies[4], auc[4]],
    'mean': [np.mean(accuracies), np.std(accuracies)],
    'std':  [np.mean(auc), np.std(auc)],
}   
    
    # Create a DataFrame
    df = pd.DataFrame(data, index=['accuracy', 'auc'])

    Logger.current_logger().report_table(
        "Final Results", 
        "PD 1", 
        iteration=4, 
        table_plot=df
    )


    print(f'Accuracies over 5 seeds: {accuracies}')
    print(f'Mean of accuracies over 5 seeds: {np.mean(accuracies)}, Std of accuracies over 5 seeds: {np.std(accuracies)}')
    print()
    print(f'AUC over 5 seeds: {auc}')
    print(f'Mean of AUC over 5 seeds: {np.mean(auc)}, Std of AUC over 5 seeds: {np.std(auc)}')
    print('\n'+24*'='+' Experiment Ended '+24*'='+'\n') 


if __name__ == "__main__":
    main()
