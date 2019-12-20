import h5py as h5
import torch
from torch.utils import data
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.handlers import EarlyStopping, ModelCheckpoint, Timer
import pickle
import pathlib
from time import localtime, strftime, time
import wandb
import PIL
import json
import subprocess

#* Custom classes and functions
import src.modules.loss_funcs
from src.modules.classes import *
from src.modules.loss_funcs import *
from src.modules.helper_functions import *
from src.modules.eval_funcs import *
from src.modules.reporting import *


def calc_lr_vs_loss(model, optimizer, loss, train_generator, batch_size, n_train, n_epochs = 1, start_lr = 0.000001, end_lr = 0.1):
    '''Performs a scan over a learning rate-interval of interest with an exponentially increasing learning rate.

    Input: 
    model - model class, nn.module class
    optimizer - nn.optim class 
    loss - custom loss class 
    train_generator - a dataloader, torch.utils.data.dataloader instance.
    batch_size - batch size, int
    n_train - number of training samples, int 

    Output:
    lr - list of learning rates
    loss_vals - list of loss function values
    '''

    #* ======================================================================== 
    #* MAKE LR SCAN
    #* ========================================================================    

    #* Use a linearly or exponentially increasing LR throughout number of epochs
    n_steps = n_train*n_epochs/batch_size
    gamma = (end_lr/start_lr)**(1.0/n_steps) #* exponentially increasing lr
    
    lambda1 = lambda step: gamma**step
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    
    device = get_device()
    lr_trainer = create_supervised_trainer(model, optimizer, loss, device=device)
    
    lr = []
    loss_vals= []

    def log_lr(engine, lr, loss_vals, optimizer, scheduler, n_steps):
        loss_vals.append(engine.state.output)
        lr.append(get_lr(optimizer))
        scheduler.step()
    lr_trainer.add_event_handler(Events.ITERATION_COMPLETED, log_lr, lr, loss_vals, optimizer, scheduler, n_steps)
    
    print(strftime("%d/%m %H:%M", localtime()), ': LR-scan begun.')
    lr_trainer.run(train_generator, max_epochs=n_epochs)
    print(strftime("%d/%m %H:%M", localtime()), ': LR-scan finished!')

    return lr, loss_vals

def evaluate_model(model_dir, wandb_ID = None):
    '''Evaluates a model contained in save_dir

    Input
    save_dir: Full path to model directory to be evaluated
    '''
    
    #* ======================================================================== #
    #* SAVE OPERATION PLOTS
    #* ======================================================================== #
    if wandb_ID is not None:
        WANDB_DIR = get_project_root()+'/models'
        wandb.init(resume=True, id=wandb_ID, dir=WANDB_DIR)
    log_operation_plots(model_dir, wandb_ID=wandb_ID)

    #* ======================================================================== #
    #* PREDICT USING BEST MODEL
    #* ======================================================================== #

    predict(model_dir, wandb_ID=wandb_ID)
    
    #* ======================================================================== #
    #* REPORT PERFORMANCE
    #* ======================================================================== #

    log_performance_plots(model_dir, wandb_ID=wandb_ID) 
    summarize_model_performance(model_dir, wandb_ID=wandb_ID)

    if wandb_ID is not None:
        wandb.log()
        wandb.join()
    
    # * Update the meta_pars-file
    with open(model_dir+'/meta_pars.json') as json_file:
        meta_pars = json.load(json_file)
    meta_pars['status'] = 'Finished'
    with open(model_dir+'/meta_pars.json', 'w') as fp:
        json.dump(meta_pars, fp)
    
    # * Make a .dvc-file to track the model
    path_to_model_dir = Path(model_dir).resolve().parent
    model_name = model_dir.split('/')[-1]
    subprocess.run(['dvc', 'add', model_name], cwd=path_to_model_dir)
    
    # * Make a wandb-.dvc-file aswell if predictions are logged.
    if wandb_ID is not None:
        WANDB_NAME_IN_WANDB_DIR = wandb.run.dir.split('/')[-1]
        subprocess.run(['dvc', 'add', WANDB_NAME_IN_WANDB_DIR], cwd=WANDB_DIR+'/wandb/')

def explore_lr(hyper_pars, data_pars, architecture_pars, meta_pars, n_epochs = 1, start_lr = 0.000001, end_lr = 0.1, save = True):
    '''Calculates loss as a function of learning rate in a given interval and saves the graph and the dictionaries used to generate the plot.
    Used to choose lr-schedule.
    '''

    #* ======================================================================== 
    #* SETUP AND UNPACK
    #* ======================================================================== 
    print(strftime("%d/%m %H:%M", localtime()), ': Learning rate finder initiated...')

    batch_size = hyper_pars['batch_size']

    #* Use GPU if avaiable
    device = get_device()
    print('Used device:', device)
    
    #* The script expects a H5-file with a structure as shown at https://github.com/ehrhorn/CubeML
    #* Extract DATA parameters
    data_dir = data_pars['data_dir'] #* WHere to load data from
    
    print(strftime("%d/%m %H:%M", localtime()), ': Loading data...')
    train_set = load_data(hyper_pars, data_pars, architecture_pars, meta_pars, 'train')
    n_train = get_set_length(train_set)
    print(strftime("%d/%m %H:%M", localtime()), ': Data loaded!')

    #* ======================================================================== 
    #* MAKE LR SCAN
    #* ========================================================================    
    
    #* num_workers choice based on gut feeling - has to be high enough to not be a bottleneck
    dataloader_params_train = get_dataloader_params(batch_size, num_workers=8, shuffle=True, dataloader=data_pars['dataloader'])

    #* Setup generators
    if 'collate_fn' in data_pars:
        collate_fn = get_collate_fn(data_pars['collate_fn'])
    else:
        collate_fn = None

    train_generator = data.DataLoader(train_set, **dataloader_params_train, collate_fn=collate_fn)

    #* Initialize model
    model = MakeModel(architecture_pars, device)
    #* Makes model parameters to float precision
    model = model.float()
    model = model.to(device)

    #* Adjust optimizer-parameters in case function was called in a sloppy manner..
    hyper_pars['optimizer']['lr'] = start_lr
    optimizer = get_optimizer(model.parameters(), hyper_pars['optimizer'])
    
    #* Try to find a custom loss - if not, try torch's library
    loss = get_loss_func(architecture_pars['loss_func'])

    lr, loss_vals = calc_lr_vs_loss(model, optimizer, loss, train_generator, batch_size, n_train, n_epochs= n_epochs, start_lr = start_lr, end_lr = end_lr)
    
    #* ======================================================================== 
    #* SAVE RELEVANT THINGS
    #* ========================================================================
    
    lr_dir = make_lr_dir(data_dir, meta_pars['project'], batch_size)
    _ = make_plot({'x': [lr], 'y': [loss_vals], 'xscale': 'log', 'savefig': lr_dir+'/lr_vs_loss.png', 'xlabel': 'Learning Rate', 'ylabel': 'Loss'})
    pickle.dump(lr, open(lr_dir+'/lr.pickle', 'wb'))
    pickle.dump(loss_vals, open(lr_dir+'/loss_vals.pickle', 'wb'))

    print('')
    print(strftime("%d/%m %H:%M", localtime()), ': LR-finder saved at')
    print(lr_dir)

    with open(lr_dir+'/hyper_pars.json', 'w') as fp:
        json.dump(hyper_pars, fp)
    with open(lr_dir+'/data_pars.json', 'w') as fp:
        json.dump(data_pars, fp)
    with open(lr_dir+'/architecture_pars.json', 'w') as fp:
        json.dump(architecture_pars, fp)

def initiate_model_and_optimizer(save_dir, hyper_pars, data_pars, architecture_pars, meta_pars):
    device = get_device()
    model = MakeModel(architecture_pars, device)

    #* Makes model parameters to float precision
    model = model.float()

    if hyper_pars['lr_schedule']['lr_scheduler'] == 'CyclicLR':
        hyper_pars['optimizer']['lr'] = hyper_pars['lr_schedule']['base_lr']
    
    #* If training is to be continued on a pretrained model, load its parameters
    if meta_pars['objective'] == 'continue_training':
        checkpoint_path = get_project_root() + meta_pars['pretrained_path'] + '/backup.pth'

        checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)

        optimizer = get_optimizer(model.parameters(), hyper_pars['optimizer'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print('')
        print('###########################')
        print('# PRETRAINED MODEL LOADED #')
        print('###########################')
    
    elif meta_pars['objective'] == 'train_new':
        model = model.to(device)
        optimizer = get_optimizer(model.parameters(), hyper_pars['optimizer'])
    
    else:
        raise ValueError('Unknown objective set!')
        
    print('')
    print('Used device:', device)
    print('Model being trained:')
    print(model)
    print('')

    return model, optimizer, device

def predict(save_dir, wandb_ID = None):
    '''Predicts target-variables from a trained model and calculates desired functions of the target-variables. Predicts one file at a time.
    '''
    hyper_pars, data_pars, arch_pars, meta_pars = load_model_pars(save_dir)
    device = get_device()

    model_dir = save_dir+'/checkpoints'
    
    best_pars = find_best_model_pars(model_dir)
    
    model = MakeModel(arch_pars, device)
    model.load_state_dict(torch.load(best_pars, map_location=torch.device(device)))
    model = model.to(device)
    model = model.float()

    #* Loop over files
    print('\n', strftime("%d/%m %H:%M", localtime()), ': Prediction begun.')
    pred_filename = str(best_pars).split('.pth')[0].split('/')[-1]
    pred_full_address = save_dir+'/data/predict'+pred_filename+'.h5'
    data_dir = data_pars['data_dir'] #* Where to load data from
    N_FILES = len(list(Path(get_project_root()+data_dir).glob('*.h5')))
    i_file = 0

    with h5.File(pred_full_address, 'w') as f:
        
        for file in Path(get_project_root()+data_dir).iterdir():
            if file.suffix == '.h5':
                i_file += 1
                i_str = str(i_file) if i_file > 9 else '0'+str(i_file)
                print('%s/%d: Predicting on %s'%(i_str, N_FILES, get_path_from_root(str(file))))
                #* Extract validation data
                val_set = load_predictions(data_pars, meta_pars, 'val', file)
                predictions = {key: [] for key in val_set.targets}
                truths = {key: [] for key in val_set.targets}

                error_from_preds = {}

                #* Predict 
                if 'val_batch_size' in data_pars:
                    val_batch_size = data_pars['val_batch_size']
                else:
                    val_batch_size = 256
                dataloader_params = {'batch_size': val_batch_size, 'shuffle': False, 'num_workers': 8}

                #* Setup generators
                collate_fn = get_collate_fn(data_pars)
                indices = sort_indices(val_set, data_pars, dataloader_params=dataloader_params)
                val_generator = data.DataLoader(val_set, **dataloader_params, collate_fn=collate_fn)

                #* Use IGNITE to predict
                def log_prediction(engine):
                    pred, truth = engine.state.output[0], engine.state.output[1]
                    
                    #* give list of functions to calculate
                    for i_batch in range(pred.shape[0]):
                        for i_key, key in enumerate(predictions):
                            predictions[key].append(pred[i_batch, i_key])
                            truths[key].append(truth[i_batch, i_key])
                
                evaluator_val = create_supervised_evaluator(model, device=device)
                evaluator_val.add_event_handler(Events.ITERATION_COMPLETED, log_prediction)
                evaluator_val.run(val_generator)

                #* Sort w.r.t. index before saving
                for key, values in predictions.items():
                    _, sorted_vals = sort_pairs(indices, values)
                    predictions[key] = sorted_vals
                #* Sort w.r.t. index before saving
                for key, values in truths.items():
                    _, sorted_vals = sort_pairs(indices, values)
                    truths[key] = sorted_vals
                indices = sorted(indices)

                #* Run predictions through desired functions - transform back to 'true' values, if transformed
                predictions_transformed = inverse_transform_predictions(predictions, predictions.keys(), save_dir)
                truths_transformed = inverse_transform_predictions(truths, truths.keys(), save_dir)

                eval_functions = get_eval_functions(meta_pars)
                for func in eval_functions:
                    error_from_preds[func.__name__] = func(predictions_transformed, truths_transformed)

                #* Save predictions
                name = str(file).split('.')[-2].split('/')[-1]
                grp = f.create_group(name)
                grp.create_dataset('index', data=np.array(indices))
                
                for key, pred in predictions.items():
                    grp.create_dataset(key, data=np.array([x.cpu().numpy() for x in pred]))
                for key, pred in error_from_preds.items():
                    grp.create_dataset(key, data=np.array([x.cpu().numpy() for x in pred]))
                
        print(strftime("%d/%m %H:%M", localtime()), ': Predictions finished!')
    
def run_experiments(log=True):
    exp_dir = get_project_root() + '/experiments/'

    #* Loop over experiments and delete exp_file if successfully run 
    for file in Path(exp_dir).iterdir():
        if str(file).split('.')[-1] == 'json':
            
            with open(file) as json_file:
                dicts = json.load(json_file)
                hyper_pars = dicts['hyper_pars']
                data_pars = dicts['data_pars']
                arch_pars = dicts['arch_pars']
                meta_pars = dicts['meta_pars']

            #* Delete file
            Path(file).unlink()

            #* Only scan new experiments
            if meta_pars['objective'] == 'continue_training':
                scan = False
            else:
                scan= True

            model_dir, wandb_ID = train_model(hyper_pars, data_pars, arch_pars, meta_pars, scan_lr_before_train=scan, log=log)
            if log:
                evaluate_model(model_dir, wandb_ID=wandb_ID)

def train(save_dir, hyper_pars, data_pars, architecture_pars, meta_pars, earlystopping = True, scan_lr_before_train = False, wandb_ID=None, log=True):
    
    if 'val_batch_size' not in data_pars:
        data_pars['val_batch_size'] = 256
    batch_size = hyper_pars['batch_size']
    val_batch_size = data_pars['val_batch_size']
    max_epochs = hyper_pars['max_epochs']
    early_stop_patience = hyper_pars['early_stop_patience']

    print(strftime("%d/%m %H:%M", localtime()), ': Loading data...')
    train_set = load_data(hyper_pars, data_pars, architecture_pars, meta_pars, 'train')
    
    #* Only calculate train error on a fraction of the training data - a fraction equal to val. frac.
    data_pars_copy = data_pars.copy()
    hyper_pars_copy = hyper_pars.copy()
    data_pars_copy['train_frac'] = data_pars['val_frac']
    data_pars_copy['n_train_events_wanted'] = data_pars.get('n_val_events_wanted', np.inf)
    hyper_pars_copy['batch_size'] = data_pars['val_batch_size']
    trainerr_set = load_data(hyper_pars_copy, data_pars_copy, architecture_pars, meta_pars, 'train')
    val_set = load_data(hyper_pars, data_pars, architecture_pars, meta_pars, 'val')

    n_train = get_set_length(train_set)
    n_val = get_set_length(val_set)
    
    if log:
        wandb.config.update({'Trainset size': n_train})
        wandb.config.update({'Val. set size': n_val})
    print(strftime("%d/%m %H:%M", localtime()), ': Data loaded!')
    print('\nTrain set size: %d'%(n_train))
    print('Val. set size: %d'%(n_val))
    
    #* ======================================================================== #
    #* SETUP TRAINING
    #* ======================================================================== #

    #* num_workers choice based on gut feeling - has to be high enough to not be a bottleneck
    dataloader_params_train = get_dataloader_params(batch_size, num_workers=8, shuffle=True, dataloader=data_pars['dataloader'])
    dataloader_params_eval = get_dataloader_params(val_batch_size, num_workers=8, shuffle=False, dataloader=data_pars['dataloader'])
    dataloader_params_trainerr = get_dataloader_params(val_batch_size, num_workers=8, shuffle=False, dataloader=data_pars['dataloader'])


    #* Initialize model and log it - use GPU if available
    model, optimizer, device = initiate_model_and_optimizer(save_dir, hyper_pars, data_pars, architecture_pars, meta_pars)
    if log:
        with open(save_dir+'/model_arch.yml', 'w') as f:
            print(model, file=f)
        wandb.save(save_dir+'/model_arch.yml')
        wandb.config.update({'Model parameters': get_n_parameters(model)})

    #* Get type of scheduler, since different schedulers need different kinds of updating
    lr_scheduler = get_lr_scheduler(hyper_pars['lr_schedule'], optimizer, batch_size, n_train)
    type_lr_scheduler = type(lr_scheduler)
    loss = get_loss_func(architecture_pars['loss_func'])

    #* Setup generators - make a generator for training, validation on trainset and validation on test set
    collate_fn = get_collate_fn(data_pars)
    train_generator = data.DataLoader(train_set, **dataloader_params_train, collate_fn=collate_fn)#, pin_memory=True)
    val_generator = data.DataLoader(val_set, **dataloader_params_eval, collate_fn=collate_fn)#, pin_memory=True)
    trainerr_generator = data.DataLoader(trainerr_set, **dataloader_params_trainerr, collate_fn=collate_fn)#, pin_memory=True)
    
    #* Use IGNITE to train
    trainer = create_supervised_trainer(model, optimizer, loss, device=device)
    evaluator_val = create_supervised_evaluator(model, metrics={'custom_loss': Loss(loss)}, device=device)
    evaluator_train = create_supervised_evaluator(model, metrics={'custom_loss': Loss(loss)}, device=device)

    #* ======================================================================== #
    #* SETUP SAVING OF IMPROVED MODELS
    #* ======================================================================== #* 
    
    def custom_score_function(engine):
        loss = engine.state.metrics['custom_loss']
        return -loss
    
    if log:
        name = ''
        checkpointer = ModelCheckpoint(dirname = save_dir+'/checkpoints', filename_prefix = name, create_dir = True, save_as_state_dict = True, score_function = custom_score_function, score_name = 'Loss', n_saved = 5, require_empty = True)
        
        #* Add handler to evaluator
        checkpointer_dict = {'model': model}
        evaluator_val.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, checkpointer_dict)

    #* ======================================================================== #
    #* SETUP EARLY STOPPING
    #* ======================================================================== #

    #* patience = how long to wait before stopping according to score_func, trainer = which engine to stop.
    if earlystopping:
        print('Early stopping activated!')
        early_stop_handler = EarlyStopping(patience=early_stop_patience,
                                        score_function=custom_score_function, 
                                        trainer=trainer)
        evaluator_val.add_event_handler(Events.EPOCH_COMPLETED, early_stop_handler)


    #* ======================================================================== #
    #* DO LEARNING RATE SCAN
    #* ======================================================================== #

    if scan_lr_before_train:
        pretrain_hyper_pars = hyper_pars['optimizer'].copy()
        pretrain_hyper_pars['lr'] = 0.000001

        lr_model = MakeModel(architecture_pars, device)
        lr_model = lr_model.float()
        lr_model = lr_model.to(device)

        pretrain_optimizer = get_optimizer(lr_model.parameters(), pretrain_hyper_pars)
        pretrain_lr, pretrain_losses = calc_lr_vs_loss(lr_model, pretrain_optimizer, loss, train_generator, batch_size, n_train, start_lr=pretrain_hyper_pars['lr'])

        vlines = []
        if 'base_lr' in hyper_pars['lr_schedule']:
            vlines.append(hyper_pars['lr_schedule']['base_lr'])
        if 'max_lr' in hyper_pars['lr_schedule']:
            vlines.append(hyper_pars['lr_schedule']['max_lr'])
        
        if log:
            img_address = save_dir+'/figures/pretrain_lr_vs_loss.png'
            _ = make_plot({'x': [pretrain_lr], 'y': [pretrain_losses], 'xscale': 'log', 'savefig': img_address, 'xlabel': 'Learning Rate', 'ylabel': 'Loss', 'axvline': vlines})
            pickle.dump(pretrain_lr, open(save_dir+'/pretrain_lr.pickle', 'wb'))
            pickle.dump(pretrain_losses, open(save_dir+'/pretrain_loss_vals.pickle', 'wb'))
            im = PIL.Image.open(img_address)
            wandb.log({'Pretrain LR-scan': wandb.Image(im, caption='Pretrain LR-scan')}, commit = False)

    #* ======================================================================== #
    #* SETUP LOGGING
    #* ======================================================================== #*   
    
    #* If continuing training, get how many epochs completed
    if 'epochs_completed' in hyper_pars:
        epochs_completed = hyper_pars['epochs_completed']
    else:
        epochs_completed = 0

    #* Print log
    def print_log(engine, set_name, metric_name):
        print("Epoch: {}/{} - {} {}: {:.2e}"
            .format(trainer.state.epoch + epochs_completed, max_epochs + epochs_completed, set_name, metric_name, engine.state.metrics[metric_name]))

    evaluator_train.add_event_handler(Events.COMPLETED, print_log, "train", 'custom_loss')
    evaluator_val.add_event_handler(Events.COMPLETED, print_log, "validation", 'custom_loss')

    #* Log locally and to W&B
    if log:
        def log_metric(engine, set_name, metric_name, list_address):
            append_list_and_save(list_address, engine.state.metrics[metric_name])
            wandb.log({set_name+metric_name: engine.state.metrics[metric_name]}, step=trainer.state.epoch + epochs_completed)

        def log_lr(engine, set_name, optimizer, list_address):
            number = get_lr(optimizer)
            append_list_and_save(list_address, number)
            wandb.log({set_name: number}, step=trainer.state.epoch + epochs_completed)

        def log_epoch(engine, list_address):
            append_list_and_save(list_address, trainer.state.epoch + epochs_completed)

        evaluator_train.add_event_handler(Events.COMPLETED, log_metric, 'Graphs/train ', 'custom_loss', save_dir+'/data/train_error.pickle')
        evaluator_val.add_event_handler(Events.COMPLETED, log_metric, 'Graphs/val. ', 'custom_loss', save_dir+'/data/val_error.pickle')
        evaluator_train.add_event_handler(Events.COMPLETED, log_lr, 'Graphs/learning rate', optimizer, save_dir+'/data/lr.pickle')
        evaluator_val.add_event_handler(Events.COMPLETED, log_epoch, save_dir+'/data/epochs.pickle')

    #* Time training and evaluation
    time_trainer = Timer(average=True)
    time_trainer.attach(trainer, resume=Events.EPOCH_STARTED, pause=Events.EPOCH_COMPLETED, step=Events.EPOCH_COMPLETED)

    time_evaluator = Timer(average=True)
    time_evaluator.attach(evaluator_val, resume=Events.EPOCH_STARTED, pause=Events.EPOCH_COMPLETED, step=Events.EPOCH_COMPLETED)

    #* Call evaluator after each epoch
    def evaluate(trainer):
        print('\nEpoch completed',strftime("%d/%m %H:%M", localtime()))

        #! FullBatchLoader has to be treated in a special way! See the class, it has to be shuffled every epoch
        if data_pars['dataloader'] == 'FullBatchLoader':
            train_set.make_batches()

        #* Log weights, biases and gradients in histograms.
        if log:
            i_layer = 1
            step = trainer.state.epoch + epochs_completed
            for entry in model.mods: 
                if type(entry) == nn.modules.container.Sequential:
                    for seq_entry in entry:
                        i_layer = log_weights_and_grads(i_layer, seq_entry, step)
                        
                        i_layer = log_weights_and_grads(i_layer, entry, step)
        
        #* Run evaluation on train- and validation-sets
        evaluator_train.run(trainerr_generator)
        evaluator_val.run(val_generator)
        
        #* Log maximum memory allocated and speed.
        if log:
            wandb.config.update({'Avg. Events/second (train)': n_train/time_trainer.value()}, allow_val_change=True)
            wandb.config.update({'Avg. Events/second (eval.)': n_val/time_evaluator.value()}, allow_val_change=True)
            if torch.cuda.is_available():
                max_memory_allocated = torch.cuda.max_memory_allocated(device=device)/(1024*1024)
                wandb.config.update({'Max memory allocated [MiB]': max_memory_allocated}, allow_val_change=True)

    def save_backup(trainer):   
        '''Save a backup after each epoch in case something crashes...
        '''
        backup = {'epochs_completed': trainer.state.epoch + epochs_completed,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                    }
        torch.save(backup, save_dir + '/backup.pth')
    
    trainer.add_event_handler(Events.EPOCH_COMPLETED, evaluate)
    if log:
        trainer.add_event_handler(Events.EPOCH_COMPLETED, save_backup)

    #* ======================================================================== #
    #* SETUP LEARNING RATE SCHEDULER
    #* ======================================================================== #

    if type_lr_scheduler == torch.optim.lr_scheduler.CyclicLR:
        
        def update_lr(engine, lr_scheduler):
            lr_scheduler.step()

        trainer.add_event_handler(Events.ITERATION_COMPLETED, update_lr, lr_scheduler)
    elif type_lr_scheduler == torch.optim.lr_scheduler.ReduceLROnPlateau:

        def update_lr(engine, lr_scheduler, evaluator, metric_name):
            loss_val = evaluator.state.metrics[metric_name]
            lr_scheduler.step(loss_val)

        evaluator_val.add_event_handler(Events.COMPLETED, update_lr, lr_scheduler, evaluator_val, 'custom_loss')
    elif type_lr_scheduler == torch.optim.lr_scheduler.LambdaLR:
        
        def update_lr(engine, lr_scheduler):
            lr_scheduler.step()
        
        trainer.add_event_handler(Events.ITERATION_COMPLETED, update_lr, lr_scheduler)
    else:
        raise ValueError('Undefined lr_scheduler used for updating LR!')

    #* ======================================================================== #
    #* START TRAINING
    #* ======================================================================== # 
           
    print('Training begun')
    trainer.run(train_generator, max_epochs=max_epochs)
    print('\nTraining finished!')

def train_model(hyper_pars, data_pars, architecture_pars, meta_pars, scan_lr_before_train = False, log=True):
    
    #* ======================================================================== 
    #* SETUP AND LOAD DATA
    #* ======================================================================== 

    #* The script expects a H5-file with a structure as shown at https://github.com/ehrhorn/CubeML
    data_dir = data_pars['data_dir'] #* WHere to load data from
    file_keys = data_pars['file_keys'] #* which cleaning lvl and transform should be applied?
    group = meta_pars['group'] #* under which dir to save?
    project = meta_pars['project']

    if log:
        save_dir = make_model_dir(group, data_dir, file_keys, project)
        wandb_ID = save_dir.split('/')[-1]
        print('Model saved at', save_dir)
    else:
        save_dir = None
        wandb_ID = None

    #* If training is on a pretrained model, copy and update data- and hyperpars with potential new things
    if meta_pars['objective'] == 'continue_training':
        hyper_pars, data_pars, architecture_pars = update_model_pars(hyper_pars, data_pars, meta_pars)
    
    #* Save model parameters on W&B AND LOCALLY!
    #* Shut down W&B first, if it is already running
    if log:
        WANDB_NAME = save_dir.split('/')[-1]
        MODEL_NAME = save_dir.split('/')[-1]
        WANDB_DIR = get_project_root()+'/models'

        wandb.init(project=meta_pars['project'], name=WANDB_NAME, tags=meta_pars['tags'], id=wandb_ID, reinit=True, dir=WANDB_DIR)
        wandb.config.update(hyper_pars)
        wandb.config.update(data_pars)
        wandb.config.update(architecture_pars)

        with open(save_dir+'/hyper_pars.json', 'w') as fp:
            json.dump(hyper_pars, fp)
        
        with open(save_dir+'/data_pars.json', 'w') as fp:
            json.dump(data_pars, fp)
        
        with open(save_dir+'/architecture_pars.json', 'w') as fp:
            json.dump(architecture_pars, fp)
        
        meta_pars['status'] = 'Failed'
        with open(save_dir+'/meta_pars.json', 'w') as fp:
            json.dump(meta_pars, fp)

        # * Add to .gitignore immediately
        gitignore_model_path = '/'.join(save_dir.split('/')[:-1]) + '/.gitignore'
        with open(gitignore_model_path,'a') as f:
            f.write('/%s\n'%(MODEL_NAME))
        
        WANDB_NAME_IN_WANDB_DIR = wandb.run.dir.split('/')[-1]
        gitignore_wandb_path = WANDB_DIR+'/wandb/.gitignore'
        with open(gitignore_wandb_path,'a') as f:
            f.write('/%s\n'%(WANDB_NAME_IN_WANDB_DIR))

    else:
        print('Logging turned off.')

    train(save_dir, hyper_pars, data_pars, architecture_pars, meta_pars, scan_lr_before_train = scan_lr_before_train, wandb_ID=wandb_ID, log=log)
    
    # * Update the meta_pars-file and add .dvc-files to track the model in the wandb-dir and the models-dir
    if log:
        with open(save_dir+'/meta_pars.json') as json_file:
            meta_pars = json.load(json_file)
        meta_pars['status'] = 'Trained'
        with open(save_dir+'/meta_pars.json', 'w') as fp:
            json.dump(meta_pars, fp)

    return save_dir, wandb_ID
    
