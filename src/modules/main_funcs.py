import h5py as h5
import torch
from torch.utils import data
from ignite.engine import Engine, Events, create_supervised_evaluator, create_supervised_trainer 
from ignite.metrics import Accuracy, Loss
from ignite.handlers import EarlyStopping, ModelCheckpoint, Timer, TerminateOnNan
from ignite.utils import convert_tensor
import pickle
import pathlib
from time import localtime, strftime, time
import wandb
import PIL
import json
import subprocess
import multiprocessing

# Custom classes and functions
import src.modules.loss_funcs
from src.modules.classes import *
from src.modules.loss_funcs import *
from src.modules.helper_functions import *
from src.modules.eval_funcs import *
from src.modules.reporting import *
# from src.modules.preprocessing import DomChargeScaler, EnergyNoLogTransformer

def calc_lr_vs_loss(model, optimizer, loss, train_generator, BATCH_SIZE, N_TRAIN, gpus, n_epochs=1, start_lr=0.000001, end_lr=0.1):
    '''Performs a scan over a learning rate-interval of interest with an exponentially increasing learning rate.

    Input: 
    model - model class, nn.module class
    optimizer - nn.optim class 
    loss - custom loss class 
    train_generator - a dataloader, torch.utils.data.dataloader instance.
    BATCH_SIZE - batch size, int
    N_TRAIN - number of training samples, int 

    Output:
    lr - list of learning rates
    loss_vals - list of loss function values
    '''

    # ======================================================================== 
    # MAKE LR SCAN
    # ========================================================================    

    # Use a linearly or exponentially increasing LR throughout number of epochs
    n_steps = N_TRAIN*n_epochs/BATCH_SIZE
    gamma = (end_lr/start_lr)**(1.0/n_steps) # exponentially increasing lr
    
    lambda1 = lambda step: gamma**step
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    
    device = get_device(gpus[0])
    lr_trainer = create_supervised_trainer(model, optimizer, loss, device=device)
    
    lr = []
    loss_vals= []
    LOG_EVERY = 200000
    def log_lr(engine, lr, loss_vals, optimizer, scheduler, n_steps):
        loss_vals.append(engine.state.output)
        lr.append(get_lr(optimizer))
        scheduler.step()
        if engine.state.iteration%(int(LOG_EVERY/BATCH_SIZE)) == 0:
            print(get_time(), 'Event %d completed'%(engine.state.iteration*BATCH_SIZE))
    lr_trainer.add_event_handler(Events.ITERATION_COMPLETED, log_lr, lr, loss_vals, optimizer, scheduler, n_steps)
    
    print(get_time(), 'LR-scan begun.')
    lr_trainer.run(train_generator, max_epochs=n_epochs)
    print(get_time(), 'LR-scan finished!')

    return lr, loss_vals

def evaluate_model(model_dir, wandb_ID=None, predict=True):
    """Predicts on the dataset and makes performance plots induced by the model_dir. If wanted, the results are logged to W&B.
    
    Arguments:
        model_dir {str} -- Full or partial path to a trained model
    
    Keyword Arguments:
        wandb_ID {str} -- The unique W&B-ID of the experiment. If None, no logging is performed. (default: {None})
    """
    
    # ======================================================================== 
    # SAVE OPERATION PLOTS
    # ======================================================================== 
    if wandb_ID is not None:
        hyper_pars, data_pars, arch_pars, meta_pars = load_model_pars(model_dir)
        WANDB_DIR = get_project_root()+'/models'
        PROJECT = meta_pars['project']
        wandb.init(resume=True, id=wandb_ID, dir=WANDB_DIR, project=PROJECT)
    print(model_dir)
    log_operation_plots(model_dir, wandb_ID=wandb_ID)

    # ======================================================================== 
    # PREDICT USING BEST MODEL
    # ========================================================================
     
    if predict:
        hyper_pars, data_pars, arch_pars, meta_pars = load_model_pars(model_dir)

        if data_pars['dataloader'] == 'PickleLoader' or data_pars['dataloader'] == 'SqliteLoader' :
            calc_predictions_pickle(model_dir, wandb_ID=wandb_ID)
        else:
            calc_predictions(model_dir, wandb_ID=wandb_ID)
    
    # ======================================================================== #
    # REPORT PERFORMANCE
    # ======================================================================== #

    log_performance_plots(model_dir, wandb_ID=wandb_ID) 
    summarize_model_performance(model_dir, wandb_ID=wandb_ID)

    if wandb_ID is not None:
        wandb.log()
        wandb.join()
    
    # Update the meta_pars-file
    with open(model_dir+'/meta_pars.json') as json_file:
        meta_pars = json.load(json_file)
    meta_pars['status'] = 'Finished'
    with open(model_dir+'/meta_pars.json', 'w') as fp:
        json.dump(meta_pars, fp)

    # Close all open figures
    plt.close('all')

def explore_lr(hyper_pars, data_pars, arch_pars, meta_pars, save=True):
    """Calculates loss as a funciton of learning rate in a given interval and saves the graph and the dictionaries used to generate the plot. Used to choose lr-schedule.
    
    Arguments:
        hyper_pars {dict} -- Dictionary containing hyperparameters for the model.
        data_pars {dict} -- Dictionary containing datapath and relevant data parameters.
        arch_pars {dict} -- Dictionary containing the keywords required to build the model architecture
        meta_pars {dict} -- Dictionary containing metaparameters for the model such as regression-tag.
    
    Keyword Arguments:
        save {bool} -- Whether to save plots and dicts or not. (default: {True})
    """    
    
    # ======================================================================== 
    # SETUP AND UNPACK
    # ======================================================================== 
    
    print(get_time(), 'LEARNING RATE FINDER INITIATED')
    BATCH_SIZE = hyper_pars['batch_size']
    print('Loading data...')
    train_set = load_data(
        hyper_pars, 
        data_pars, 
        arch_pars, 
        meta_pars, 
        'train', 
        db_path=PATH_TRAIN_DB
        )
    N_TRAIN = get_set_length(train_set)
    n_epochs = hyper_pars['lr_finder']['n_epochs'] 
    start_lr = hyper_pars['lr_finder']['start_lr']
    end_lr = hyper_pars['lr_finder']['end_lr'] 
    hyper_pars['optimizer']['lr'] = start_lr
    gpus = meta_pars['gpu']

    # ======================================================================== 
    # MAKE LR SCAN
    # ======================================================================== 

    # num_workers choice based on gut feeling - has to be high enough to not be a bottleneck
    dataloader_params_train = get_dataloader_params(BATCH_SIZE, num_workers=8, shuffle=True, dataloader=data_pars['dataloader'])

    # Initialize model and log it - use GPU if available
    model, optimizer, device, _ = initiate_model_and_optimizer(None, hyper_pars, data_pars, arch_pars, meta_pars, n_train=N_TRAIN)

    loss = get_loss_func(
        arch_pars['loss_func'], 
        arch_pars.get('loss_func_weights', None),
        device=device
    )

    # Setup generators - make a generator for training, validation on trainset and validation on test set
    collate_fn = get_collate_fn(data_pars)
    train_generator = data.DataLoader(train_set, **dataloader_params_train, collate_fn=collate_fn)#, pin_memory=True)
    # Use IGNITE to train
    pretrain_hyper_pars = hyper_pars['optimizer'].copy()
    pretrain_hyper_pars['lr'] = start_lr
    lr, loss_vals = calc_lr_vs_loss(model, optimizer, loss, train_generator, BATCH_SIZE, N_TRAIN, gpus, n_epochs=n_epochs, start_lr=pretrain_hyper_pars['lr'], end_lr=end_lr)

    # ======================================================================== 
    # SAVE RELEVANT THINGS
    # ========================================================================
    
    data_dir = data_pars['data_dir'] # WHere to load data from
    lr_dir = make_lr_dir(data_dir, meta_pars['project'], BATCH_SIZE)
    _ = make_plot({'x': [lr], 'y': [loss_vals], 'xscale': 'log', 'savefig': lr_dir+'/lr_vs_loss.png', 'xlabel': 'Learning Rate', 'ylabel': 'Loss'})
    pickle.dump(lr, open(lr_dir+'/lr.pickle', 'wb'))
    pickle.dump(loss_vals, open(lr_dir+'/loss_vals.pickle', 'wb'))

    print('')
    print(get_time(), 'LR-finder saved at')
    print(lr_dir)

    with open(lr_dir+'/hyper_pars.json', 'w') as fp:
        json.dump(hyper_pars, fp)
    with open(lr_dir+'/data_pars.json', 'w') as fp:
        json.dump(data_pars, fp)
    with open(lr_dir+'/architecture_pars.json', 'w') as fp:
        json.dump(arch_pars, fp)

def initiate_model_and_optimizer(save_dir, hyper_pars, data_pars, arch_pars, meta_pars, n_train=None):

    gpus =  meta_pars['gpu']
    device = get_device(gpus[0])
    model = MakeModel(arch_pars, device)
    lr_scheduler = None

    # If several GPU's are available, use them all!
    print('')
    n_devices = torch.cuda.device_count()
    meta_pars['n_devices'] = n_devices
    if n_devices > 1 and len(gpus) > 1:
        model = torch.nn.DataParallel(model, device_ids=None, output_device=None, dim=0)
        print('Used devices:')
        for device_id in range(n_devices):
            name = torch.cuda.get_device_name(device=get_device(device_id))
            print(name)
    else:
        print('Used device:', get_device(gpus[0]))

    # Makes model parameters to float precision
    model = model.float()

    if hyper_pars['lr_schedule']['lr_scheduler'] == 'CyclicLR':
        hyper_pars['optimizer']['lr'] = hyper_pars['lr_schedule']['base_lr']
    
    # If training is to be continued on a pretrained model, load its parameters
    if meta_pars['objective'] == 'continue_training':
        checkpoint_path = get_project_root() + meta_pars['pretrained_path'] + '/backup.pth'

        checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
        iterations_completed = checkpoint['iterations_completed']
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)

        optimizer = get_optimizer(model.parameters(), hyper_pars['optimizer'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        lr_scheduler = get_lr_scheduler(hyper_pars, optimizer, hyper_pars['batch_size'], n_train, iterations_completed=iterations_completed)

        print(get_time(), 'PRETRAINED MODEL LOADED!')

    # If a run crashed and is reinitiated, load the state prior to crashing; model pars, optmizer pars and lr-schedule pars
    elif meta_pars['objective'] == 'continue_crashed':
        checkpoint_path = get_project_root() + get_path_from_root(meta_pars['crashed_path']) + '/backup.pth'

        checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
        iterations_completed = checkpoint['iterations_completed']
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)

        optimizer = get_optimizer(model.parameters(), hyper_pars['optimizer'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        lr_scheduler = get_lr_scheduler(hyper_pars, optimizer, hyper_pars['batch_size'], n_train, iterations_completed=iterations_completed)

        print(get_time(), 'CONTINUING TRAINING A CRASHED RUN!')

    elif meta_pars['objective'] == 'train_new':
        model = model.to(device)
        optimizer = get_optimizer(model.parameters(), hyper_pars['optimizer'])
        lr_scheduler = get_lr_scheduler(hyper_pars, optimizer, hyper_pars['batch_size'], n_train)
    
    elif meta_pars['objective'] == 'explore_lr':
        model = model.to(device)
        optimizer = get_optimizer(model.parameters(), hyper_pars['optimizer'])
        lr_scheduler = None
        
    else:
        raise ValueError('Unknown objective set!')

    print('')   
    print('Model being trained:')
    print(model)
    print('')

    return model, optimizer, device, lr_scheduler

def calc_predictions_old(save_dir, wandb_ID=None):
    '''Predicts target-variables from a trained model and calculates desired functions of the target-variables. Predicts one file at a time.
    '''
    hyper_pars, data_pars, arch_pars, meta_pars = load_model_pars(save_dir)
    particle_code = get_particle_code(data_pars['particle'])
    device = get_device(gpus[0])

    model_dir = save_dir+'/checkpoints'
    
    best_pars = find_best_model_pars(model_dir)
    n_devices = meta_pars.get('n_devices', 0)
    model = MakeModel(arch_pars, device)
    # If several GPU's have been used during training, wrap it in dataparalelle
    if n_devices > 1:
        model = torch.nn.DataParallel(model, device_ids=None, output_device=None, dim=0)
    model.load_state_dict(torch.load(best_pars, map_location=torch.device(device)))
    model = model.to(device)
    model = model.float()

    # Loop over files
    print(get_time(), 'Prediction begun.')
    pred_filename = str(best_pars).split('.pth')[0].split('/')[-1]
    pred_full_address = save_dir+'/data/predict'+pred_filename+'.h5'
    i_file = 0
    n_predicted = 0
    n_predictions_wanted = data_pars.get('n_predictions_wanted', np.inf)
    
    # Create list of files to predict on - either load it from model-directory if specific files are to be predicted on or iterate over the data directory
    path = get_project_root() + data_pars['data_dir']
    try:
        with open(save_dir+'/val_files.pickle', 'rb') as f:
            file_list = pickle.load(f)
        file_list = sorted([get_project_root()+file for file in file_list])
        USE_WHOLE_FILE = True
    except FileNotFoundError: 
        file_list = sorted([str(file) for file in Path(path).iterdir() if file.suffix == '.h5' and confirm_particle_type(particle_code, file)])
        USE_WHOLE_FILE = False

    N_FILES = len(file_list)
    with h5.File(pred_full_address, 'w') as f:
        
        for file in file_list:
            
            # Do not predict more than wanted - takes up space aswell...
            if n_predicted > n_predictions_wanted:
                break

            i_file += 1
            i_str = str(i_file)
             # TODO: Is always 0.00, since n_predicitons_wanted is np.inf.. It should be changed to len(val_set)
            print('Progress: %.0f %%. Predicting on %s'%(100*n_predicted/n_predictions_wanted, get_path_from_root(str(file))))
            
            # Extract validation data
            val_set = load_predictions(data_pars, meta_pars, 'val', file, use_whole_file=USE_WHOLE_FILE)
            predictions = {key: [] for key in val_set.targets}
            truths = {key: [] for key in val_set.targets}

            error_from_preds = {}

            # Predict 
            VAL_BATCH_SIZE = data_pars.get('val_batch_size', 256) # ! Predefined size !
            dataloader_params = {'batch_size': VAL_BATCH_SIZE, 'shuffle': False, 'num_workers': 8}

            # Setup generators
            collate_fn = get_collate_fn(data_pars)
            indices = sort_indices(val_set, data_pars, dataloader_params=dataloader_params)
            val_generator = data.DataLoader(val_set, **dataloader_params, collate_fn=collate_fn)

            # Use IGNITE to predict
            def log_prediction(engine):
                pred, truth = engine.state.output[0], engine.state.output[1]
                
                # give list of functions to calculate
                for i_batch in range(pred.shape[0]):
                    for i_key, key in enumerate(predictions):
                        predictions[key].append(pred[i_batch, i_key])
                        truths[key].append(truth[i_batch, i_key])
            
            evaluator_val = create_supervised_evaluator(model, device=device)
            evaluator_val.add_event_handler(Events.ITERATION_COMPLETED, log_prediction)
            evaluator_val.run(val_generator)

            # Sort w.r.t. index before saving
            for key, values in predictions.items():
                _, sorted_vals = sort_pairs(indices, values)
                predictions[key] = sorted_vals
            # Sort w.r.t. index before saving
            for key, values in truths.items():
                _, sorted_vals = sort_pairs(indices, values)
                truths[key] = sorted_vals
            indices = sorted(indices)

            # Run predictions through desired functions - transform back to 'true' values, if transformed
            predictions_transformed = inverse_transform(predictions, save_dir)
            truths_transformed = inverse_transform(truths, save_dir)

            eval_functions = get_eval_functions(meta_pars)
            for func in eval_functions:
                error_from_preds[func.__name__] = func(predictions_transformed, truths_transformed)

            # Save predictions
            # name = str(file).split('.')[-2].split('/')[-1]
            name = Path(file).stem
            grp = f.create_group(name)
            grp.create_dataset('index', data=np.array(indices))
            
            for key, pred in predictions.items():
                grp.create_dataset(key, data=np.array([x.cpu().numpy() for x in pred]))
            for key, pred in error_from_preds.items():
                grp.create_dataset(key, data=np.array([x.cpu().numpy() for x in pred]))
            
            n_predicted += len(val_set)
                
        print(get_time(), 'Predictions finished!')

def calc_predictions_pickle(
    save_dir, 
    wandb_ID=None, 
    ensemble_creation=False, 
    n_predictions_wanted=-1,
    db_path=PATH_VAL_DB
    ):
    '''Predicts target-variables from a trained model and calculates desired functions of the target-variables. Predicts one file at a time.
    '''

    # Load the best model 
    hyper_pars, data_pars, arch_pars, meta_pars = load_model_pars(save_dir)
    model = load_best_model(save_dir)

    # Allow for possibility of setting number of wanted predictions
    if n_predictions_wanted != -1:
        data_pars['n_predictions_wanted'] = n_predictions_wanted
    LOG_EVERY = int(meta_pars.get('log_every', 200000)/4) 
    VAL_BATCH_SIZE = data_pars.get('val_batch_size', 256) # ! Predefined size !
    gpus = meta_pars['gpu']
    device = get_device(gpus[0])

    # Setup dataloader and generator - num_workers choice based on gut feeling - has to be high enough to not be a bottleneck
    dataloader_params_eval = get_dataloader_params(VAL_BATCH_SIZE, num_workers=8, shuffle=False, dataloader=data_pars['dataloader'])
    val_set = load_data(
        hyper_pars, data_pars, arch_pars, meta_pars, 'predict', db_path=db_path
        )
    collate_fn = get_collate_fn(data_pars)
    loss = get_loss_func(
        arch_pars['loss_func'], 
        arch_pars.get('loss_func_weights', None),
        device=device
    )
    val_generator = data.DataLoader(val_set, **dataloader_params_eval, collate_fn=collate_fn)#, pin_memory=True)
    N_VAL = get_set_length(val_set)
    
    # Run evaluator!
    predictions, truths, indices, loss_vals = run_pickle_evaluator(
        model, 
        val_generator, 
        val_set.targets, 
        gpus, 
        LOG_EVERY=LOG_EVERY, 
        VAL_BATCH_SIZE=VAL_BATCH_SIZE, 
        N_VAL=N_VAL, 
        loss_func=loss)
    
    # Little hack - if we are predicting for ensemble, just return now
    if ensemble_creation:
        return predictions, indices
    
    # Run predictions through desired functions - transform back to 'true' values, if transformed
    predictions_transformed = inverse_transform(predictions, save_dir)
    truths_transformed = inverse_transform(truths, save_dir)

    eval_functions = get_eval_functions(meta_pars)
    error_from_preds = {}
    for func in eval_functions:
        error_from_preds[func.__name__] = func(predictions_transformed, truths_transformed)

    # Save predictions in h5-file.
    pred_full_address = save_dir+'/data/predictions.h5'
    
    print(get_time(), 'Saving predictions...')
    with h5.File(pred_full_address, 'w') as f:
        # We have to save as an integer
        f.create_dataset('index', data=np.array(indices))
        f.create_dataset('loss', data=np.array(loss_vals))
        for key, pred in predictions.items():
            f.create_dataset(key, data=pred)#np.array([x.cpu().numpy() for x in pred]))
        for key, pred in error_from_preds.items():
            f.create_dataset(key, data=pred)#np.array([x.cpu().numpy() for x in pred]))
    print(get_time(), 'Predictions saved!')

    # save_for_mads(predictions_transformed, indices, save_dir)

def calc_raw_predictions(model_dir, n_predictions_wanted=np.inf, db_path=None):

    if db_path == None:
        raise ValueError('A DB must be supplied!')

    # ======================================================================== 
    # PREDICT USING BEST MODEL
    # ========================================================================
    predictions, indices = calc_predictions_pickle(
        model_dir, 
        ensemble_creation=True, 
        n_predictions_wanted=n_predictions_wanted,
        db_path=db_path
        )
    return predictions, indices 

def save_for_mads(d, indices, save_dir):
    # Create DB for Mads
    print('')
    print(get_time(), 'Creating DB for MADS!')
    import pandas as pd
    keys = [key for key in d]
    mads_keys = get_mads_keys(keys)
    d = convert_keys(d, keys, mads_keys)
    d.update({'event_no': np.array(indices)})
    df = pd.DataFrame.from_dict(d)
    df.sort_values(by='event_no', inplace=True)
    with sqlite3.connect(save_dir+'/data/mads.db') as con:
        df.to_sql('truth', con=con, if_exists='replace')
    print(get_time(), 'Creation finished!')
    
def pickle_evaluator(model, device, non_blocking=False):
    """Custom evaluator. Prepares an Ignite Engine for inference specifically for evaluation with the PickleLoader as dataloader
    
    Arguments:
        model {torch.Module} -- The torch model ready for inference
        device {str} -- Used device - GPU or CPU
    
    Keyword Arguments:
        non_blocking {bool} -- Something used by cuda, not sure what it is, but Ignite defaults to false (default: {False})
    
    Returns:
        Engine -- an evaluator
    """    
    def _prepare_batch(batch, device=None, non_blocking=False):
        """Prepare batch for evaluation: pass to a device with options.
        """
        x, y = batch
        seq, lens, scalars, true_indices = x
        extracted = (seq, lens, scalars)
        return (convert_tensor(extracted, device=device, non_blocking=non_blocking),
                convert_tensor(y, device=device, non_blocking=non_blocking), true_indices)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            x, y_target, indices = _prepare_batch(batch, device=device, non_blocking=non_blocking)
            y_pred = model(x)
            return (y_pred, y_target, indices)

    engine = Engine(_inference)
    
    return engine    

def run_experiment(file, log=True, debug_mode=False, gpu_id=None):
    """Runs the experiment defined by file and deletes file.
    
    Arguments:
        file {str} -- Absolute path to experiment-file (a JSON-file).
    
    Keyword Arguments:
        log {bool} -- Whether or not to log plots, performance etc. locally and to W&B (default: {True})
    """

    with open(file) as json_file:
        dicts = json.load(json_file)
        hyper_pars = dicts['hyper_pars']
        data_pars = dicts['data_pars']
        arch_pars = dicts['arch_pars']
        meta_pars = dicts['meta_pars']

    if gpu_id:
        # Force an error - not implemented yet
        print(meta_pars)
        a+=1

    # Delete file
    Path(file).unlink()
    
    # Only scan new experiments
    scan = meta_pars.get('lr_scan', False)
    if meta_pars['objective'] == 'explore_lr':
        explore_lr(hyper_pars, data_pars, arch_pars, meta_pars)
    else:
        model_dir, wandb_ID = train_model(hyper_pars, data_pars, arch_pars, meta_pars, scan_lr_before_train=scan, log=log, debug_mode=debug_mode)
        if log:
            evaluate_model(model_dir, wandb_ID=wandb_ID)

def run_experiments(log=True, newest_first=False):
    """Loops over the experiment-defining files in ~/CubeML/experiments/ and runs each one of them using run_experiment. Continously checks if new experiments have been added. 
    
    Keyword Arguments:
        log {bool} -- Whether to log plots, performance etc. locally and to W&B (default: {True})
    """

    exp_dir = get_project_root() + '/experiments'
    exps = sorted(Path(exp_dir).glob('*.json'), reverse=newest_first)
    n_exps = len([str(exp) for exp in Path(exp_dir).glob('*.json')])
    
    # ! Someone online set to add next line to ensure CUDA works...
    # multiprocessing.set_start_method('spawn')
    while n_exps>0:
        
        for exp in exps:
            run_experiment(exp, log=log)
        
        exp_dir = get_project_root() + '/experiments/'
        exps = Path(exp_dir).glob('*.json')
        n_exps = len([str(exp) for exp in Path(exp_dir).glob('*.json')])

def run_pickle_evaluator(model, val_generator, targets, gpus, LOG_EVERY=50000, VAL_BATCH_SIZE=1028, N_VAL=np.inf, loss_func=None):
    """Runs inference with a trained model over a dataset.
    
    Arguments:
        model {torch.nn.Module} -- Trained and loaded model
        val_generator {torch.utils.dataloader} -- Dataloader
        targets {list} -- Prediction targets.
    
    Keyword Arguments:
        LOG_EVERY {int} -- How often to print for sanity (default: {50000})
        VAL_BATCH_SIZE {int} -- Batchsize to be used during inference (default: {1028})
        N_VAL {int} -- How many predictions to make in dataset.Accuracy (default: {np.inf})
    
    Returns:
        dict, dict, list -- Predictions, truths and indices in dataset
    """    

    # Use IGNITE to predict
    predictions = {key: [] for key in targets}
    truths = {key: [] for key in targets}
    
    # If a probabilistic regression is performed, make sure to save 
    # standard deviations aswell 
    # If a probabilistic regression is performed, make sure to save 
    # standard deviations aswell 
    loss_name = type(loss_func).__name__
    PROBABILISTIC_REGRESSION = is_probabilistic(loss_name)
    CLASSIFICATION = is_classification(loss_name)

    if PROBABILISTIC_REGRESSION:
        sigma_keys = ['sigma_'+key for key in targets]
        d = {key: [] for key in sigma_keys}
        predictions.update(d)
    else:
        sigma_keys = None

    loss_vals = []
    indices_PadSequence_sorted = []
    device = get_device(gpus[0])
    n_targets = len(targets)
    
    # The handler saving predictions
    def log_prediction(engine):

        # The indices returned by the Ignite Engine defined in pickle_evaluator sorts the indices for us        
        pred, target, indices = engine.state.output
        truth = target[0]
        weights = target[1]

        # Save loss values aswell - we call .item() to detach it from gradient graph
        loss = loss_func(pred, target, predict=True)
        loss_vals.extend([loss_val.item() for loss_val in loss])
        indices_PadSequence_sorted.extend(indices)

        # send through softmax to convert to probabilities - softmax is done in loss function, not by model.
        if CLASSIFICATION:
            pred = torch.nn.Softmax(dim=-1)(pred)
        for i_batch in range(pred.shape[0]):
            for i_key, key in enumerate(targets):
                predictions[key].append(pred[i_batch, i_key].item())
                truths[key].append(truth[i_batch, i_key].item())
            
            # If probabilistic inference, remember to save sigmas aswell
            if PROBABILISTIC_REGRESSION:
                for i_key, key in zip(range(n_targets, 2*n_targets), sigma_keys):
                    predictions[key].append(pred[i_batch, i_key].item())

        # Log for sanity...
        if engine.state.iteration%(max(1, int(LOG_EVERY/VAL_BATCH_SIZE))) == 0:
            n_predicted = engine.state.iteration*VAL_BATCH_SIZE
            print(get_time(), 'Progress %.0f %%: Predicted %d of %d'%(100*n_predicted/N_VAL, n_predicted, N_VAL)) 

    # Start predicting!
    print(get_time(), 'Prediction begun.')
    evaluator_val = pickle_evaluator(model, device)
    evaluator_val.add_event_handler(Events.ITERATION_COMPLETED, log_prediction)
    evaluator_val.run(val_generator)
    print(get_time(), 'Prediction finished!')

    _, loss_vals_sorted = sort_pairs(indices_PadSequence_sorted, loss_vals)
    for key, values in predictions.items():
        _, sorted_vals = sort_pairs(indices_PadSequence_sorted, values)
        predictions[key] = sorted_vals
        
    # Sort w.r.t. index before saving
    for key, values in truths.items():
        _, sorted_vals = sort_pairs(indices_PadSequence_sorted, values)
        truths[key] = sorted_vals
    indices = sorted(indices_PadSequence_sorted)

    return predictions, truths, indices, loss_vals_sorted

def train(
    save_dir, 
    hyper_pars, 
    data_pars, 
    arch_pars, 
    meta_pars, 
    earlystopping=True, 
    scan_lr_before_train=False, 
    wandb_ID=None, 
    log=True, 
    debug_mode=False
    ):
    """Main training script. Takes experiment-defining dictionaries as input and trains the model induced by them.
    
    Arguments:
        save_dir {str} -- Absolute path to the model's diretory
        hyper_pars {dict} -- Dictionary containing hyperparameters for the model.
        data_pars {dict} -- Dictionary containing datapath and relevant data parameters.
        arch_pars {dict} -- Dictionary containing the keywords required to build the model architecture
        meta_pars {dict} -- Dictionary containing metaparameters for the model such as regression-tag.
    
    Keyword Arguments:
        earlystopping {bool} -- Whether or not to use early stopping (default: {True})
        scan_lr_before_train {bool} -- Whether or not to perform a learning rate scan before training. (default: {False})
        wandb_ID {str} -- If supplied along with log=True, the training is logged to W&B (default: {None})
        log {bool} -- Whether or not to log locally and to W&B (default: {True})
    
    Raises:
        ValueError: If unknown parameters are given in the model-defining dictionaries (or if required keywords are missing!)
    
    Returns:
        None
    """  
      
    data_pars['val_batch_size'] = data_pars.get('val_batch_size', 256) # ! 256 chosen as a default parameter
    BATCH_SIZE = hyper_pars['batch_size']
    VAL_BATCH_SIZE = data_pars['val_batch_size']
    MAX_EPOCHS = hyper_pars['max_epochs']
    EARLY_STOP_PATIENCE = hyper_pars['early_stop_patience']
    LOG_EVERY = meta_pars.get('log_every', 200000) # ! 200000 chosen as a default parameter

    # Only calculate train error on a fraction of the training data - a fraction equal to val. frac.
    data_pars_copy = data_pars.copy()
    hyper_pars_copy = hyper_pars.copy()
    data_pars_copy['n_train_events_wanted'] = data_pars.get('n_val_events_wanted', np.inf)
    hyper_pars_copy['batch_size'] = data_pars['val_batch_size']
    
    print(get_time(), 'Loading data...')
    # We split in each file (after its been shuffled..)
    # Now load data
    train_set = load_data(
        hyper_pars, 
        data_pars, 
        arch_pars, 
        meta_pars, 
        'train', 
        debug_mode=debug_mode, 
        db_path=PATH_TRAIN_DB
        )
    trainerr_set = load_data(
        hyper_pars_copy, 
        data_pars_copy, 
        arch_pars, 
        meta_pars, 
        'train', 
        debug_mode=debug_mode, 
        db_path=PATH_TRAIN_DB
        )
    val_set = load_data(
        hyper_pars, 
        data_pars, 
        arch_pars, 
        meta_pars, 
        'val', 
        debug_mode=debug_mode, 
        db_path=PATH_VAL_DB
        )

    N_TRAIN = get_set_length(train_set)
    N_VAL = get_set_length(val_set)
    MAX_ITERATIONS = MAX_EPOCHS*N_TRAIN//BATCH_SIZE
    # Used for some lr-schedulers, so just add it.
    hyper_pars['lr_schedule']['train_set_size'] = N_TRAIN

    if log:
        wandb.config.update({'Trainset size': N_TRAIN})
        wandb.config.update({'Val. set size': N_VAL})
    print(get_time(), 'Data loaded!')
    print('\nTrain set size: %d'%(N_TRAIN))
    print('Val. set size: %d'%(N_VAL))
    
    # ====================================================================== #
    # SETUP TRAINING
    # ====================================================================== #

    # num_workers choice based on gut feeling - 
    # has to be high enough to not be a bottleneck
    n_workers = meta_pars['n_workers']
    dataloader_params_train = get_dataloader_params(BATCH_SIZE, num_workers=n_workers, shuffle=True, dataloader=data_pars['dataloader'])
    dataloader_params_eval = get_dataloader_params(VAL_BATCH_SIZE, num_workers=n_workers, shuffle=False, dataloader=data_pars['dataloader'])
    dataloader_params_trainerr = get_dataloader_params(VAL_BATCH_SIZE, num_workers=n_workers, shuffle=False, dataloader=data_pars['dataloader'])
    
    # Initialize model and log it - use GPU if available
    model, optimizer, device, lr_scheduler = initiate_model_and_optimizer(save_dir, hyper_pars, data_pars, arch_pars, meta_pars, n_train=N_TRAIN)

    if log:
        with open(save_dir+'/model_arch.yml', 'w') as f:
            print(model, file=f)
        wandb.save(save_dir+'/model_arch.yml')
        wandb.config.update({'Model parameters': get_n_parameters(model)})
    print('N_PARAMETERS:', get_n_parameters(model))
    
    # Get type of scheduler, since different schedulers need different kinds of updating.
    type_lr_scheduler = type(lr_scheduler)
    loss = get_loss_func(
        arch_pars['loss_func'], 
        arch_pars.get('loss_func_weights', None),
        device=device
    )

    # Setup generators - make a generator for training, validation on trainset and validation on test set
    collate_fn = get_collate_fn(data_pars)
    train_generator = data.DataLoader(train_set, **dataloader_params_train, collate_fn=collate_fn)#, pin_memory=True)
    val_generator = data.DataLoader(val_set, **dataloader_params_eval, collate_fn=collate_fn)#, pin_memory=True)
    trainerr_generator = data.DataLoader(trainerr_set, **dataloader_params_trainerr, collate_fn=collate_fn)#, pin_memory=True)
    
    # Use IGNITE to train
    trainer = create_supervised_trainer(model, optimizer, loss, device=device)
    evaluator_val = create_supervised_evaluator(model, metrics={'custom_loss': Loss(loss)}, device=device)
    evaluator_train = create_supervised_evaluator(model, metrics={'custom_loss': Loss(loss)}, device=device)

    # ========================================================================
    # SETUP SAVING OF IMPROVED MODELS
    # ========================================================================  
    
    def custom_score_function(engine):
        loss = engine.state.metrics['custom_loss']
        return -loss
    
    if log:
        name = ''
        checkpointer = ModelCheckpoint(dirname=save_dir+'/checkpoints', filename_prefix=name, create_dir=True, save_as_state_dict=True, score_function=custom_score_function, score_name='Loss', n_saved=2, require_empty=False)
        
        # Add handler to evaluator
        checkpointer_dict = {'model': model}
        evaluator_val.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, checkpointer_dict)

    # ====================================================================== #
    # SETUP EARLY STOPPING
    # ====================================================================== #

    # patience = how long to wait before stopping according to score_func, trainer = which engine to stop.
    if earlystopping:
        print('Early stopping activated!')
        early_stop_handler = EarlyStopping(patience=EARLY_STOP_PATIENCE,
                                        score_function=custom_score_function, 
                                        trainer=trainer)
        evaluator_val.add_event_handler(Events.EPOCH_COMPLETED, early_stop_handler)


    # # ====================================================================== #
    # # DO LEARNING RATE SCAN
    # # ====================================================================== #

    # if scan_lr_before_train:
    #     pretrain_hyper_pars = hyper_pars['optimizer'].copy()
    #     pretrain_hyper_pars['lr'] = 0.000001

    #     lr_model = MakeModel(arch_pars, device)
    #     lr_model = lr_model.float()
    #     lr_model = lr_model.to(device)

    #     pretrain_optimizer = get_optimizer(lr_model.parameters(), pretrain_hyper_pars)
    #     pretrain_lr, pretrain_losses = calc_lr_vs_loss(lr_model, pretrain_optimizer, loss, train_generator, BATCH_SIZE, N_TRAIN, gpus, start_lr=pretrain_hyper_pars['lr'])

    #     vlines = []
    #     if 'base_lr' in hyper_pars['lr_schedule']:
    #         vlines.append(hyper_pars['lr_schedule']['base_lr'])
    #     if 'max_lr' in hyper_pars['lr_schedule']:
    #         vlines.append(hyper_pars['lr_schedule']['max_lr'])
        
    #     if log:
    #         img_address = save_dir+'/figures/pretrain_lr_vs_loss.png'
    #         _ = make_plot({'x': [pretrain_lr], 'y': [pretrain_losses], 'xscale': 'log', 'savefig': img_address, 'xlabel': 'Learning Rate', 'ylabel': 'Loss', 'axvline': vlines})
    #         pickle.dump(pretrain_lr, open(save_dir+'/pretrain_lr.pickle', 'wb'))
    #         pickle.dump(pretrain_losses, open(save_dir+'/pretrain_loss_vals.pickle', 'wb'))
    #         im = PIL.Image.open(img_address)
    #         wandb.log({'Pretrain LR-scan': wandb.Image(im, caption='Pretrain LR-scan')}, commit=False)

    # ====================================================================== #
    # SETUP LOGGING
    # ====================================================================== #   
    
    # If continuing training, get how many epochs completed
    ITERATIONS_COMPLETED = get_iterations_completed(meta_pars)

    # Print log
    def print_log(engine, set_name, metric_name):
        print("Events: {}/{} - {} {}: {:.2e}"
            .format((trainer.state.iteration+ITERATIONS_COMPLETED)*BATCH_SIZE, (MAX_ITERATIONS)*BATCH_SIZE, set_name, metric_name, engine.state.metrics[metric_name]))

    evaluator_train.add_event_handler(Events.COMPLETED, print_log, "train", 'custom_loss')
    evaluator_val.add_event_handler(Events.COMPLETED, print_log, "validation", 'custom_loss')

    # Log locally and to W&B
    if log:
        def log_metric(engine, set_name, metric_name, list_address):
            append_list_and_save(list_address, engine.state.metrics[metric_name])
            wandb.log({set_name+metric_name: engine.state.metrics[metric_name]}, step=(trainer.state.iteration+ITERATIONS_COMPLETED)*BATCH_SIZE)

        def log_lr(engine, set_name, optimizer, list_address):
            number = get_lr(optimizer)
            append_list_and_save(list_address, number)
            wandb.log({set_name: number}, step=(trainer.state.iteration+ITERATIONS_COMPLETED)*BATCH_SIZE)

        def log_epoch(engine, list_address):
            append_list_and_save(list_address, (trainer.state.iteration+ITERATIONS_COMPLETED)*BATCH_SIZE)

        evaluator_train.add_event_handler(Events.COMPLETED, log_metric, 'Graphs/train ', 'custom_loss', save_dir+'/data/train_error.pickle')
        evaluator_val.add_event_handler(Events.COMPLETED, log_metric, 'Graphs/val. ', 'custom_loss', save_dir+'/data/val_error.pickle')
        evaluator_train.add_event_handler(Events.COMPLETED, log_lr, 'Graphs/learning rate', optimizer, save_dir+'/data/lr.pickle')
        evaluator_val.add_event_handler(Events.COMPLETED, log_epoch, save_dir+'/data/epochs.pickle')

    # Time training and evaluation
    time_trainer = Timer(average=True)
    time_trainer.attach(trainer, resume=Events.ITERATION_STARTED, pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    time_evaluator = Timer(average=True)
    time_evaluator.attach(evaluator_val, resume=Events.ITERATION_STARTED, pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    # Call evaluator after each iteration - only evaluate after each LOG_EVERY events
    def evaluate(trainer):

        if (trainer.state.iteration+ITERATIONS_COMPLETED)%(int(LOG_EVERY/BATCH_SIZE)) == 0:
            print('')
            print(get_time(), 'Event %d completed'%((trainer.state.iteration+ITERATIONS_COMPLETED)*BATCH_SIZE))
            
            # # ? Log weights, biases and gradients in histograms - mostly for debugging?
            # if log:
            #     i_layer = 1
            #     step = trainer.state.epoch + ITERATIONS_COMPLETED
            #     for entry in model.mods: 
            #         if type(entry) == nn.modules.container.Sequential:
            #             for seq_entry in entry:
            #                 i_layer = log_weights_and_grads(i_layer, seq_entry, step)
                            
            #                 i_layer = log_weights_and_grads(i_layer, entry, step)
            
            # Run evaluation on train- and validation-sets
            evaluator_train.run(trainerr_generator)
            evaluator_val.run(val_generator)
            if data_pars['dataloader'] == 'FullBatchLoader':
                trainerr_set.make_batches()
                val_set.make_batches()
            elif data_pars['dataloader'] == 'PickleLoader':
                trainerr_set.shuffle_indices()
                val_set.shuffle_indices()
            elif data_pars['dataloader'] == 'SqliteLoader':
                trainerr_set.shuffle_indices()
                val_set.shuffle_indices()

            # Log maximum memory allocated and speed.
            if log:
                wandb.config.update({'Avg. Events/second (train)': BATCH_SIZE/time_trainer.value()}, allow_val_change=True)
                wandb.config.update({'Avg. Events/second (eval.)': VAL_BATCH_SIZE/time_evaluator.value()}, allow_val_change=True)
                if torch.cuda.is_available():
                    max_memory_allocated = torch.cuda.max_memory_allocated(device=device)/(1024*1024)
                    wandb.config.update({'Max memory allocated [MiB]': max_memory_allocated}, allow_val_change=True)

                # Save a backup after each epoch incase something crashes...
                backup = {'iterations_completed': trainer.state.iteration + ITERATIONS_COMPLETED,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                        }
                torch.save(backup, save_dir + '/backup.pth')

    trainer.add_event_handler(Events.ITERATION_COMPLETED, evaluate)
    trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())
    
    # ! Full batch-loaders has to be treated in a special way! See the class, it has to be shuffled every epoch
    if data_pars['dataloader'] == 'SqliteLoader':

        def shuffle_batches(engine):
            train_set.shuffle_indices()
        trainer.add_event_handler(Events.EPOCH_COMPLETED, shuffle_batches)

    

    # ======================================================================== #
    # SETUP LEARNING RATE SCHEDULER
    # ======================================================================== #
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
    
    elif type_lr_scheduler == torch.optim.lr_scheduler.OneCycleLR:
        
        def update_lr(engine, lr_scheduler):
            lr_scheduler.step()
        trainer.add_event_handler(Events.ITERATION_COMPLETED, update_lr, lr_scheduler)

    else:
        raise ValueError('Undefined lr_scheduler (%s) used for updating LR!'%(type_lr_scheduler))

    # ======================================================================== #
    # START TRAINING
    # ======================================================================== # 
           
    print('Training begun')
    train_epochs = MAX_EPOCHS-(ITERATIONS_COMPLETED*BATCH_SIZE//N_TRAIN)
    trainer.run(train_generator, max_epochs=train_epochs)
    print('\nTraining finished!')

def train_model(hyper_pars, data_pars, arch_pars, meta_pars, scan_lr_before_train=False, log=True, debug_mode=False):
    
    # ======================================================================== 
    # SETUP AND LOAD DATA
    # ======================================================================== 

    # If training is on a pretrained model, copy and update data- and hyperpars with potential new things
    if meta_pars['objective'] == 'continue_training':
        save_dir, hyper_pars, data_pars, arch_pars = update_model_pars(hyper_pars, data_pars, meta_pars)
        wandb_ID = save_dir.split('/')[-1]
    elif meta_pars['objective'] == 'continue_crashed':
        save_dir = get_project_root() + get_path_from_root(meta_pars['crashed_path'])
        hyper_pars, data_pars, arch_pars, new_meta_pars = load_model_pars(save_dir)
        new_meta_pars['objective'] = 'continue_crashed'
        new_meta_pars['crashed_path'] = meta_pars['crashed_path']
        meta_pars = new_meta_pars

    # The script expects a H5-file with a structure as shown at https://github.com/ehrhorn/CubeML
    data_dir = data_pars['data_dir'] # WHere to load data from
    file_keys = data_pars.get('file_keys', None) # which cleaning lvl and transform should be applied?
    group = meta_pars['group'] # under which dir to save?
    project = meta_pars['project']
    particle = data_pars.get('particle', 'any')

    if meta_pars['objective'] == 'train_new':
        if log:
            save_dir = make_model_dir(group, data_dir, file_keys, project, particle=particle)
            print('Model saved at', save_dir)
        else:
            save_dir = None
            wandb_ID = None
    
    # Save model parameters on W&B AND LOCALLY!
    # Shut down W&B first, if it is already running
    if log:
        wandb_ID = save_dir.split('/')[-1]
        WANDB_NAME = save_dir.split('/')[-1]
        MODEL_NAME = save_dir.split('/')[-1]
        WANDB_DIR = get_project_root()+'/models'
        n_seq_feats = len(data_pars['seq_feat'])

        wandb.init(project=meta_pars['project'], name=WANDB_NAME, tags=meta_pars['tags'], id=wandb_ID, reinit=True, dir=WANDB_DIR)
        wandb.config.update(hyper_pars, allow_val_change=True)
        wandb.config.update(data_pars, allow_val_change=True)
        wandb.config.update(arch_pars, allow_val_change=True)
        wandb.config.update({'n_seq_feats': n_seq_feats}, allow_val_change=True)

        with open(save_dir+'/hyper_pars.json', 'w') as fp:
            json.dump(hyper_pars, fp)
        
        with open(save_dir+'/data_pars.json', 'w') as fp:
            json.dump(data_pars, fp)
        
        with open(save_dir+'/architecture_pars.json', 'w') as fp:
            json.dump(arch_pars, fp)
        
        meta_pars['status'] = 'Failed'
        n_devices = len(meta_pars['gpu'])
        meta_pars['n_devices'] = n_devices
        with open(save_dir+'/meta_pars.json', 'w') as fp:
            json.dump(meta_pars, fp)

    else:
        print('Logging turned off.')

    train(save_dir, hyper_pars, data_pars, arch_pars, meta_pars, scan_lr_before_train=scan_lr_before_train, wandb_ID=wandb_ID, log=log, debug_mode=debug_mode)
    
    # Update the meta_pars-file and add .dvc-files to track the model in the wandb-dir and the models-dir
    if log:
        with open(save_dir+'/meta_pars.json') as json_file:
            meta_pars = json.load(json_file)
        meta_pars['status'] = 'Trained'
        with open(save_dir+'/meta_pars.json', 'w') as fp:
            json.dump(meta_pars, fp)

    return save_dir, wandb_ID
    
