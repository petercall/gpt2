#Training hyperparameters
epochs = 150
val_interval = 1
smallest_val_loss = float("inf")
stop_patience = 6          #How many validation loops with no decrease in validation loss before the training stops

#LR_shceduler hyperparameters
scheduler_patience = 2      #How many validation loops with no decrease in validation loss before the learning rate is multiplied by factor
scheduler_factor = .3        #The factor that the learning rate gets multiplied by when it is not improving

#Checkpoint hyperparameters
checkpoint_storage_location = "../../checkpoints/pretrain-2.pth"

#Graph hyperparameters
graph_save_location = "../../outputs/graphs"

