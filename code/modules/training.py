
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

#Define a validation function
@torch.no_grad()
def validation_func(model, loss_func, device, val_loader):
   model.eval()
   val_loop_losses = []
  
   for x, y in val_loader:
       x, y = x.to(device), y.to(device)
      
       logits = model(x)    #logits has shape (B, T, vocab_size)
       val_loop_losses.append(loss_func(logits.view(-1, logits.shape[-1]), y.view(-1)).item())    #We reshape logits to be of shape (B*T, vocab_size) and targets to be of shape (B*T)
      
   return np.mean(val_loop_losses)
    
        
#Define a training loop function
def train(model, optimizer, scheduler, loss_func, device, train_loader, val_loader, validation_func, epochs, val_interval, smallest_val_loss, patience, checkpoint_filepath):
    train_losses = []
    val_losses = []
    patience_count = 0
    val_epoch_count = 0
    
    for epoch in tqdm(range(epochs)):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)      
            optimizer.zero_grad()

            logits = model(x)   #of shape (B, T, vocab_size)
            loss = loss_func(logits.view(-1, logits.shape[-1]), y.view(-1))   #Reshape to size (B*T, vocab_size) and (vocab_size), which is what CELoss is expecting
            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()


        if epoch % val_interval == 0:
            val_epoch_count += 1
            val_loss = validation_func(model, loss_func, device, val_loader)
            val_losses.append(val_loss)


            if val_loss < smallest_val_loss:
                patience_count = 0     
                smallest_val_loss = val_loss
                checkpoint_dict = {
                    "epoch" : epoch,
                    "model_state" : model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                    "scheduler_state" : scheduler.state_dict()
                }
                torch.save(checkpoint_dict, checkpoint_filepath)
            else:
                patience_count += 1
                if patience_count > patience:
                    return train_losses, val_losses, smallest_val_loss
                
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            
            print(f"Epoch: {epoch+1}/{epochs}, Training Loss: {round(loss.item(),3)}, Validation Loss: {round(val_loss,3)}")
      
        if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()

    return train_losses, val_losses, smallest_val_loss    


#Define a function to graph the losses
def graph_losses(losses, name, save_location): 
    plt.plot(np.arange(len(losses)), losses)
    plt.xlabel("Epochs")
    plt.ylabel(f"{name} Loss")
    plt.title(f"{name} Loss over Time")  
    plt.savefig(f"{save_location}/{name}_loss_{len(losses)}_Epochs.png")
    plt.clf()