import matplotlib.pyplot as plt

#Define a function to graph the losses
def graph_losses(losses, name, save_location): 
    plt.plot(np.arange(len(losses)), losses)
    plt.xlabel("Epochs")
    plt.ylabel(f"{name} Loss")
    plt.title(f"{name} Loss over Time")  
    plt.savefig(f"{save_location}/{name}_loss_{len(losses)}_Epochs.png")
    plt.clf()