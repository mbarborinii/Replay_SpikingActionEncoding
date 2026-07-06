import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader
from scipy.integrate import odeint

# ----------------------------
# 4. Training Loop
# ----------------------------
def interleave_pos_neg_v2(x):
    """
    This is just for positions
    """
    # Avoid in-place operations by using torch.cat instead of in-place assignment
    pos = F.relu(x)        # even columns: positives
    neg = F.relu(-x)       # odd  columns: abs negatives
    # Stack interleaved: [pos[0], neg[0], pos[1], neg[1], ...]
    out = torch.stack([pos, neg], dim=2).reshape(x.size(0), x.size(1) * 2)
    # out[:, :4] = out[:, :4] * 5
    
    return out

def interleave_pos_neg_v3(x):
    """
    This is for positions and velocities
    """
    out = torch.empty(x.size(0), x.size(1) * 4, device=x.device, dtype=x.dtype)
    out[:, 0::2] = F.relu(x)        # even columns: positives
    out[:, 1::2] = F.relu(-x)       # odd  columns: abs negatives
    return out


def delete(one, two, three=None):
    del one
    del two
    del three
    # del four
    # del five
    # torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()  # optional, helps with inter-process fragmentation
    
    
    
    
    
    
    
    
    
    
    
    

#############################################################################################
#############################################################################################
#############################################################################################
#############################################################################################    
#############################################################################################




def plot(losses, output, target, dyn_sys_population=None, decoder=None, figures_directory = None, date=None, trial=None, spike_or_rate=None, sequence_id=None, trial_id=None, rates_function = None, save = True):
    
    """
    This assumes you're working in the SPiking ActionEncoding directory
    """
    a=0
    output_and_target_len = len(output[-1][0][0])  # should be 4 for positions and velocities of 2 masses
    output = np.asarray(output)
    target = np.asarray(target)
    
    fig = plt.figure(figsize=(12, int(3*output_and_target_len)))
    
    colors1 = plt.cm.Blues(np.linspace(0, 1, output_and_target_len+2))
    colors2 = plt.cm.Reds(np.linspace(0, 1, output_and_target_len+2))
    
    ax1 = fig.add_subplot(int(output_and_target_len+1), 1, 1)
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Avg. loss")
    ax1.scatter(np.linspace(0,len(losses), len(losses)), losses)
    ax1.set_xlim(0, len(losses))
    ax1.grid()
    
    label_qualitative = ['position', 'velocity']#, 'acceleration']
    # TODO change labels
    # TODO check it works
    for i in range(output_and_target_len):
        # I am changing this due to the different backprop, and how output is shaped
        label = label_qualitative[i // 2]
        ax_name = f'ax{i+2}'
        
        # TODO change labels, now it is pos vel acc pos, should be pos pos vel vel etc etc 
        ax_name = fig.add_subplot(int(output_and_target_len+1), 1, i+2)
        ax_name.set_xlabel("Time")   
        ax_name.set_ylabel(f'{label} trajectory for mass {i%2+1}')
        ax_name.plot(np.linspace(0,len(output[-1, 0, :, i]), len(output[-1, 0, :, i])), output[-1, 0, :, i].squeeze(), label=f'Output_{label}_{i%2+1}', linewidth=2.5, color=colors1[i+2])
        ax_name.plot(np.linspace(0,len(target[-1, 0, :, i]), len(target[-1, 0, :, i])), target[-1, 0, :, i].squeeze(), label=f'Target_{label}_{i%2+1}', linewidth=2.5, color=colors2[i+2], linestyle='dashed')
        ax_name.grid()
        # ax_name.xaxis.tick_top()
        plt.legend()
        
    plt.tight_layout()
    fig.align_ylabels()

    
    if save:
        if figures_directory is not None:
            plt.savefig(f"{figures_directory}/mass_oscillation_training.png")  
        else:
            plt.savefig(f"{spike_or_rate}/{date}/{trial}/mass_oscillation_training_{date}_{trial}_{sequence_id}_{trial_id}.png")  
   
    if rates_function is not None:
        fig = plt.figure(figsize=(8, 8))
        plt.plot(rates_function[0][:][:][:].squeeze())
        if figures_directory is not None:
            plt.savefig(f"{figures_directory}/Neurons_rates_with_function_mass_oscillation_training.png")  
        else:
            plt.savefig(f"{spike_or_rate}/{date}/{trial}/Neurons_rates_with_function_mass_oscillation_training_{date}_{sequence_id}_{trial_id}.png") 
   
    # with open(f"{spike_or_rate}/{date}/{trial}/parameters_{date}_{trial}_{sequence_id}_{trial_id}.txt", 'w') as file:
    #     for key, value in params.items():
    #         file.write(f"{key}: {value}\n")
    
    if dyn_sys_population is not None:
        fig = plt.figure(figsize=(8, 8), layout='constrained')
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.set_xlabel("Out layer")
        ax1.set_ylabel("In layer")
        im = ax1.imshow(dyn_sys_population.ff_weights.detach().cpu().numpy(), cmap='coolwarm')
        ax1.set_title("feedforward weights (matrix)")
        fig.colorbar(im , ax=ax1)
        ax2 = fig.add_subplot(2, 1, 2)
        plt.hist(dyn_sys_population.ff_weights.detach().cpu().numpy().flatten(), bins=30, color='skyblue', edgecolor='black')
        ax2.set_xlabel("Weight values")
        ax2.set_ylabel("Quantity")
        ax2.set_title("feedforward weights (histogram)")
        if save:
            if figures_directory is not None:
                plt.savefig(f"{figures_directory}/feedforward_weights_mass_oscillation_training.png")       
            else:
                plt.savefig(f"{spike_or_rate}/{date}/{trial}/feedforward_weights_mass_oscillation_training_{date}_{trial}_{sequence_id}_{trial_id}.png")       
        
        fig = plt.figure(figsize=(10, 10), layout='constrained')
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.set_xlabel("Out layer")
        ax1.set_ylabel("In layer")
        im = ax1.imshow(dyn_sys_population.rec_weights[:,:,0].detach().cpu().numpy(), cmap='coolwarm')
        ax1.set_title("recurrent weights (matrix)")
        fig.colorbar(im , ax=ax1)
        ax2 = fig.add_subplot(2, 1, 2)
        plt.hist(dyn_sys_population.rec_weights.detach().cpu().numpy().flatten(), bins=30, color='skyblue', edgecolor='black')
        ax2.set_xlabel("Weight values")
        ax2.set_ylabel("Quantity")
        ax2.set_title("recurrent weights (histogram)")
        
        if save:
            if figures_directory is not None:
                plt.savefig(f"{figures_directory}/recurrent_weights_mass_oscillation_training.png")       
            else:
                plt.savefig(f"{spike_or_rate}/{date}/{trial}/recurrent_weights_mass_oscillation_training_{date}_{trial}_{sequence_id}_{trial_id}.png")       

        
        fig = plt.figure(figsize=(8, 8), layout='constrained')
        ax2 = fig.add_subplot(2, 1, 2)
        plt.hist(dyn_sys_population.a.detach().cpu().numpy().flatten(), bins=30, color='skyblue', edgecolor='black')
        ax2.set_xlabel("Weight values")
        ax2.set_ylabel("Quantity")
        ax2.set_title("a parameter (histogram)")
        if save:
            if figures_directory is not None:   
                plt.savefig(f"{figures_directory}/a_parameter_mass_oscillation_training.png")
            else:
                plt.savefig(f"{spike_or_rate}/{date}/{trial}/a_parameter_mass_oscillation_training_{date}_{trial}_{sequence_id}_{trial_id}.png")       
        
        
        fig = plt.figure(figsize=(8, 8), layout='constrained')
        ax2 = fig.add_subplot(2, 1, 2)
        plt.hist(dyn_sys_population.b.detach().cpu().numpy().flatten(), bins=30, color='skyblue', edgecolor='black')
        ax2.set_xlabel("Weight values")
        ax2.set_ylabel("Quantity")
        ax2.set_title("b parameter (histogram)")
        if save:
            if figures_directory is not None:   
                plt.savefig(f"{figures_directory}/b_parameter_mass_oscillation_training.png")
            else:
                plt.savefig(f"{spike_or_rate}/{date}/{trial}/b_parameter_mass_oscillation_training_{date}_{trial}_{sequence_id}_{trial_id}.png")       
        
        
    if decoder is not None:
        fig = plt.figure(figsize=(8, 8), layout='constrained')
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.set_ylabel("Out layer")
        ax1.set_xlabel("In layer")
        im = ax1.imshow(decoder.ff_weights.detach().cpu().numpy().T, cmap='coolwarm')
        ax1.set_title("readout weights (matrix)")
        fig.colorbar(im , ax=ax1)
        ax2 = fig.add_subplot(2, 1, 2)
        plt.hist(decoder.ff_weights.detach().cpu().numpy().flatten(), bins=30, color='skyblue', edgecolor='black')
        ax2.set_xlabel("Weight values")
        ax2.set_ylabel("Quantity")
        ax2.set_title("readout weights (histogram)")
        if save:
            if figures_directory is not None:
                plt.savefig(f"{figures_directory}/readout_weights_mass_oscillation_training.png")       
            else:
                plt.savefig(f"{spike_or_rate}/{date}/{trial}/readout_weights_mass_oscillation_training_{date}_{trial}_{sequence_id}_{trial_id}.png")       
        
    
    plt.close('all')
    
    return 



    