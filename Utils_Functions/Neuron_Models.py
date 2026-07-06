import matplotlib.pyplot as plt
from networkx import sigma
import numpy as np
import torch
import torch.nn as nn
import mujoco 
import Utils_Functions.Utils as Utils

# NOTE: one can use device=input.device, dtype=input.dtype to infer the device and dtype from the input tensor
class SurrGradSpike(torch.autograd.Function):
    """
    Here we implement our spiking nonlinearity which also implements 
    the surrogate gradient. By subclassing torch.autograd.Function, 
    we will be able to use all of PyTorch's autograd functionality.
    Here we use the normalized negative part of a fast sigmoid 
    as this was done in Zenke & Ganguli (2018).
    """

    scale = 10
    threshold = 0

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we compute a step function of the input Tensor
        and return it. ctx is a context object that we use to stash information which 
        we need to later backpropagate our error signals. To achieve this we use the 
        ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > SurrGradSpike.threshold] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor we need to compute the 
        surrogate gradient of the loss with respect to the input. 
        Here we use the normalized negative part of a fast sigmoid 
        as this was done in Zenke & Ganguli (2018).
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input/(SurrGradSpike.scale*torch.abs(input)+1.0)**2
        return grad


class SLAYERGradSpike(torch.autograd.Function):
    """
    Surrogate gradient using the SLAYER kernel (Shrestha & Orchard, 2018),
    as implemented in the SE-adlif repository (IGITUGraz/SE-adlif).

    Forward pass: standard Heaviside step function (spike if input > threshold).
    Backward pass: exponentially decaying surrogate gradient:

        dL/dinput = grad_output * (c * alpha) / (2 * exp(|input| * alpha))

    Compared to the fast sigmoid surrogate (SurrGradSpike), this gradient
    decays *exponentially* away from threshold rather than polynomially.
    At equal peak height (alpha=5.0, c=0.4 gives peak=1.0), SLAYER maintains
    a substantially fatter gradient signal near threshold, which can accelerate
    learning of adaptation parameters but requires gradient clipping for stability.

    Class attributes (can be overridden before use):
        alpha (float): Controls the sharpness of the exponential decay.
                       Higher = narrower gradient. Default: 5.0
        c (float):     Scaling constant. Together with alpha, sets the peak
                       gradient magnitude to c*alpha/2. Default: 0.4
        threshold (float): Spike threshold. Default: 0.0
    """

    alpha = 4.0
    c = 0.4
    threshold = 0.0

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > SLAYERGradSpike.threshold] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # SLAYER kernel: c * alpha / (2 * exp(|x| * alpha))
        surrogate = (SLAYERGradSpike.c * SLAYERGradSpike.alpha /
                     (2.0 * torch.exp(input.abs() * SLAYERGradSpike.alpha)))
        return grad_output * surrogate


spike_fn = SLAYERGradSpike.apply




# if we want to get the things from torch nn module we write
# class Neuron(torch.nn.Module): 
# and in __init__ we put super().__init__(), I think it will be enough
class Neuron:
    
    def __init__(self, batch_size, nb_inputs, nb_neurons, beta, requires_grad, fwd_scale=0.1, device="cuda", dtype=torch.float):
        self.name = self.__class__.__name__
        self.nb_inputs = nb_inputs
        self.nb_neurons = nb_neurons
        self.beta = beta
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size
        self.fwd_scale = fwd_scale
        self.requires_grad = requires_grad
        self.shape_weights = (self.nb_inputs, self.nb_neurons)
        self.shape_states = (self.batch_size, self.nb_neurons)
                
    def init_parameter_copy(self, parameter, requires_grad):
        return torch.nn.Parameter(parameter.clone(), requires_grad=requires_grad)
    
    def init_weights_random_normal(self, shape, scale, neurons, requires_grad):
        # Assumes we use the normal distribution
        if shape is None:
            shape = self.shape_weights
        if scale is None:
            scale = self.fwd_scale
        if neurons is None:
            neurons = self.nb_inputs
        if requires_grad is None:
            requires_grad = self.requires_grad
        weights = torch.empty(shape, device=self.device, dtype=self.dtype, requires_grad=requires_grad)
        torch.nn.init.normal_(weights, mean=0.0, std=scale / np.sqrt(neurons)) 
        return weights

    def init_weights_random_unif(self, shape, param_scalar, requires_grad, min_value = 0.0):
        # Assumes we use the uniform distribution
        if shape is None:
            shape = self.shape_weights
        param = torch.empty(shape, device=self.device, dtype=self.dtype, requires_grad=requires_grad)
        torch.nn.init.uniform_(param, a=min_value, b=param_scalar) 
        return param
    
    def init_state(self):
        return torch.zeros(self.shape_states,device=self.device, dtype=self.dtype, requires_grad=True) # TODO Occhio?! It could lead to wrong stuff maybe?
    
    def reset_state(self,state_to_reset):
        return torch.zeros_like(state_to_reset).detach_().zero_()
    

    def state_dict(self) -> dict:
        return dict(self.__dict__)
    
    def detach_state(self):
        """Call this AFTER each outer time step."""
        if hasattr(self, 'mem'):
            self.mem = self.mem.detach()
        if hasattr(self, 'curr'):
            self.curr = self.curr.detach()
        if hasattr(self, 'rst'):
            self.rst = self.rst.detach()



class LI(Neuron):
    """
    Class to initialize and compute a feedforward layer of Leaky Integrator (LI) neurons.

    This class implements a feedforward layer of Leaky Integrator (LI) neurons, which accumulate
    input over time with a leaky (decaying) membrane potential. The layer supports computation
    of membrane potentials for each neuron at each time step, using a simple leaky integration
    model without spiking or synaptic currents.

    Attributes:
        nb_inputs (int): Number of input neurons.
        nb_neurons (int): Number of LI neurons in the layer.
        beta (float): Membrane decay constant (leak rate).
        device (str or torch.device): Device to store tensors (e.g., 'cuda' or 'cpu').
        dtype (torch.dtype): Data type for tensors (e.g., torch.float).
        ff_weights (torch.Tensor): Feedforward weight matrix of shape (nb_inputs, nb_neurons).
        mem (torch.Tensor): Membrane potential tensor of shape (batch_size, nb_neurons).
    """

    def __init__(self, batch_size, nb_inputs, nb_neurons, beta, requires_grad, fwd_scale=0.1, weights=None, device="cuda", dtype=torch.float):
        """
        Initialize the LI neuron layer with weights and parameters.

        Args:
            batch_size (int): Batch size for input data.
            nb_inputs (int): Number of input neurons.
            nb_neurons (int): Number of LI neurons in the layer.
            fwd_scale (float): Scaling factor for feedforward weight initialization.
            beta (float): Membrane decay constant (leak rate).
            weights (torch.Tensor, optional): Predefined weight matrix of shape (nb_inputs, nb_neurons).
            device (str or torch.device, optional): Device to store tensors (default: "cuda").
            dtype (torch.dtype, optional): Data type for tensors (default: torch.float).
            requires_grad (bool, optional): Whether the weights require gradients (default: True).
        """
        super().__init__(batch_size=batch_size, nb_inputs=nb_inputs, nb_neurons=nb_neurons, beta=beta, fwd_scale=fwd_scale, device=device, dtype=dtype, requires_grad=requires_grad)

        if weights is not None:
            self.ff_weights = self.init_parameter_copy(weights, self.requires_grad)
        else:
            # Initialize the feedforward layer weights
            self.ff_weights = self.init_weights_random_normal(self, self.requires_grad)

        self.ff_weights_init = self.init_parameter_copy(self.ff_weights, self.requires_grad).detach()

        # Initialize the synaptic current and membrane potential
        self.mem = self.init_state()
        
        
    def reset(self):
        self.mem = self.reset_state(self.mem)
    
    

    def state_dict(self):
        """
        Returns a dictionary containing the state of the LI layer, including weights and parameters.
        This can be used for saving and loading the model state.

        Returns:
            dict: Dictionary containing the state of the layer.
        """
        return {
            'ff_weights': self.ff_weights,
            'beta': self.beta,
        }
        
        
    def forward(self, input_activity_t):
        """
        Compute the membrane potential of the LI neuron layer for a single time step.

        Args:
            input_activity_t (torch.Tensor): Input activity tensor of shape (batch_size, nb_inputs)
                                             for a single time step.

        Returns:
            torch.Tensor: Updated membrane potential tensor of shape (batch_size, nb_neurons).
        """

        self.mem = (self.beta * self.mem +
                    torch.einsum("ab,bc->ac", input_activity_t, self.ff_weights))

        return self.mem






class LIF(Neuron):
    """
    Class to initialize and compute a feedforward layer of Leaky Integrate-and-Fire (LIF) neurons.

    This class implements a feedforward layer of LIF neurons, which accumulate input over time with a leaky (decaying) membrane potential and emit spikes when the membrane potential crosses a threshold. The layer supports computation of membrane potentials and spike outputs for each neuron at each time step, using surrogate gradients to enable backpropagation through the non-differentiable spike function.

    Attributes:
        nb_inputs (int): Number of input neurons.
        nb_neurons (int): Number of LIF neurons in the layer.
        beta (float): Membrane decay constant (leak rate).
        device (str or torch.device): Device to store tensors (e.g., 'cuda' or 'cpu').
        dtype (torch.dtype): Data type for tensors (e.g., torch.float).
        ff_weights (torch.Tensor): Feedforward weight matrix of shape (nb_inputs, nb_neurons).
        mem (torch.Tensor): Membrane potential tensor of shape (batch_size, nb_neurons).
        rst (torch.Tensor): Reset state tensor of shape (batch_size, nb_neurons), indicating which neurons have just spiked.
    """

    def __init__(self, batch_size, nb_inputs, nb_neurons, beta, requires_grad, fwd_scale=0.1, weights=None, device="cuda", dtype=torch.float):
        """
        Initialize the LIF neuron layer with weights and parameters.

        Args:
            batch_size (int): Batch size for input data.
            nb_inputs (int): Number of input neurons.
            nb_neurons (int): Number of LIF neurons in the layer.
            fwd_scale (float): Scaling factor for feedforward weight initialization.
            beta (float): Membrane decay constant (leak rate).
            weights (torch.Tensor, optional): Predefined weight matrix of shape (nb_inputs, nb_neurons).
            device (str or torch.device, optional): Device to store tensors (default: "cuda").
            dtype (torch.dtype, optional): Data type for tensors (default: torch.float).
            requires_grad (bool, optional): Whether the weights require gradients (default: True).
        """
        super().__init__(batch_size=batch_size, nb_inputs=nb_inputs, nb_neurons=nb_neurons, beta=beta, fwd_scale=fwd_scale, device=device, dtype=dtype, requires_grad=requires_grad)

        if weights is not None:
            self.ff_weights = self.init_parameter_copy(weights, self.requires_grad)
        else:
            # Initialize the feedforward layer weights
            self.ff_weights = self.init_weights_random_normal(self,self.requires_grad)

        # Initialize the synaptic current and membrane potential
        self.mem = self.init_state()
        self.rst = self.init_state()

    def state_dict(self):
        """
        Returns a dictionary containing the state of the LI layer, including weights and parameters.
        This can be used for saving and loading the model state.

        Returns:
            dict: Dictionary containing the state of the layer.
        """
        return {
            'ff_weights': self.ff_weights,
            'beta': self.beta,
        }

    def reset(self):
        self.mem = self.reset_state(self.mem)

    def forward(self, input_activity_t):
        """
        Compute the membrane potential and spike output of the LIF neuron layer for a single time step.

        The membrane potential is updated using a leaky integration of the input, and a spike is emitted if the membrane potential crosses the threshold. The reset state is updated to reflect which neurons have spiked.

        Args:
            input_activity_t (torch.Tensor): Input activity tensor of shape (batch_size, nb_inputs) for a single time step.

        Returns:
            tuple:
                - out (torch.Tensor): Spike output tensor of shape (batch_size, nb_neurons), with 1 indicating a spike.
                - mem (torch.Tensor): Updated membrane potential tensor of shape (batch_size, nb_neurons).
        """

        self.mem = (self.beta * self.mem + torch.einsum("ab,bc->ac",
                    input_activity_t, self.ff_weights)) * (1.0 - self.rst)

        mthr = self.mem - 1.0
        out = spike_fn(mthr)
        self.rst = out.detach()

        return self.rst, self.mem





class RLIF(Neuron):
    """
    Class to initialize and compute a recurrent layer of Leaky Integrate-and-Fire (LIF) neurons.

    This class implements a recurrent layer of LIF neurons, which accumulate input over time with a leaky (decaying) membrane potential and emit spikes when the membrane potential crosses a threshold. The layer supports computation of membrane potentials and spike outputs for each neuron at each time step, using surrogate gradients to enable backpropagation through the non-differentiable spike function.

    Attributes:
        nb_inputs (int): Number of input neurons.
        nb_neurons (int): Number of LIF neurons in the layer.
        beta (float): Membrane decay constant (leak rate).
        device (str or torch.device): Device to store tensors (e.g., 'cuda' or 'cpu').
        dtype (torch.dtype): Data type for tensors (e.g., torch.float).
        ff_weights (torch.Tensor): Feedforward weight matrix of shape (nb_inputs, nb_neurons).
        mem (torch.Tensor): Membrane potential tensor of shape (batch_size, nb_neurons).
        rst (torch.Tensor): Reset state tensor of shape (batch_size, nb_neurons), indicating which neurons have just spiked.
    """

    def __init__(self, batch_size, nb_inputs, nb_neurons, beta, requires_grad, fwd_scale=0.1, weights=None, device="cuda", dtype=torch.float):
        """
        Initialize the LIF neuron recurrent layer with weights and parameters.

        Args:
            batch_size (int): Batch size for input data.
            nb_inputs (int): Number of input neurons.
            nb_neurons (int): Number of LIF neurons in the layer.
            fwd_scale (float): Scaling factor for feedforward weight initialization.
            beta (float): Membrane decay constant (leak rate).
            weights (torch.Tensor, optional): Predefined weight matrix of shape (nb_inputs, nb_neurons).
            device (str or torch.device, optional): Device to store tensors (default: "cuda").
            dtype (torch.dtype, optional): Data type for tensors (default: torch.float).
            requires_grad (bool, optional): Whether the weights require gradients (default: True).
        """
        super().__init__(batch_size=batch_size, nb_inputs=nb_inputs, nb_neurons=nb_neurons, beta=beta, fwd_scale=fwd_scale, device=device, dtype=dtype, requires_grad=requires_grad)

        self.struct = None # no need for low dim now

        if weights is not None:
            if self.struct:
                self.ff_weights = self.init_parameter_copy(weights[0], self.requires_grad)
                self.rec_weights_left = self.init_parameter_copy(weights[1], self.requires_grad)
                self.rec_weights_right = self.init_parameter_copy(weights[2], self.requires_grad)
                self.rec_weights = torch.einsum("ikd,kjd->ijd", self.rec_weights_left, self.rec_weights_right)
            else:
                self.ff_weights = self.init_parameter_copy(weights[0], self.requires_grad)
                self.rec_weights = self.init_parameter_copy(weights[1], self.requires_grad)
                # self.rec_weights_temp = weights[1]
        else:
            # Initialize feedforward and recurrent weights
            # feedforward do not necessarily need to be learnt, just an encoding
            self.ff_weights = self.init_weights_random_normal(self, shape = (nb_inputs, nb_neurons), requires_grad=requires_grad)
            
            if self.struct is None:
                # Recurrent weights need to be learnt
                self.rec_weights = self.init_weights_random_normal(self, shape = (nb_neurons, nb_neurons, depth), 
                                                                   scale = self.rec_scale, neurons = nb_neurons, requires_grad=requires_grad)

            else:
                # low-rank recurrent weights
                self.rec_weights_left = self.init_weights_random_normal(self, shape = (nb_neurons, self.recurrent_weights_rank, depth), 
                                                                        scale = self.rec_scale, neurons = nb_neurons,requires_grad=requires_grad)
                self.rec_weights_right = self.init_weights_random_normal(self, shape = (self.recurrent_weights_rank,nb_neurons, depth), 
                                                                         scale = self.rec_scale, neurons = nb_neurons,requires_grad=requires_grad)
                self.rec_weights = torch.einsum("ikd,kjd->ijd", self.rec_weights_left, self.rec_weights_right)
            
        
        self.ff_weights_init = self.init_parameter_copy(weights[0], self.requires_grad).detach()
        self.rec_weights_init = self.init_parameter_copy(weights[1], self.requires_grad).detach()

        # Initialize the synaptic current and membrane potential
        self.mem = self.init_state()
        self.rst = self.init_state()

    def state_dict(self):
        """
        Returns a dictionary containing the state of the LI layer, including weights and parameters.
        This can be used for saving and loading the model state.

        Returns:
            dict: Dictionary containing the state of the layer.
        """
        return {
            'ff_weights': self.ff_weights,
            'rec_weights': self.rec_weights,
            'beta': self.beta,
        }

    def reset(self):
        self.mem = self.reset_state(self.mem)

    def forward(self, input_activity_t):
        """
        Compute the membrane potential and spike output of the LIF neuron layer for a single time step.

        The membrane potential is updated using a leaky integration of the input, and a spike is emitted if the membrane potential crosses the threshold. The reset state is updated to reflect which neurons have spiked.

        Args:
            input_activity_t (torch.Tensor): Input activity tensor of shape (batch_size, nb_inputs) for a single time step.

        Returns:
            tuple:
                - out (torch.Tensor): Spike output tensor of shape (batch_size, nb_neurons), with 1 indicating a spike.
                - mem (torch.Tensor): Updated membrane potential tensor of shape (batch_size, nb_neurons).
        """

        if self.struct==None:                
            h1 = torch.einsum("ba,ac->bc", input_activity_t, self.ff_weights) + \
                torch.einsum("bd,edz->bdz", self.rst, self.rec_weights).squeeze(-1)
            h1 = h1.unsqueeze(-1)
        else:
            # pass
            self.mem_inner_dim = torch.einsum("bdz,dez->be", self.rst, self.rec_weights_left)
            
            h1 = torch.einsum("ba,ac->bc", input_activity_t, self.ff_weights).unsqueeze(-1) 
            h1a =  torch.einsum("ab,bcd->acd", self.mem_inner_dim, self.rec_weights_right)
            
            h1 = h1 + h1a

        self.mem = (self.beta * self.mem + torch.einsum("ab,bc->ac",
                    input_activity_t, self.ff_weights)) * (1.0 - self.rst)

        mthr = self.mem - 1.0
        out = spike_fn(mthr)
        self.rst = out.detach()

        return self.rst, self.mem, self.mem  # duplicate so the full structure does not fall apart





class SE_adRLIF(Neuron):
    """
    Class to initialize and compute a recurrent layer of Symplectic-Euler discretized adaptive(ad) Leaky Integrate-and-Fire (SE-adRLIF) neurons.

    This class implements a recurrent layer of SE-adLIF neurons with both feedforward and recurrent connections. Each neuron's membrane potential is updated by a leaky integration of the weighted input and recurrent activity, and a spike is emitted when the membrane potential crosses a threshold. The layer supports computation of membrane potentials and spike outputs for each neuron at each time step, using surrogate gradients to enable backpropagation through the non-differentiable spike function.
    The discreet integration is carried with the Symplectic-Euler method, which is suitable for simulating dynamical systems with energy conservation properties.

    Other than the membrane voltage, the neuron's behaviour is determined by the parameter w, which is the adaptation current. 

    Attributes:
        nb_inputs (int): Number of input neurons.
        nb_neurons (int): Number of recurrent neurons.
        depth (int) : number of times the network can be repeated (e.g. for multiple joints or multiple variables)
        beta (float): Membrane decay constant.
        gamma (float): Adaptation decay constant.
        a (float): Adaptation parameter
        b (float): Adaptation parameter
        device (str or torch.device): Device to store tensors (e.g., 'cuda' or 'cpu').
        dtype (torch.dtype): Data type for tensors (e.g., torch.float).
        ff_weights (torch.Tensor): Feedforward weight matrix of shape (nb_inputs, nb_neurons).
        rec_weights (torch.Tensor): Recurrent weight matrix of shape (nb_neurons, nb_neurons).
        mem (torch.Tensor): Membrane potential tensor of shape (batch_size, nb_neurons).
        rst (torch.Tensor): Reset state tensor of shape (batch_size, nb_neurons), indicating which neurons have just spiked.
    """

    def __init__(self, batch_size, nb_inputs, nb_neurons, beta, beta_requires_grad, gamma, gamma_requires_grad, a, a_requires_grad, b, b_requires_grad, requires_grad,
                 fwd_scale=0.1, rec_scale=0.9, weights=None, depth = 1, device="cuda", dtype=torch.float, struct = False, inner_dim = None):
        """
        Initialize the RLIF neuron layer with weights and parameters.

        Args:
            batch_size (int): Batch size for input data.
            nb_inputs (int): Number of input neurons.
            nb_neurons (int): Number of RECURRENT neurons.
            fwd_scale (float): Scaling factor for feedforward weight initialization.
            rec_scale (float): Scaling factor for recurrent weight initialization.
            beta (float): Membrane decay constant.
            weights (tuple of torch.Tensor, optional): Tuple containing predefined feedforward and recurrent weight matrices.
            device (str or torch.device, optional): Device to store tensors (default: "cuda").
            dtype (torch.dtype, optional): Data type for tensors (default: torch.float).
            requires_grad (bool, optional): Whether the weights require gradients (default: True).
        """

        super().__init__(batch_size=batch_size, nb_inputs=nb_inputs, nb_neurons=nb_neurons, beta=beta, fwd_scale=fwd_scale, device=device, dtype=dtype, requires_grad=requires_grad)

        self.shape_states = (batch_size, nb_neurons, depth)
        self.param_shape = (nb_neurons, depth)
        self.rec_scale = rec_scale

        if struct:
            self.struct = struct
            assert inner_dim is not None, "Recurrent matrix rank must be specified when struct is True"
            self.recurrent_weights_rank = inner_dim
            self.mem_inner_dim = self.init_state()

        else:
            self.struct = None
        
        # self.mask = torch.eye(self.nb_neurons, self.nb_neurons, device=device).bool()

        if weights is not None:
            if self.struct:
                self.ff_weights = self.init_parameter_copy(weights[0], self.requires_grad)
                self.rec_weights_left = self.init_parameter_copy(weights[1], self.requires_grad)
                self.rec_weights_right = self.init_parameter_copy(weights[2], self.requires_grad)
                self.rec_weights = torch.einsum("ikd,kjd->ijd", self.rec_weights_left, self.rec_weights_right)
            else:
                self.ff_weights = self.init_parameter_copy(weights[0], self.requires_grad)
                self.rec_weights = self.init_parameter_copy(weights[1], self.requires_grad)
                # self.rec_weights_temp = weights[1]
        else:
            # Initialize feedforward and recurrent weights
            # feedforward do not necessarily need to be learnt, just an encoding
            self.ff_weights = self.init_weights_random_normal(self, shape = (nb_inputs, nb_neurons), requires_grad=requires_grad)
            
            if self.struct is None:
                # Recurrent weights need to be learnt
                self.rec_weights = self.init_weights_random_normal(self, shape = (nb_neurons, nb_neurons, depth), 
                                                                   scale = self.rec_scale, neurons = nb_neurons, requires_grad=requires_grad)

            else:
                # low-rank recurrent weights
                self.rec_weights_left = self.init_weights_random_normal(self, shape = (nb_neurons, self.recurrent_weights_rank, depth), 
                                                                        scale = self.rec_scale, neurons = nb_neurons,requires_grad=requires_grad)
                self.rec_weights_right = self.init_weights_random_normal(self, shape = (self.recurrent_weights_rank,nb_neurons, depth), 
                                                                         scale = self.rec_scale, neurons = nb_neurons,requires_grad=requires_grad)
                self.rec_weights = torch.einsum("ikd,kjd->ijd", self.rec_weights_left, self.rec_weights_right)
            
        
        self.ff_weights_init = self.init_parameter_copy(weights[0], self.requires_grad).detach()
        self.rec_weights_init = self.init_parameter_copy(weights[1], self.requires_grad).detach()
        
        if a.dim() == 0:
            self.a = self.init_weights_random_unif(self, param_scalar=a, shape = self.param_shape, requires_grad=a_requires_grad) 
        else: 
            assert(a.shape == self.param_shape), "Parameter 'a' must be a scalar or have shape (batch_size, nb_neurons, depth)"
            self.a = self.init_parameter_copy(a, requires_grad=a_requires_grad)
            
        if b.dim() == 0:   
            self.b = self.init_weights_random_unif(self, param_scalar=b, shape = self.param_shape, requires_grad=b_requires_grad) 
        else:
            assert(b.shape == self.param_shape), "Parameter 'b' must be a scalar or have shape (batch_size, nb_neurons, depth)"
            self.b = self.init_parameter_copy(b, requires_grad=b_requires_grad)

        self.a_init = self.init_parameter_copy(self.a, requires_grad=False).detach()
        self.b_init = self.init_parameter_copy(self.b, requires_grad=False).detach()
        
        
        # Here now I use self.a_init and self.b_init to reset self.a and self.b so that what I actually learn is a small perturbation around the already sampled values.
        # This is done because the optimization of the dynamics of the system requires a bit of a variation from an already "good" starting point. 
        # Changing directly as and bs without the use of this small perturbation will lead to nan values during backprop
        # The same will be done for beta and gamma
        
        self.a = nn.Parameter(torch.zeros_like(self.a_init), requires_grad=self.a.requires_grad)
        self.b = nn.Parameter(torch.zeros_like(self.b_init), requires_grad=self.b.requires_grad)
        
        

        
        # Having beta and gamma assumes that the "time constants" of the neruons are the same
        # but having a and b different (and learnable) for each neuron implies that each neuron has a sort of "inner frequency" 
        # which aids in the temporal processing of the network.    
        
        if isinstance(beta, torch.Tensor):
            if beta.dim() == 0:
                if beta_requires_grad:
                    self.beta = torch.nn.Parameter( beta * torch.ones(self.param_shape,device=self.device,dtype=self.dtype),requires_grad=beta_requires_grad)
                else:
                    self.beta = beta
            else:
                assert(beta.shape == self.param_shape), "Parameter 'beta' must be a scalar or have shape (batch_size, nb_neurons, depth)"
                self.beta = self.init_parameter_copy(beta, requires_grad=beta_requires_grad)
        else:
            Warning("Beta is not a tensor, setting to default 0.9")
            self.beta = torch.tensor(0.9)
            
            
            
            
            
        if isinstance(gamma, torch.Tensor):
            if gamma.dim() == 0:
                if gamma_requires_grad:
                    self.gamma = torch.nn.Parameter( gamma * torch.ones(self.param_shape,device=self.device,dtype=self.dtype),requires_grad=gamma_requires_grad)
                else:
                    self.gamma = gamma
            else:
                assert(gamma.shape == self.param_shape), "Parameter 'gamma' must be a scalar or have shape (batch_size, nb_neurons, depth)"
                self.gamma = self.init_parameter_copy(gamma, requires_grad=gamma_requires_grad)
        else:
            Warning("Gamma is not a tensor, setting to default 0.9")
            self.gamma = torch.tensor(0.9)

        # if not beta and not gamma:
        #     self.beta = self.init_weights_random_unif(self, min_value= 0.83, param_scalar=0.96, shape = self.shape_states, requires_grad=False) 
        #     self.gamma = self.init_weights_random_unif(self, min_value= 0.92, param_scalar=0.99, shape = self.shape_states, requires_grad=False) 
        # else:
        #     self.beta = beta[0]
        #     self.gamma = gamma[0]

        self.beta_init = self.init_parameter_copy(self.beta, requires_grad=False).detach()
        self.gamma_init = self.init_parameter_copy(self.gamma, requires_grad=False).detach()
        
        
        # For the reason behind the following lines, see above at self.a. and self.b

        self.beta = nn.Parameter(torch.zeros_like(self.beta), requires_grad=self.beta.requires_grad)
        self.gamma = nn.Parameter(torch.zeros_like(self.gamma), requires_grad=self.gamma.requires_grad)
        
        

        # Initialize synaptic current, membrane potential, and spike output
        self.mem = self.init_state()
        self.curr = self.init_state()
        self.rst = self.init_state()

    def reset(self):
        """
        Reset the membrane potential, adaptation current, and reset state of the SE-adLIF layer.
        This is useful to clear the state before processing a new batch of inputs.
        """
        self.mem = self.reset_state(self.mem)
        self.curr = self.reset_state(self.curr)
        self.rst = self.reset_state(self.rst)
       
    
    def init(self):
        """
        Reset the membrane potential, adaptation current, and reset state of the SE-adLIF layer.
        This is useful to clear the state before processing a new batch of inputs.
        """
        self.mem = self.init_state() 
        self.curr = self.init_state() 
        self.rst = self.init_state()
    
        return self.mem, self.curr, self.rst
    

        
        
    def state_dict(self):
        """
        Returns a dictionary containing the state of the SE-adLIF layer, including weights and parameters.
        This can be used for saving and loading the model state.

        Returns:
            dict: Dictionary containing the state of the layer.
        """
        return {
            'ff_weights': self.ff_weights,
            'rec_weights': self.rec_weights,
            'a': self.a,
            'b': self.b,
            'beta': self.beta,
            'gamma': self.gamma
        } if self.struct is None else {
            'ff_weights': self.ff_weights,
            'rec_weights_left': self.rec_weights_left,
            'rec_weights_right': self.rec_weights_right,
            'a': self.a,
            'b': self.b,
            'beta': self.beta,
            'gamma': self.gamma
        }


    def forward(self, input_activity_t):
        """
        Compute the activity of the recurrent SE-adLIF layer for a single time step.

        The membrane potential is updated using a leaky integration of the feedforward input and recurrent activity. A spike is emitted if the membrane potential crosses the threshold, and the reset state is updated accordingly.
        Based on the updated membrane value, the adaptation current is updated using the Symplectic-Euler method using a leacky integration.

        Args:
            input_activity_t (torch.Tensor): Input activity tensor of shape (batch_size, nb_inputs) for a single time step.

        Returns:
            tuple:
                - out (torch.Tensor): Spike output tensor of shape (batch_size, nb_neurons), with 1 indicating a spike.
                - mem (torch.Tensor): Updated membrane potential tensor of shape (batch_size, nb_neurons).
        """

        #mask = torch.eye(self.nb_neurons, self.nb_neurons).byte()
        
        # self.rec_weights_temp = self.rec_weights.clone().masked_fill_(self.mask, 0)  # ensure no self-connections in recurrent weights

        # Compute input and recurrent contributions
        # h1 = torch.einsum("zba,ac->zbc", input_activity_t, self.ff_weights).transpose(1, 2) + \
        #     torch.einsum("zbd,bcd->zcd", self.rst, self.rec_weights)
        
        
        if self.struct==None:                
            h1 = torch.einsum("ba,ac->bc", input_activity_t, self.ff_weights) + \
                torch.einsum("bdz,edz->bd", self.rst, self.rec_weights)
            h1 = h1.unsqueeze(-1)
        else:
            # pass
            self.mem_inner_dim = torch.einsum("bdz,dez->be", self.rst, self.rec_weights_left)
            
            h1 = torch.einsum("ba,ac->bc", input_activity_t, self.ff_weights).unsqueeze(-1) 
            h1a =  torch.einsum("ab,bcd->acd", self.mem_inner_dim, self.rec_weights_right)
            
            h1 = h1 + h1a

        a = self.a_init + 2.00 * torch.tanh(self.a)
        b = self.b_init + 2.00 * torch.tanh(self.b)
        gamma = self.gamma_init + 0.010 * torch.tanh(self.gamma)
        beta = self.beta_init + 0.010 * torch.tanh(self.beta)

        # h0 = h1.unsqueeze(-1)
        # Update synaptic current and membrane potential
        self.mem = (beta * self.mem) + (1.0 - beta) * (-self.curr + h1)#.unsqueeze(-1))# + h1a)

        mthr = self.mem - 1.0
        out = spike_fn(mthr) #SurrGradSpike
        self.mem = self.mem * (1.0 - out)  # Reset membrane potential where spike occurred
        
        self.curr = (gamma * self.curr)  + (1.0 - gamma)* (a * self.mem + b * out) 
        
        self.rst = out#.detach()  # Reset spikes

        return self.rst, self.curr, self.mem











def recover_idxs(model, input_space):
    joint_idxs = [mujoco.mj_name2id( model, mujoco.mjtObj.mjOBJ_JOINT, list(input_space.keys())[0])]
    return joint_idxs
    


class GaussPop(Neuron):
    
    """
    Class to initialize and compute a feedforward layer of Leaky Integrate-and-Fire (LIF) neurons.

    This class implements a feedforward layer of LIF neurons, which accumulate input over time with a leaky (decaying) membrane potential and emit spikes when the membrane potential crosses a threshold. The layer supports computation of membrane potentials and spike outputs for each neuron at each time step, using surrogate gradients to enable backpropagation through the non-differentiable spike function.

    Attributes:
        nb_inputs (int): Number of input neurons.
        nb_neurons (int): Number of LIF neurons in the layer.
        beta (float): Membrane decay constant (leak rate).
        device (str or torch.device): Device to store tensors (e.g., 'cuda' or 'cpu').
        dtype (torch.dtype): Data type for tensors (e.g., torch.float).
        ff_weights (torch.Tensor): Feedforward weight matrix of shape (nb_inputs, nb_neurons).
        mem (torch.Tensor): Membrane potential tensor of shape (batch_size, nb_neurons).
        rst (torch.Tensor): Reset state tensor of shape (batch_size, nb_neurons), indicating which neurons have just spiked.
    """
    

    def __init__(self, input_space, batch_size, nb_inputs, nb_neurons, beta, requires_grad, depth = 1, model=None, max_curr=5, fwd_scale=0.1, weights=None, device="cuda", dtype=torch.float):
        """
        Initialize the LIF neuron layer with weights and parameters.

        Args:
            batch_size (int): Batch size for input data.
            nb_inputs (int): Number of input neurons.
            nb_neurons (int): Number of LIF neurons in the layer.
            fwd_scale (float): Scaling factor for feedforward weight initialization.
            beta (float): Membrane decay constant (leak rate).
            weights (torch.Tensor, optional): Predefined weight matrix of shape (nb_inputs, nb_neurons).
            device (str or torch.device, optional): Device to store tensors (default: "cuda").
            dtype (torch.dtype, optional): Data type for tensors (default: torch.float).
            requires_grad (bool, optional): Whether the weights require gradients (default: True).
        """
        super().__init__(batch_size=batch_size, nb_inputs=nb_inputs, nb_neurons=nb_neurons, beta=beta, fwd_scale=fwd_scale, device=device, dtype=dtype, requires_grad=requires_grad)

        self.shape_states = (self.batch_size, self.nb_inputs, nb_neurons) # this is like the SE-adLIF but the depth is technically the nb_inputs
        self.encoded_positions = torch.zeros((nb_inputs, nb_neurons), device=device, dtype=dtype) # values of the encoded joint positions, the space where I am "living"
        self.max_curr = torch.tensor(max_curr) # Hz whatever
        self.input_space = input_space
        self.depth = depth
        
        if model is not None:
            self.model = model
            self.joint_idxs = recover_idxs(self.model, input_space=input_space)
        else:
            self.model = None
            
        if weights is not None:
            self.ff_weights = self.init_parameter_copy(weights, self.requires_grad)
        else:
            # Initialize the feedforward layer weights
            self.ff_weights = self.init_weights_random_normal(self,self.requires_grad) # Check dimensionality 


        # Initialize the population with the joint angles and pre-compute sigma for each input
        self.sigma = torch.zeros(nb_inputs, device=device, dtype=dtype)
        self.neighbor_ratio = 0.2  # Controls the width of the Gaussian tuning curve
        
        for i, input_name in enumerate(list(input_space.keys())):
            # Create a range of encoded positions for each input
            self.encoded_positions[i] = torch.linspace(input_space[input_name]['min'], input_space[input_name]['max'], nb_neurons, device=self.device, dtype=self.dtype)
            
            # Pre-compute sigma for this input dimension
            spacing = (input_space[input_name]['max'] - input_space[input_name]['min']) / (nb_neurons - 1) if nb_neurons > 1 else 1.0
            self.sigma[i] = spacing / torch.sqrt(torch.tensor(-2.0 * np.log(self.neighbor_ratio), device=self.device, dtype=self.dtype))
         
        self.encoded_positions = self.encoded_positions.detach().clone()  
        # this is needed to shape the input current    
        diff  = torch.abs(self.encoded_positions[:,0]- self.encoded_positions[:,-1])
        diff = diff.repeat_interleave(nb_neurons)
        diff = torch.reshape(diff, (nb_inputs, nb_neurons))    
        self.normalization = diff.detach().clone()
        
        # Initialize the input currents as zeros
        self.input_curr = self.init_state() # this is the "synaptic current" that will be fed to the membrane potential update, it has the same shape as the SE-adLIF but the depth is technically the nb_inputs

        # Initialize the synaptic current and membrane potential
        self.mem = self.init_state()
        self.rst = self.init_state()

    def state_dict(self):
        """
        Returns a dictionary containing the state of the LI layer, including weights and parameters.
        This can be used for saving and loading the model state.

        Returns:
            dict: Dictionary containing the state of the layer.
        """
        return {
            'ff_weights': self.ff_weights,
            'beta': self.beta,
        }

    def reset(self):
        self.mem = self.reset_state(self.mem)
    
    def gaussian_input_currents_vectorized(self, input_joint_positions_t, i_max=1.0):
        """
        Vectorized computation of gaussian input currents for entire batch at once.
        
        Args:
            input_joint_positions_t (torch.Tensor): Shape (batch_size, nb_inputs), input positions
            i_max (float): Maximum current value
            
        Returns:
            torch.Tensor: Shape (batch_size, nb_inputs, nb_neurons), gaussian currents for all neurons
        """
        # encoded_positions: (nb_inputs, nb_neurons)
        # input_joint_positions_t: (batch_size, nb_inputs)
        # sigma: (nb_inputs,)
        
        # Expand for broadcasting
        positions = self.encoded_positions.unsqueeze(0)  # (1, nb_inputs, nb_neurons)
        x0 = input_joint_positions_t.unsqueeze(-1)  # (batch_size, nb_inputs, 1)
        sigma = self.sigma.unsqueeze(0).unsqueeze(-1)  # (1, nb_inputs, 1)
        
        # Compute gaussian: (batch_size, nb_inputs, nb_neurons)
        diff = positions - x0  # (batch_size, nb_inputs, nb_neurons)
        currents = i_max * torch.exp(-0.5 * (diff / sigma) ** 2)
        
        return currents



    def forward(self, input_joint_positions_t):
        """
        Step function to update the state of the population.
        
        This function should be called at each time step to update the spikes based on the current joint positions.
        
        The closer the input is to the encoded positions, the higher the current in the neuron so the most probable the spike.
        """

        if self.model is not None:
            # maybe his does not work for multiplwe joints
            mask = torch.isin(input_joint_positions_t[:, :, 0], self.joint_idxs)#.cuda())
            input_joint_positions_t = input_joint_positions_t[:, :, 1][mask].view(input_joint_positions_t .size(0), -1)#input_joint_positions_t[:,mask,1]#[:, input_joint_positions_t[:, :, 0] == joint_id, 1]
        
        # Vectorized computation of gaussian currents for entire batch at once
        self.input_curr = self.gaussian_input_currents_vectorized(input_joint_positions_t, i_max=1.0) 

        # WEIGHTS RESHAPING
        W_new = self.ff_weights.view(self.nb_inputs, self.nb_inputs, self.nb_neurons) 
        weights = W_new.diagonal(dim1=0, dim2=1)  # [k, self.nb_inputs]
        weights = weights.permute(1, 0)

        # simple lif model without threshold nor inner current
        self.mem = (self.beta * self.mem + self.input_curr * weights.unsqueeze(0) ) * (1.0 - self.rst)
        
        # print(input_curr)
        # print(self.mem)
        mthr = self.mem - 1.0
        out = spike_fn(mthr) #SurrGradSpike
        self.mem = self.mem * (1.0 - out)  # Reset membrane potential where spike occurred
        
        self.rst = out.detach()  # Reset spikes

        # Now I flatten them so they are compatible with the other eencoding methods
        if self.depth == 1:
            rst_out = torch.flatten(self.rst, start_dim=1)
            mem_out = torch.flatten(self.mem, start_dim = 1)
        else:
            rst_out = self.rst.detach().clone() # TODO .detach().clone() or no?
            mem_out = self.mem.detach().clone()
            
        return rst_out, mem_out















# TODO make this class compatible
 
# class Model_Complete(Neuron):

#     # Which One neurons / things? be careful with the ** kwargs

#     def __init__(self, batch_size, nb_inputs, nb_neurons, nb_output_neurons,
#                  beta, gamma, beta_decode, a, b, fwd_scale=0.1, rec_scale=0.9, fwd_scale_decode=0.1,
#                  weights=None, weights_decoder=None, temporal_steps = 1, depth = 1, 
#                  device="cuda", dtype=torch.float, requires_grad=True, struct = False, inner_dim = None):

#         super().__init__(batch_size=batch_size, nb_inputs=nb_inputs, nb_neurons=nb_neurons, beta=beta, fwd_scale=fwd_scale, device=device, dtype=dtype, requires_grad=requires_grad)

#         self.nb_input_neurons = nb_inputs
#         self.nb_recurrent_neurons = nb_neurons
#         self.nb_output_neurons = nb_output_neurons
#         self.depth = depth
#         self.beta = beta
#         self.gamma = gamma
#         self.beta_decode = beta_decode
#         self.requires_grad = requires_grad  
#         self.shape_states = (batch_size, nb_neurons, depth)
#         self.rec_scale = rec_scale
#         self.fwd_scale_decode = fwd_scale_decode
#         Warning("This class is yet to be debugged!")
#         i = input("Press Enter to continue: ")
        
#         if struct:
#             self.struct = struct
#             assert inner_dim is not None, "Recurrent matrix rank must be specified when struct is True"
#             self.recurrent_weights_rank = inner_dim
#             self.mem_inner_dim = self.init_state()

#         else:
#             self.struct = None
        
#         # self.mask = torch.eye(self.nb_neurons, self.nb_neurons, device=device).bool()

#         if weights is not None:
#             if self.struct:
#                 self.ff_weights = self.init_parameter_copy(weights[0], requires_grad=self.requires_grad)
#                 self.rec_weights_left = self.init_parameter_copy(weights[1], requires_grad=self.requires_grad)
#                 self.rec_weights_right = self.init_parameter_copy(weights[2], requires_grad=self.requires_grad)
#                 self.rec_weights = torch.einsum("ikd,kjd->ijd", self.rec_weights_left, self.rec_weights_right)
#             else:
#                 self.ff_weights = self.init_parameter_copy(weights[0], requires_grad=self.requires_grad)
#                 self.rec_weights = self.init_parameter_copy(weights[1], requires_grad=self.requires_grad)
#                 # self.rec_weights_temp = weights[1]
#         else:
#             # Initialize feedforward and recurrent weights
#             # feedforward do not necessarily need to be learnt, just an encoding
#             self.ff_weights = self.init_weights_random_normal(self, shape = (self.nb_recurrent_neurons, self.nb_recurrent_neurons), requires_grad=requires_grad)
            
#             if self.struct is None:
#                 # Recurrent weights need to be learnt
#                 self.rec_weights = self.init_weights_random_normal(self, shape = (self.nb_recurrent_neurons, self.nb_recurrent_neurons, depth), 
#                                                                    scale = self.rec_scale, neurons = self.nb_recurrent_neurons, requires_grad=requires_grad)

#             else:
#                 # low-rank recurrent weights
#                 self.rec_weights_left = self.init_weights_random_normal(self, shape = (self.nb_recurrent_neurons, self.recurrent_weights_rank, depth), 
#                                                                         scale = self.rec_scale, neurons = self.nb_recurrent_neurons,requires_grad=requires_grad)
#                 self.rec_weights_right = self.init_weights_random_normal(self, shape = (self.recurrent_weights_rank,self.nb_recurrent_neurons, depth), 
#                                                                          scale = self.rec_scale, neurons = self.nb_recurrent_neurons,requires_grad=requires_grad)
#                 self.rec_weights = torch.einsum("ikd,kjd->ijd", self.rec_weights_left, self.rec_weights_right)
            
        
#         self.ff_weights_init = self.init_parameter_copy(self.ff_weights).detach()
#         self.rec_weights_init = self.init_parameter_copy(self.rec_weights).detach()
        
        
#         if weights_decoder is not None:
#             self.ff_weights_decoder = self.init_parameter_copy(weights_decoder, requires_grad=self.requires_grad)
#         else:
#             # Initialize the feedforward decoder layer weights
#             self.ff_weights_decoder = self.init_weights_random_normal(self, shape = (self.nb_recurrent_neurons, self.nb_output_neurons), requires_grad=requires_grad, scale = self.fwd_scale_decode)
        
        
#         #TODO complete this one like above
        
#         if a.dim() == 0:
#             self.a = self.init_weights_random_unif(self, param_scalar=a, shape = self.shape_states, requires_grad=False) 
#         else: 
#             assert(a.shape == self.shape_states), "Parameter 'a' must be a scalar or have shape (batch_size, nb_neurons, depth)"
#             self.a = self.init_parameter_copy(a)
            
#         if b.dim() == 0:   
#             self.b = self.init_weights_random_unif(self, param_scalar=b, shape = self.shape_states, requires_grad=False) 
#         else:
#             assert(b.shape == self.shape_states), "Parameter 'b' must be a scalar or have shape (batch_size, nb_neurons, depth)"
#             self.b = self.init_parameter_copy(b)

#         self.a_init = self.init_parameter_copy(a).detach()
#         self.b_init = self.init_parameter_copy(b).detach()

        
#         # Having beta and gamma assumes that the "time constants" of the neurons are the same
#         # but having a and b different (and learnable) for each neuron implies that each neuron has a sort of "inner frequency" 
#         # which aids in the temporal processing of the network.    
        
#         if isinstance(beta[0], torch.Tensor):
#             if beta[0].dim() == 0:
#                 if beta[1]:
#                     self.beta = torch.nn.Parameter( beta[0] * torch.ones(self.shape_states,device=self.device,dtype=self.dtype),requires_grad=beta[1])
#                 else:
#                     self.beta = beta[0]
#             else:
#                 assert(beta[0].shape == self.shape_states), "Parameter 'beta' must be a scalar or have shape (batch_size, nb_neurons, depth)"
#                 if beta[1]:
#                     self.beta = self.init_parameter_copy(beta[0])
#                 else:
#                     self.beta = beta[0]
#         else:
#             Warning("Beta is not a tensor, setting to default 0.9")
#             self.beta = torch.tensor(0.9)
            
            
            
            
            
#         if isinstance(gamma[0], torch.Tensor):
#             if gamma[0].dim() == 0:
#                 if gamma[1]:
#                     self.gamma = torch.nn.Parameter( gamma[0] * torch.ones(self.shape_states,device=self.device,dtype=self.dtype),requires_grad=gamma[1])
#                 else:
#                     self.gamma = gamma[0]
#             else:
#                 assert(gamma[0].shape == self.shape_states), "Parameter 'gamma' must be a scalar or have shape (batch_size, nb_neurons, depth)"
#                 if gamma[1]:
#                     self.gamma = self.init_parameter_copy(gamma[0])
#                 else:
#                     self.gamma = gamma[0]
#         else:
#             Warning("Gamma is not a tensor, setting to default 0.9")
#             self.gamma = torch.tensor(0.9)
              
#         # if not beta and not gamma:
#         #     self.beta = self.init_weights_random_unif(self, min_value= 0.83, param_scalar=0.96, shape = self.shape_states, requires_grad=False) 
#         #     self.gamma = self.init_weights_random_unif(self, min_value= 0.92, param_scalar=0.99, shape = self.shape_states, requires_grad=False) 
#         # else:
#         #     self.beta = beta
#         #     self.gamma = gamma


#         # Initialize synaptic current, membrane potential, and spike output
#         self.mem = self.init_state()
#         self.curr = self.init_state()
#         self.rst = self.init_state()
        
#         self.mem_out = self.init_state(shape=(batch_size, nb_output_neurons, depth))
        
#         self.dimensions = (batch_size, self.nb_recurrent_neurons, depth) 


#     def reset(self):
#         """
#         Reset the membrane potential, adaptation current, and reset state of the SE-adLIF layer.
#         This is useful to clear the state before processing a new batch of inputs.
#         """
#         self.mem = self.reset_state(self.mem)
#         self.curr = self.reset_state(self.curr)
#         self.rst = self.reset_state(self.rst)
#         self.mem_out = self.reset_state(self.mem_out)
       

#     def detach_hidden(self):
#         """
#         This funcion is specifically for backprop using time windows smaller than the totl length of the training period.
#         Detach the hidden states (membrane potential, adaptation current, and reset state) from the current computation graph.
#         This prevents gradients from propagating through time beyond the current time window,
#         """
#         self.mem = self.detach_state(self.mem)
#         self.curr = self.detach_state(self.curr)
#         self.rst = self.detach_state(self.rst)
#         self.mem_out = self.detach_state(self.mem_out)


#     def forward(self, input_activity_t, n_steps = 1, T = 1, teach_time = 0):

#         input_activity_t = torch.repeat_interleave(input_activity_t, n_steps, dim=1)
#         h1 = torch.einsum("bta,ac->btc", input_activity_t, self.ff_weights)
        
#         if self.struct:                
#             mem_inner_dim = self.mem_inner_dim[-1]
        
#         mem = self.mem[-1] #copy/detach?
#         curr = self.curr[-1]
#         rst = self.rst[-1]
#         mem_out = self.mem_out[-1]

                
#         for t in range(T):
    
#             if t >= teach_time:
#                 h1[:,int(n_steps*t):int(n_steps*t)+n_steps] =  torch.repeat_interleave(torch.unsqueeze(torch.einsum("ba,ac->bc", Utils.interleave_pos_neg_v2(mem_out), self.ff_weights), dim=1), n_steps, dim=1)    # autoregressive input 

#             for step_counter in range( n_steps ):

#                 if self.struct:
#                     mem_inner_dim = torch.einsum("bdz,dez->be", rst, self.rec_weights_left)
                    
#                     h2 =  torch.einsum("ab,bcd->acd", mem_inner_dim, self.rec_weights_right)    
                    
#                 else:
#                     h2 = torch.einsum("bdz,ddz->bd", rst, self.rec_weights).unsqueeze(-1)
                

#                 mem_temp_rec  = (self.beta * mem) + (1.0 - self.beta) * (-curr + h1[:,t+step_counter].unsqueeze(-1) + h2)
                
#                 # spikes in the recurrent layer
#                 mthr = mem_temp_rec  - 1.0
#                 out = spike_fn(mthr) #SurrGradSpike

#                 mem_temp_rec  = mem_temp_rec  * (1.0 - out)

#                 current_rec = (self.gamma * curr)  + (1.0 - self.gamma)* (self.a * mem_temp_rec+ self.b * out)

#                 # self.mem.append(mem_temp_rec)
#                 # self.curr.append(current_rec)
#                 # self.rst.append(out)
                
#                 mem = mem_temp_rec
#                 curr = current_rec
#                 rst = out
#                 # print(rst.sum().item())
                
#                 h3 = torch.einsum("bnd,no->bod", rst, self.ff_weights_decoder)
                
#                 mem_decode = (self.beta_decode * mem_out + (1.0 - self.beta_decode) * h3)
                
#                 mem_out = mem_decode

#             self.mem.append(mem)
#             self.curr.append(curr)
#             self.rst.append(rst)
#             self.mem_out.append(mem_out)#.detach()
#             if self.struct:
#                 self.mem_inner_dim.append( mem_inner_dim )

            
#         return self.rst, self.curr, self.mem, self.mem_out 


