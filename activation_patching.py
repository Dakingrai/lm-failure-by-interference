class StopForward(Exception):
    pass

class InterveneMLPNeuron:
    def __init__(self, 
                 model, 
                 intervene_neurons,
                 coeff,
                 stop=False, 
                 verbose=False) -> None:
        self.model = model
        self.intervene_neurons = intervene_neurons
        self.verbose = False
        self.coeff = coeff
        self.hooks = []
        self.model.eval()

        def get_hook(neuron):
            # output dims: [batch, token_len, n_attn_heads, d_model/n_attn_heads]
            def hook(module, input, output):
                output[:, :, neuron] = self.coeff * output[:, :, neuron] 
                return output
            return hook
            
        for layer, neuron in self.intervene_neurons:
            hook = self.model.blocks[layer].mlp.hook_post.register_forward_hook(get_hook(neuron))
            self.hooks.append(hook)

    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()
        
    def close(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

class InterveneOV_Neuron:
    def __init__(self, 
                 model, 
                 intervene_heads,
                 intervene_neurons,
                 coeff,
                 stop=False, 
                 verbose=False) -> None:
        self.model = model
        self.intervene_heads = intervene_heads
        self.intervene_neurons = intervene_neurons
        self.verbose = False
        self.coeff = coeff
        self.hooks = []
        self.model.eval()

        def get_head_hook(head):
            # output dims: [batch, token_len, n_attn_heads, d_model/n_attn_heads]
            def hook(module, input, output):
                output[:, :, head, :] = self.coeff * output[:, :, head, :] #1.3 for 4-paren
                return output
            return hook
        
        def get_neuron_hook(neuron):
            # output dims: [batch, token_len, n_attn_heads, d_model/n_attn_heads]
            def hook(module, input, output):
                output[:, :, neuron] = self.coeff * output[:, :, neuron] 
                return output
            return hook
        
        if self.intervene_heads:
            for layer, head in self.intervene_heads:
                hook = self.model.blocks[layer].attn.hook_z.register_forward_hook(get_head_hook(head))
                self.hooks.append(hook)

        if self.intervene_neurons:
            for layer, neuron in self.intervene_neurons:
                hook = self.model.blocks[layer].mlp.hook_post.register_forward_hook(get_neuron_hook(neuron))
                self.hooks.append(hook)

    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()
        
    def close(self):
        for hook in self.hooks:
            hook.remove()

class InterveneOV:
    def __init__(self, 
                 model, 
                 intervene_heads,
                 coeff,
                 stop=False, 
                 verbose=False) -> None:
        self.model = model
        self.intervene_heads = intervene_heads
        self.verbose = False
        self.coeff = coeff
        self.hooks = []
        self.model.eval()

        def get_hook(layer, head):
            # output dims: [batch, token_len, n_attn_heads, d_model/n_attn_heads]
            def hook(module, input, output):
                output[:, :, head, :] = self.coeff * output[:, :, head, :] #1.3 for 4-paren
                return output
            return hook
            
        for layer, head in self.intervene_heads:
            self.hooks.append(self.model.blocks[layer].attn.hook_z.register_forward_hook(get_hook(layer, head)))

    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()
        
    def close(self):
        for hook in self.hooks:
            hook.remove()

class InterveneNeurons:
    def __init__(self, 
                 model, 
                 intervene_neurons,
                 coeff,
                 stop=False, 
                 verbose=False) -> None:
        self.model = model
        self.intervene_neurons = intervene_neurons
        self.verbose = False
        self.coeff = coeff
        self.hooks = []
        self.model.eval()

        def get_hook(layer, neuron):
            # output dims: [batch, token_len, n_attn_heads, d_model/n_attn_heads]
            def hook(module, input, output):
                output[:, :, neuron] = self.coeff * output[:, :, neuron] 
                return output
            return hook
            
        for neuron in self.intervene_neurons:
            l = int(neuron.split("N")[0].split("L")[1])
            n = int(neuron.split("N")[1])
            # intervene on the first layer output
            self.hooks.append(self.model.blocks[l].mlp.hook_pre.register_forward_hook(get_hook(l, n)))

    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()
        
    def close(self):
        for hook in self.hooks:
            hook.remove()