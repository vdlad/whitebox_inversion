import torch.nn as nn
from transformers import AutoModelForCausalLM
import types

def get_model_wrapper(model_id: str, maximum_layer=None: int, keep_head=False: bool) -> nn.Module:
    if maximum_layer is not None:
        return truncate_model(model_id, maximum_layer, keep_head)
    else:
        return AutoModelForCausalLM.from_pretrained(model_id)

def truncate_model(model_id: str, maximum_layer: int, keep_head=False: bool) -> nn.Module:
    '''This truncates the LM up to the target layer, meaning
    that inference is faster. As such the output of model.generate
    will be the desired activation layer'''
    model = AutoModelForCausalLM.from_pretrained(model_id)
    
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        # Truncate the list of layers
        model.transformer.h = nn.ModuleList(model.transformer.h[:maximum_layer + 1])
        
        # Adjust the config to reflect the new number of layers
        if hasattr(model.config, 'num_hidden_layers'):
            model.config.num_hidden_layers = maximum_layer + 1
            
        if not keep_head:
             # Remove the language model head if desired
            if hasattr(model, 'lm_head'):
                delattr(model, 'lm_head')
                
            # The final layer norm seems important to smooth running
            # however, we can replace it with an identity function
            if hasattr(model.transformer, 'ln_f'):
                model.transformer.ln_f = nn.Identity(model.transformer.ln_f.normalized_shape)
        
        # Modify the forward method to return the output of the last transformer block
        def new_forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, output_attentions=None, output_hidden_states=None, return_dict=None):
            outputs = self.transformer(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            return outputs[0]  # Return the last hidden state directly
        
        model.forward = types.MethodType(new_forward, model)
    else:
        raise ValueError("Model structure not as expected. Please check the model architecture.")
    
    return model

