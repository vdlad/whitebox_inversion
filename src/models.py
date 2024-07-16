import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import types

def get_model_wrapper(model_id: str, maximum_layer: int = None, keep_head: bool = False, device="cuda"):
    if maximum_layer is not None:
        return truncate_model(model_id, maximum_layer, keep_head, device), \
            AutoTokenizer.from_pretrained(model_id)
    else:
        return AutoModelForCausalLM.from_pretrained(model_id).to(device=device), \
            AutoTokenizer.from_pretrained(model_id)

def truncate_model(model_id: str, maximum_layer: int, keep_head: bool = False, device="cuda"):
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        model.transformer.h = nn.ModuleList(model.transformer.h[:maximum_layer + 1])

        if hasattr(model.config, 'num_hidden_layers'):
            model.config.num_hidden_layers = maximum_layer + 1

        if not keep_head:
            if hasattr(model, 'lm_head'):
                delattr(model, 'lm_head')

            if hasattr(model.transformer, 'ln_f'):
                model.transformer.ln_f = nn.Identity(model.transformer.ln_f.normalized_shape)

        def new_forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, output_attentions=None, output_hidden_states=None, return_dict=None):
            outputs = self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            return outputs[0]

        model.forward = types.MethodType(new_forward, model)
    else:
        raise ValueError("Model structure not as expected. Please check the model architecture.")

    return model