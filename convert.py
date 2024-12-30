import sys
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import coremltools as ct
from model import Model, MaskedLMModel, ModernBertRotaryEmbedding

"""
Convert a ModernBERT HuggingFace model to CoreML.
"""

torch.set_grad_enabled(False)

class ModelWrapper(nn.Module):
    def __init__(self, model: Model, max_length=8192):
        super().__init__()
        self.model: Model = model
        self.global_rope = ModernBertRotaryEmbedding(model.config.hidden_size // model.config.num_heads, model.config.global_rope_theta)
        self.local_rope = ModernBertRotaryEmbedding(model.config.hidden_size // model.config.num_heads, model.config.local_rope_theta)
        self.global_cos, self.global_sin = self.global_rope(torch.zeros((1,)), torch.arange(max_length).unsqueeze(0))
        self.local_cos, self.local_sin = self.local_rope(torch.zeros((1,)), torch.arange(max_length).unsqueeze(0))

        self.global_cos = self.global_cos.squeeze(0) # .T
        self.global_sin = self.global_sin.squeeze(0) # .T
        self.local_cos = self.local_cos.squeeze(0) # .T
        self.local_sin = self.local_sin.squeeze(0) # .T
    
    def forward(self, input_ids, sequence_length):
        mask = torch.full_like(input_ids, 1)
        mask_x = torch.cumsum(mask, dim=1) - 1
        local_sin = self.local_sin[mask_x].transpose(-1, -2)
        local_cos = self.local_cos[mask_x].transpose(-1, -2)
        global_sin = self.global_sin[mask_x].transpose(-1, -2)
        global_cos = self.global_cos[mask_x].transpose(-1, -2)
        mask_x = mask_x[:, None]
        distances = torch.abs(mask_x - torch.permute(mask_x, (0, 2, 1)))
        distances = distances <= (model.config.local_attention_window_size // 2)
        zeros = torch.zeros_like(distances, dtype=torch.float16)
        # Mask of over sequence length tokens is going to be all -inf, 
        # .softmax outputs NaNs for those positions, let's see if that causes issues
        global_attention_mask = torch.where((mask_x < sequence_length), zeros, -torch.inf)
        sliding_window_mask = torch.where((mask_x < sequence_length) & distances, zeros, -torch.inf)

        # return local_sin
        
        return self.model(
            input_ids,
            global_attention_mask,
            local_sin=local_sin,
            local_cos=local_cos,
            global_sin=global_sin,
            global_cos=global_cos,
            sliding_window_mask=global_attention_mask,
        )

model_name_or_path = "answerdotai/ModernBERT-base"
max_seq_len = 1024
if len(sys.argv) == 3:
    model_name_or_path = sys.argv[1]
    max_seq_len = int(sys.argv[2])
elif len(sys.argv) == 2 and sys.argv[1].isnumeric():
    max_seq_len = int(sys.argv[1])
elif len(sys.argv) == 2:
    model_name_or_path = sys.argv[1]
else:
    assert False, f"Usage: {sys.argv[0]} model_name_or_path [max_seq_len]"

print(f"Converting {model_name_or_path} to CoreML...")
model = MaskedLMModel.from_pretrained(model_name_or_path).eval()
# model.layers = model.layers[:4]
# model.rotate()
wmodel = ModelWrapper(model)

input_ids = torch.zeros( (1, max_seq_len), dtype=torch.int)
input_ids[..., :] = 50283 # PAD
seq = torch.tensor([50281,510,5347,273,6181,310,50284,15,50282], dtype=torch.int)
input_ids[..., :seq.shape[-1]] = seq
sequence_length = torch.tensor((10,), dtype=torch.int32)
# mask = torch.zeros((1,1,max_seq_len,max_seq_len))
# mask[:,:,seq.shape[-1]:,:] = -1e4
# mask[:,:,:,seq.shape[-1]:] = -1e4

output_name = "hidden_states" if isinstance(model, MaskedLMModel) else "logits"

input_shape = ct.EnumeratedShapes(shapes=[[1, 256], [1, 512], [1, 1024], [1, 2048]])

mlmodel= ct.convert(
    # torch.jit.trace(model, (input_ids, mask)),
    torch.jit.trace(wmodel, (input_ids, sequence_length)),
    inputs=[
        # ct.TensorType(name="input_ids", shape=input_ids.shape, dtype=np.int32),
        ct.TensorType(name="input_ids", shape=input_shape, dtype=np.int32),
        # ct.TensorType(name="mask", shape=mask.shape, dtype=np.float16, default_value=np.zeros_like(mask).astype(np.float16)),
        ct.TensorType(name="sequence_length", shape=sequence_length.shape, dtype=np.int32),
    ],
    outputs=[
        ct.TensorType(name=output_name),
    ],
    minimum_deployment_target=ct.target.iOS16,
    compute_precision=ct.precision.FLOAT16,
    # For initial prediction:
    compute_units=ct.ComputeUnit.CPU_AND_NE,
)
assert isinstance(mlmodel, ct.models.MLModel), "unexpected converted model type"

input_output_descriptions = {
    "input_ids": "Indices of input sequence tokens in the vocabulary",
    # "mask": "Mask for defining which tokens should attend to each other. 0 means attend and large negative number (e.g. -1e4) means do not attend.",
    "sequence_length": "Non padded number of tokens in input_ids",
    "hidden_states": "Raw outputs from the model. Typically further processed by a task-specific head.",
    "logits": "Un-normalized per-token predictions.",
}

for k in mlmodel.input_description:
    mlmodel.input_description[k] = input_output_descriptions[k]
for k in mlmodel.output_description:
    mlmodel.output_description[k] = input_output_descriptions[k]

mlmodel.user_defined_metadata["Source Model"] = model_name_or_path

mlmodel.save(f"{model_name_or_path.replace('/', '-')}-{max_seq_len}-NoRot.mlpackage")

model = MaskedLMModel.from_pretrained(model_name_or_path).eval() # Reload non-rotated model.
# coreml_out = torch.from_numpy(mlmodel.predict({"input_ids": input_ids.numpy(), "mask": mask.numpy()})[output_name])
coreml_out = torch.from_numpy(mlmodel.predict({"input_ids": input_ids.numpy(), "sequence_length": sequence_length.numpy()})[output_name])
# torch_out = model(input_ids, mask)
torch_out = wmodel(input_ids, sequence_length)
# Sometime useful for debugging.
# print("CoreML Top 4\n", coreml_out.topk(4, dim=1))
# print("Torch Top 4", torch_out.topk(4, dim=1))
# print("CoreML<>Torch max absolute difference:", (coreml_out - torch_out).abs().max())

kl = F.kl_div(F.log_softmax(coreml_out[...,:seq.shape[-1]], dim=1), F.log_softmax(torch_out[...,:seq.shape[-1]], dim=1), log_target=True, reduction='batchmean')
print("CoreML<>Torch KL divergence:", kl) # smaller is better
