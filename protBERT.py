import torch
import numpy as np
import coremltools as ct
# import coremltools.optimize.coreml as cto
from transformers import BertForMaskedLM, BertTokenizer
import re

from BertANE import BertForMaskedLMANE

# 1.Download the pre-trained model and tokenizer from Hugging Face's model hub
tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert', do_lower_case=False, torchscript=True)
torch_model = BertForMaskedLM.from_pretrained('Rostlab/prot_bert', return_dict=False).eval()

# 2. Instantiate an optimized version of the model and restore the weights
optimized_model = BertForMaskedLMANE(torch_model.config).eval()
optimized_model.load_state_dict(torch_model.state_dict(), strict=True)

# 3. Remove the final dropout and pooler layers to reveal the last hidden layer sized at [max_Tokens, features(i.e. 768)]
torch_model = torch.nn.Sequential(*(list(torch_model.children())[:-1])).eval()
print(torch_model.modules)
optimized_model = torch.nn.Sequential(*(list(optimized_model.children())[:-1])).eval()
print(optimized_model.modules)

# Optional: Append a custom layer that returns only the hidden states
class ActivationsOnly(torch.nn.Module):
    def forward(self, input_tuple):
        output = input_tuple[0]
        return output

# Optional: Append a custom layer that permutes the ouput
class BS1CtoBSC(torch.nn.Module):
    def forward(self, input):
        output = input.permute(0, 3, 1, 2)  # Swaps dimensions to [B, S, 1, C] format
        output = torch.squeeze(output)  # [B, S, C]
        return output

optimized_model = torch.nn.Sequential(*(list(optimized_model.children())), ActivationsOnly(), BS1CtoBSC()).eval()

# 4. Trace the optimized model using dummy inputs
trace_input = {'input': torch.randint(1, 20, (1, 512), dtype=torch.int64)}
traced_model = torch.jit.trace(optimized_model, example_kwarg_inputs=trace_input)

# 5. Convert the traced model and save it to disk
model = ct.convert(
     traced_model,
     convert_to='mlprogram',
     inputs=[ct.TensorType(name='input_ids', shape=ct.Shape(shape=(1, ct.RangeDim(lower_bound=1, upper_bound=512, default=256))), dtype=np.int32)],
     outputs=[ct.TensorType(name='features', dtype=np.float32)],
     compute_precision=ct.precision.FLOAT32)

model.save('ProtBERT.mlpackage')

# # TODO: Load the Core ML model, quantize weights
# model = ct.models.MLModel('ProtBERT.mlpackage')
# op_config = cto.OpLinearQuantizerConfig(mode='linear_symmetric', dtype=np.int8)
# config = cto.OptimizationConfig(global_config=op_config)
# compressed_model = ct.optimize.coreml.linear_quantize_weights(model, config)
# model.save('ProtBERT.mlpackage')

# 6. Load the Core ML model and verify that its prediction matches that of the original PyTorch model
relTolerance = 1e-04
absTolerance = 1e-03

example_input = 'M V H L T P E E K S A V T A L W G K V N V D E V G G E A L G R L L V V Y P W T Q R F F E S F G D L S T P D A V M G N P K V K A H G K K V L G A F S D G L A H L D N L K G T F A T L S E L H C D K L H V D P E N F R L L G N V L V C V L A H H F G K E F T P P V Q A A Y Q K V V A G V A N A L A H K Y H'
example_input = re.sub(r'[UZOB]', 'X', example_input)
encoded_input = tokenizer(example_input, return_tensors='pt')

model = ct.models.MLModel('ProtBERT.mlpackage')
coreMLOutput = model.predict({'input_ids': encoded_input['input_ids'].type('torch.FloatTensor')})
coreMLTensor = coreMLOutput.get('features')
coreMLTensor = np.expand_dims(coreMLTensor, axis=0)

with torch.no_grad():
    torchOutput = torch_model(encoded_input['input_ids'])
torchTensor = torchOutput[0].detach().cpu().numpy() if torchOutput[0].requires_grad else torchOutput[0].cpu().numpy()

np.testing.assert_allclose(coreMLTensor, torchTensor, relTolerance, absTolerance)
print('Congrats on the new model!')



