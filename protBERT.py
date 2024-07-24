import torch
import numpy as np
import coremltools as ct
from transformers import BertForMaskedLM, BertTokenizer
import re

from BertANE import BertForMaskedLMANE

# 1.Download the pre-trained model and tokenizer from Hugging Face's model hub
tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert', do_lower_case=False, torchscript=True)
torch_model = BertForMaskedLM.from_pretrained('Rostlab/prot_bert', return_dict=False).eval()

# 2. Instantiate an optimized version of the model and restore the weights
optimized_model = BertForMaskedLMANE(torch_model.config).eval()
optimized_model.load_state_dict(torch_model.state_dict(), strict=True)

# 3. Remove the classifier head to reveal the encoder's output sized at [input_ids, hidden_size (i.e. 1024)]
torch_model = torch.nn.Sequential(*(list(torch_model.children())[:-1])).eval()
optimized_model = torch.nn.Sequential(*(list(optimized_model.children())[:-1])).eval()

# Optional: Append a custom layer that returns only the hidden states
class ActivationsOnly(torch.nn.Module):
    def forward(self, input_tuple):
        output = input_tuple[0]
        return output

# Optional: Append a custom layer that permutes the ouput back to BSC format
class BS1CtoBSC(torch.nn.Module):
    def forward(self, input):
        output = input.permute(0, 3, 1, 2)  # Swaps dimensions to [B, S, 1, C] format
        output = torch.squeeze(output)  # [B, S, C]
        return output

optimized_model = torch.nn.Sequential(*(list(optimized_model.children())), ActivationsOnly(), BS1CtoBSC()).eval()

# 4. Trace the optimized model using dummy inputs
trace_input = {'input': torch.randint(1, 20, (1, 512), dtype=torch.int64)}
traced_model = torch.jit.trace(optimized_model, example_kwarg_inputs=trace_input)

# 5. Convert the traced model to a FP32 model package and save it to disk
model_f32 = ct.convert(
     traced_model,
     convert_to='mlprogram',
     inputs=[ct.TensorType(name='input_ids', shape=ct.Shape(shape=(1, 512)), dtype=np.int32)],
     outputs=[ct.TensorType(name='features', dtype=np.float32)],
     compute_precision=ct.precision.FLOAT32)

model_f32.save('ProtBERT_FP32.mlpackage')

# Convert the traced model to a FP16 model package and save it to disk
model_f16 = ct.convert(
     traced_model,
     convert_to='mlprogram',
     inputs=[ct.TensorType(name='input_ids', shape=ct.Shape(shape=(1, 512)), dtype=np.int32)],
     outputs=[ct.TensorType(name='features', dtype=np.float32)],
     compute_precision=ct.precision.FLOAT16)

model_f16.save('ProtBERT_FP16.mlpackage')

# 6. Load the Core ML model and verify that its prediction matches that of the original PyTorch model
example_input = 'M V H L T P E E K S A V T A L W G K V N V D E V G G E A L G R L L V V Y P W T Q R F F E S F G D L S T P D A V M G N P K V K A H G K K V L G A F S D G L A H L D N L K G T F A T L S E L H C D K L H V D P E N F R L L G N V L V C V L A H H F G K E F T P P V Q A A Y Q K V V A G V A N A L A H K Y H'
example_input = re.sub(r'[UZOB]', 'X', example_input)
encoded_input = tokenizer(example_input, return_tensors='pt', max_length=512, padding="max_length")

model_f32 = ct.models.MLModel('ProtBERT_FP32.mlpackage')
prediction_f32 = model_f32.predict({'input_ids': encoded_input['input_ids'].type('torch.FloatTensor')})
core32_tensor = prediction_f32.get('features')
core32_tensor = np.expand_dims(core32_tensor, axis=0)

model_f16 = ct.models.MLModel('ProtBERT_FP16.mlpackage')
prediction_f16 = model_f16.predict({'input_ids': encoded_input['input_ids'].type('torch.FloatTensor')})
core16_tensor = prediction_f16.get('features')
core16_tensor = np.expand_dims(core16_tensor, axis=0)

with torch.no_grad():
    torch_output = torch_model(encoded_input['input_ids'])
torch_tensor = torch_output[0].detach().cpu().numpy() if torch_output[0].requires_grad else torch_output[0].cpu().numpy()

relTolerance = 1e-02
absTolerance = 1e-05
np.testing.assert_allclose(core32_tensor, torch_tensor, relTolerance, absTolerance)
print('Congrats on the new FP32 model!')

relTolerance = 1e-01
absTolerance = 8e-02
np.testing.assert_allclose(core16_tensor, torch_tensor, relTolerance, absTolerance)
print('Congrats on the new FP16 model!')




