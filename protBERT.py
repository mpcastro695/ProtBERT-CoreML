import torch
import numpy as np
import coremltools as ct
from transformers import BertForMaskedLM, BertTokenizer
import re
import sys

# Downloads the model and tokenizer from Hugginface's model hub
tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False, torchscript=True)
model = BertForMaskedLM.from_pretrained("Rostlab/prot_bert", torchscript=True).eval()
torch.save(model, 'protBERT.pth')
print(model.modules)

# Encodes an example input
example_input = 'M V H L T P E E K S A V T A L W G K V N V D E V G G E A L G R L L V V Y P W T Q R F F E S F G D L S T P D A V M G N P K V K A H G K K V L G A F S D G L A H L D N L K G T F A T L S E L H C D K L H V D P E N F R L L G N V L V C V L A H H F G K E F T P P V Q A A Y Q K V V A G V A N A L A H K Y H'
example_input = re.sub(r"[UZOB]", "X", example_input)
encoded_input = tokenizer(example_input, return_tensors='pt')
print(encoded_input)

# Runs a forward pass
output = model(**encoded_input)
print(output)

# Traces the PyTorch model using dummy inputs
trace_input = {'input_ids': torch.randint(1, 20, (1, 512), dtype=torch.int64),
                'token_type_ids': torch.zeros(1, 512, dtype=torch.int64),
                'attention_mask': torch.zeros(1, 512, dtype=torch.int64)}
traced_model = torch.jit.trace(model, example_kwarg_inputs=trace_input)

# Converts the traced model
model = ct.convert(
     traced_model,
     convert_to="mlprogram",
     inputs=[ct.TensorType(name='input_ids', shape=ct.Shape(shape=(1, 512)), dtype=np.int32),
             ct.TensorType(name='token_type_ids', shape=ct.Shape(shape=(1, 512)), dtype=np.int32),
             ct.TensorType(name='attention_mask', shape=ct.Shape(shape=(1, 512)), dtype=np.int32)],
     outputs=[ct.TensorType(name='prediction_scores', dtype=np.float32)],
     compute_precision=ct.precision.FLOAT32
  )
# Sets attention_mask input as optional (Core ML gather layers don't use attention masks)
spec = model.get_spec()
attention_mask = spec.description.input[2]
attention_mask.type.isOptional = True
model = ct.models.model.MLModel(spec, weights_dir=model.weights_dir)

# Saves the CoreML model to disk :)
model.save("ProtBERT.mlpackage")

# Loads the CoreML model and runs a forward pass
model = ct.models.MLModel('ProtBERT.mlpackage')
max_Tokens = 512
inputSize = encoded_input["input_ids"].size(dim=1)
padded_input_ids = torch.nn.functional.pad(encoded_input["input_ids"], [0, max_Tokens-inputSize], "constant", 0)
padded_token_type_ids = torch.nn.functional.pad(encoded_input["token_type_ids"], [0, max_Tokens-inputSize], "constant", 0)
padded_attention_mask = torch.nn.functional.pad(encoded_input["token_type_ids"], [0, max_Tokens-inputSize], "constant", 0)

prediction = model.predict({"input_ids": padded_input_ids.type('torch.FloatTensor'), "token_type_ids": padded_token_type_ids.type('torch.FloatTensor'), "attention_mask": padded_attention_mask.type('torch.FloatTensor')})
np.set_printoptions(precision=4, threshold=sys.maxsize)
print(prediction.get('prediction_scores'))

