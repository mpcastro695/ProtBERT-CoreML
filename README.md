# ProtBERT +  Core ML
### Updated April 10th, 2023

A Python script for converting [ProtBERT](https://www.biorxiv.org/content/biorxiv/early/2020/07/21/2020.07.12.199554.full.pdf) to Apple’s [Core ML](https://developer.apple.com/documentation/coreml) format. ProtBERT was trained with a masked language modeling objective on a corpus of over 217 million protein sequences, [UniRef100](https://www.uniprot.org/help/uniref). The output embeddings encode per-protein and per-residue features that can used as inputs for downstream tasks. A sample project is included for demoing the converted model.

## Environment Setup
You’ll need an environment running Python 3.8 and the following packages installed:

	PyTorch (used v. 2.0.0)
	Transformers (used v. 4.27.1)
	Core ML Tools (used v. 6.3)
 
 If you're on an Apple Silicon Mac, you can clone the Conda environment from the included `environment.yml` file.   

## Model Conversion
Once your environment is set up, all you have to do is run the included python script `protBERT.py`. The script will:

1.	Download the ProtBERT model and tokenizer from Huggingface’s model hub [Rostlab/prot_bert](https://huggingface.co/Rostlab/prot_bert) and save the model to disk as a PyTorch model object `protBERT.pt`
2.	Encode an example input, run a forward pass on the PyTorch model, and print the output
3.	Trace the model to produce a ScriptFunction representation and save the traced model to disk `traced_protBERT.py`
4.	Convert the traced model using Core ML Tools and save it to disk as a model package `ProtBERT.mlpackage`
5.	Load the converted model package, run a forward pass, and print the output

## Inference
Now you can incorporate the model package, tokenizer and vocab files (`ProtBERT.mlpackage`, `ProtTokenizer.swift`, `vocab.txt`) into your Xcode project. Xcode will automatically generate a class for your model package. The tokenizer has a vocab size of 30: 25 AAs -of which, U,Z,O, and B are mapped to X- and 5 model tokens. The tokenizer expects as input up to 510 uppercased, single-letter AA IDs. To extract features from a sequence:

```
let model = try! ProtBERT()
let tokenizer = ProtTokenizer()

let exampleInput = “MKSILDGLADTTFRTITTDLLYVGSNDIQYEDIKGDMASKLGYFPQKFPLTSFRGSPFQEKMTAGDNPQLVPADQVNITEFYNKSLSSFKENEENIQCGENFMDIECFMVLNPSQQLAIAVLSLTLGTFTVLENLLVLCVILHSRSLRCRPSYHFIGSLAVADLLGSVIFVYSFIDFHVFHRKDSRNVFLFKLGGVTASFTASVGSLFLTAIDRYISIHRPLAYKRIVTRPKAVVAFCLMWTIAIVIAVLPLLGWNCEKLQSVCSDIFPHIDETYLMFWIGVTSVLLLFIVYAYMYILWKAHSHAVRMIQRGTQKSIIIHTSEDGKVQVTRPDQARMDIRLAKTLVLILVVLIICWGPLLAIMVYDVFGKMNKLIKTVFAFCSMLCLLNSTVNPIIYALRSKDLRHAFRSMFPSCEGTAQPLDNSMGDSDCLHKHANNAASVHRAAESCIKSTVKIAKVTMSVSTDTSAEAL”

let encodedInput = tokenizer.tokenize(protSequence: exampleInput)

guard let output = try? model.prediction(input_ids: encodedInput, token_type_ids: MLShapedArray(scalars: Array(repeating: Int32(0), count: 512), shape:[1,512]), attention_mask: nil) else {
    fatalError("Unexpected runtime error.")
}
print(output.prediction_scores)
```
The model’s output is of size [max_seq_length, vocab_size] (i.e., 512 x 30). The compute precision was set to 32-bit floating point during model conversion, thus the converted model runs almost entirely on the GPU. Setting the precision to 16-bit floating point and running the converted model on the ANE produces a buffer overflow error. Please refer to the ProtBERT Demo project for more details on using the converted model.

![Alt text](<ProtBERT Demo/ProtBERT Demo/Assets.xcassets/Welcome.imageset/Screenshot 2023-09-13 at 9.20.27 PM.png>)
![Alt text](<ProtBERT Demo/ProtBERT Demo/Assets.xcassets/Features.imageset/Screenshot 2023-07-15 at 4.38.22 PM.png>)

