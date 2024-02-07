# ProtBERT +  Core ML


Set of scripts for optimizing and converting [ProtBERT](https://www.biorxiv.org/content/biorxiv/early/2020/07/21/2020.07.12.199554.full.pdf) -part of a family of pre-trained transformer models for protein sequences- to Apple’s [Core ML](https://developer.apple.com/documentation/coreml) format. ProtBERT was trained with a masked-language modeling objective across a corpus of over 217 million protein sequences, [UniRef100](https://www.uniprot.org/help/uniref). The output embeddings encode per-protein and per-residue features that can used as inputs for downstream proteomic tasks. 

## Environment Setup
You will need an environment with Python 3.8 and the following packages installed:

* PyTorch (used v. 2.0.0)
* Transformers (used v. 4.27.1)
* Core ML Tools (used v. 7.1)
 
 If you're on an Apple Silicon Mac, you can clone the Conda environment from the included `environment.yml` file.   

## Model Conversion
Once your environment is set up, just run the script `protBERT.py`. The script will:

1.	Download the pre-trained model and tokenizer from Hugging Face’s model hub, [Rostlab/prot_bert](https://huggingface.co/Rostlab/prot_bert)

2.	Instantiate an optimized version of the model from `BertANE.py` (see *Optimizations*)

3.	Remove the classifier head from both models to reveal their encoders' ouputs

4.	Trace the optimized model to get a *TorchScript* representation

5.  Convert the trace with Core ML Tools and save it to disk as a Core ML model package `ProtBERT.mlpackage`

5.	Load the model package from disk and verify its prediction against the base model

## Inference
Now you can incorporate the model package, tokenizer and vocab files (`ProtBERT.mlpackage`, `ProtTokenizer.swift`, `vocab.txt`) into your project. Xcode will automatically generate a class for your model package. The tokenizer has a vocab size of 30: 25 AAs -of which, U,Z,O, and B are mapped to X- and 5 model tokens. The tokenizer expects as input up to 510 uppercased, single-letter AA IDs. To extract features from a sequence:

```
let model = try! ProtBERT()
let tokenizer = ProtTokenizer()

let exampleInput = “MKSILDGLADTTFRTITTDLLYVGSNDIQYEDIKGDMASKLGYFPQKFPLTSFRGSPFQEKMTAGDNPQLVPADQVNITEFYNKSLSSFKENEENIQCGENFMDIECFMVLNPSQQLAIAVLSLTLGTFTVLENLLVLCVILHSRSLRCRPSYHFIGSLAVADLLGSVIFVYSFIDFHVFHRKDSRNVFLFKLGGVTASFTASVGSLFLTAIDRYISIHRPLAYKRIVTRPKAVVAFCLMWTIAIVIAVLPLLGWNCEKLQSVCSDIFPHIDETYLMFWIGVTSVLLLFIVYAYMYILWKAHSHAVRMIQRGTQKSIIIHTSEDGKVQVTRPDQARMDIRLAKTLVLILVVLIICWGPLLAIMVYDVFGKMNKLIKTVFAFCSMLCLLNSTVNPIIYALRSKDLRHAFRSMFPSCEGTAQPLDNSMGDSDCLHKHANNAASVHRAAESCIKSTVKIAKVTMSVSTDTSAEAL”

let encodedInput = tokenizer.tokenize(protSequence: exampleInput)

guard let output = try? model.prediction(input_ids: encodedInput, token_type_ids: MLShapedArray(scalars: Array(repeating: Int32(0), count: 512), shape:[1,512]), attention_mask: nil) else {
    fatalError("Unexpected runtime error.")
}
print(output.features)
```
The model’s output is a [MLMultiArray](https://developer.apple.com/documentation/coreml/mlmultiarray) of size [input_ids, hidden_size]. Please refer to the `ProtBERT Demo` project for more details on using the converted model.

## Optimizations
Following guidance from this [paper](https://machinelearning.apple.com/research/neural-engine-transformers), the following changes were made prior to conversion:

- Linear (dense) layers were replaced with their 2D convolution equivalent
- Layer normalization was replaced with an [optimized](https://github.com/apple/ml-ane-transformers/blob/main/ane_transformers/reference/layer_norm.py) equivalent
- Self-attention modules were replaced with an [optimized](https://github.com/apple/ml-ane-transformers/blob/main/ane_transformers/reference/multihead_attention.py) equivalent

The pre-trained weights are reshaped to match the layer changes by registering pre-hooks, per the paper. The optimized model package sees a significant reduction in memory usage, though latency is largely unaffected.

## GPU / ANE Support?
The model is currently entirely CPU-bound. As of macOS Sonoma 14.3, running the model in an Xcode project produces the following warning:
```
MLESEngine is not currently supported for models with range shape inputs that try to utilize the Neural Engine.
```

![Alt text](<ProtBERT Demo/ProtBERT Demo/Assets.xcassets/Features.imageset/Screenshot.png>)

