import Foundation
import CoreML


let tokenizer = ProtTokenizer()
let example_input = "A E T C Z A O"
let encoded_input = tokenizer.tokenize(protSequence: example_input)
print(encoded_input)

