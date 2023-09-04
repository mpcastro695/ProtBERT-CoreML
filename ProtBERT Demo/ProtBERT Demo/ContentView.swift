//
//  ContentView.swift
//  ProtBERT Demo
//
//  Created by Martin Castro on 4/5/23.
//

import SwiftUI
import CoreML

struct ContentView: View {
    
    let model = try! ProtBERT()
    @StateObject var tokenizer = ProtTokenizer()
    
    @State private var entry = String()
    @State private var results: [MLMultiArray] = []
    
    var body: some View {
        NavigationStack(path: $results) {
            VStack {
                HStack{
                    ZStack(alignment: .center){
                        Image("SecondaryStructures")
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .frame(maxWidth: 250)
                    }.frame(maxWidth: .infinity, maxHeight: .infinity)
                    Divider()
                    VStack(alignment: .leading){
                        Text("ProtBERT +  Core ML")
                            .font(.largeTitle)
                            .bold()
                            .padding(.bottom, 5)
                        Text("ProtBERT was pre-trained to predict randomly masked amino acids (AAs) across a corpus of over 217 million protein sequences. Enter a string of AA IDs to predict per-residue features.")
                            .font(.callout)
                            .padding(.bottom)
                        Text("\(Image(systemName: "exclamationmark.triangle")) The converted Model Package is experimental and its results not yet validated.")
                            .font(.caption2)
                            .padding(.bottom, 5)
                        Text("Martin C. 2023")
                            .font(.caption)
                        Text("martincastro521@gmail.com")
                            .font(.caption)
                    }
                    .padding()
                    .frame(width: 350)
                    .foregroundColor(.secondary)
                            
                }
                .padding(10)
                
                TextField(text: $entry, prompt: Text("Enter up to 510 uppercase, single-letter AA ids")) {
                    Text("Sequence")
                }
                .textFieldStyle(RoundedBorderTextFieldStyle())
                
                Button {
                    let encodedInput = tokenizer.tokenize(protSequence: entry)
                    guard let output = try? model.prediction(input_ids: encodedInput, token_type_ids: MLShapedArray(scalars: Array(repeating: Int32(0), count: 512), shape:[1,512]), attention_mask: nil) else {
                            fatalError("Unexpected runtime error.")
                    }
                    results.append(output.prediction_scores)
                } label: {
                    Text("Extract Features")
                }

            }
            .padding()
            .navigationDestination(for: MLMultiArray.self) { result in
                Features(result: result)
            }
        }
        .frame(minWidth: 600, minHeight: 400)
        .environmentObject(tokenizer)
    }
    
    func convertToArray(from mlMultiArray: MLMultiArray) -> [Int32] {
        let length = mlMultiArray.count
        let intPointer = mlMultiArray.dataPointer.bindMemory(to: Int32.self, capacity: length)
        let intBuffer = UnsafeBufferPointer(start: intPointer, count: length)
        let output = Array(intBuffer)
        return output
    }
    
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
