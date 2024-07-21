//
//  ContentView.swift
//  ProtBERT Demo
//
//  Created by Martin Castro on 4/5/23.
//

import SwiftUI
import CoreML

struct ContentView: View {
    
    let model = try! ProtBERT_FP16()
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
                        Text("ProtBERT was trained with a masked-langauge modeling objective across a corpus of over 217 million protein sequences.")
                            .font(.callout)
                            .padding(.bottom)
                        Text("Enter up to 510 uppercased amino acids to extract per-sequence and per-residue features")
                            .font(.caption)
                    }
                    .padding()
                    .frame(width: 350)
                    .foregroundColor(.secondary)
                            
                }
                .padding(10)
                
                TextEditor(text: $entry)
                    .textFieldStyle(RoundedBorderTextFieldStyle())
                    .frame(height: 80)
                
                Button {
                    let encodedInput = tokenizer.tokenize(protSequence: entry)
                    guard let output = try? model.prediction(input_ids: encodedInput) else {
                            fatalError("Unexpected runtime error.")
                    }
                    print(output.features.shape)
                    results.append(output.features)
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
