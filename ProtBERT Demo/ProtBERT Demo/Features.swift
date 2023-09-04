//
//  FeatureMap.swift
//  ProtBERT Demo
//
//  Created by Martin Castro on 4/8/23.
//

import SwiftUI
import CoreML

struct Features: View {
    
    var result: MLMultiArray
    @EnvironmentObject var tokenizer: ProtTokenizer
    
    private let vocabSize = 30
    private let singleRow = [GridItem()]
    
    var body: some View{
        VStack(alignment: .leading) {
            VStack(alignment: .leading){
                Text("Features")
                    .font(.title)
                    .bold()
                Text("Sequence length: \(tokenizer.tokenCount - 2)")
            }
            .foregroundColor(.secondary)
            .padding()
            ScrollView(.horizontal){
                LazyHGrid(rows: singleRow) {
                    ForEach(Array(getTokenSlices(for: result, tokenCount: tokenizer.tokenCount).enumerated()), id: \.offset) { index, tokenSlice in
                        VStack{
                            Features2DVisual(values: Array(tokenSlice))
                            Text("\(String(tokenizer.tokenSequence[index]))")
                                .font(.headline)
                                .bold()
                            Text("Index: \(index)")
                        }
                    }
                }
            }.frame(minHeight: 280)
        }
    }
    
    private func convertToArray(from mlMultiArray: MLMultiArray) -> [Float32] {
        let length = mlMultiArray.count
        let floatPointer = mlMultiArray.dataPointer.bindMemory(to: Float32.self, capacity: length)
        let floatBuffer = UnsafeBufferPointer(start: floatPointer, count: length)
        let output = Array(floatBuffer)
        return output
    }
    
    private func getTokenSlices(for multiArray: MLMultiArray, tokenCount: Int) -> [ArraySlice<Float32>] {
        let flattenedArray = convertToArray(from: multiArray)
        
        var tokenSlices: [ArraySlice<Float32>] = []
        var startIndice = 0
        var endIndice = 29
        for _ in 0..<tokenCount {
            // Grab the appropriate slice of feature values (e.g., 0..29, 30..59, 60-99, ...)
            let slice = flattenedArray[startIndice...endIndice]
            tokenSlices.append(slice)
            startIndice += vocabSize
            endIndice += vocabSize
        }
        return tokenSlices
    }
}


struct Features2DVisual: View {
    
    let gridSize = CGSize(width: 30, height: 30)
    let spacing: CGFloat = 6
    let height = 6
    let width = 5
    let posLimit: Double = 1
    let negLimit: Double = -10
    
    let values: [Float32]
    
    var body: some View {
        Canvas { context, size in
            var x = 0
            var y = 0
            for i in 0..<(height*width){
                
                let origin = CGPoint(x: CGFloat(x)*(gridSize.width + spacing), y: CGFloat(y)*(gridSize.height + spacing))
                let rect = CGRect(origin: origin, size: gridSize)
                let rectPath = Path(rect)
                
                //Calculate cell color
                let value = values[i]
                if value >= 0 { // Positive, shade Green
                    let opacity = min(Double(value)/posLimit, 1)
                    let color = Color.green.opacity(opacity)
                    context.fill(rectPath, with: .color(color))
                }else if value < -12{ // Shade gray
                    let color = Color.gray.opacity(0.3)
                    context.fill(rectPath, with: .color(color))
                }else{ // Negative, shade red
                    let opacity = min(Double(value)/negLimit, 1)
                    let color = Color.red.opacity(Double(value)/negLimit)
                    context.fill(rectPath, with: .color(color))
                }
                
                let label = String(format: "%.1f", value)
                context.draw(Text(label).font(.caption), in: rect)
                context.stroke(rectPath, with: .color(.secondary), lineWidth: 1)
                
                // Increment position, L -> R, T -> B
                x += 1
                if x >= width {
                    x -= width
                    y += 1
                }
            }
        }
        .frame(width:180, height: 210)
        .padding()
    }
}

struct Feature2DVisial_Previews: PreviewProvider {

    static var previews: some View {
        let values: [Float32] = [-18, -18, -19, -24, -21, 2, 0, -3, -2, -2, 0, 0, -2, -2, -3, -1, -1, -1, -1, 0, 0, -6, -3, -4, -5, 0, -18, -19, -20, -20]
        Features2DVisual(values: values)
    }
}
