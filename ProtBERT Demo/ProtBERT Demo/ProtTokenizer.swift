//
//  ProtTokenizer.swift
//
//
//  Created by Martin Castro on 4/4/23.
//

import Foundation
import CoreML

public class ProtTokenizer: ObservableObject {
    
    static let maxTokens = 512
    static let overheadTokens = 2
    private var lookupDictionary: [Substring: Int] = [:]
    
    @Published var aaSequence: [String] = []
    @Published var tokenSequence: [String] = []
    @Published var tokenCount = 0

    public init(){
        lookupDictionary = self.loadVocabulary()
    }

    private func loadVocabulary() -> [Substring: Int] {
        let fileName = "vocab"
        let expectedVocabularySize = 30

        guard let url = Bundle.main.url(forResource: fileName, withExtension: "txt") else {
            fatalError("Vocabulary file is missing")
        }
        guard let rawVocabulary = try? String(contentsOf: url) else {
            fatalError("Vocabulary file has no contents.")
        }
        let words = rawVocabulary.split(separator: "\n")
        guard words.count == expectedVocabularySize else {
            fatalError("Vocabulary file is not the correct size.")
        }
        let values = 0..<words.count
        let vocabulary = Dictionary(uniqueKeysWithValues: zip(words, values))
        return vocabulary
    }

    private func replaceUnkownAminoAcids(protSequence: String) -> String {
        //Bridge to NSString to access APIs for replacing substrings
        var cleanedString = protSequence.filter { !$0.isWhitespace } as NSString
        cleanedString = cleanedString.replacingOccurrences(of: "U", with: "X") as NSString
        cleanedString = cleanedString.replacingOccurrences(of: "Z", with: "X") as NSString
        cleanedString = cleanedString.replacingOccurrences(of: "O", with: "X") as NSString
        cleanedString = cleanedString.replacingOccurrences(of: "B", with: "X") as NSString
        return cleanedString as String
    }

    private func tokenID(of token: Substring) -> Int {
        let unkownTokenID = lookupDictionary[Substring("[UNK]")]!
        return lookupDictionary[token] ?? unkownTokenID
    }

    public func tokenize(protSequence: String) -> MLMultiArray {
        let cleanedAAString = replaceUnkownAminoAcids(protSequence: protSequence)
        var inputTokens = Array(cleanedAAString).map{String($0)}
        inputTokens = Array(inputTokens.prefix(ProtTokenizer.maxTokens - ProtTokenizer.overheadTokens))
        inputTokens.insert("[CLS]", at: 0)
        inputTokens.append("[SEP]")
        tokenSequence = inputTokens
        tokenCount = inputTokens.count
        
        // Get the token IDs
        var inputIDs: [Int32] = []
        for token in tokenSequence {
            let tokenID = tokenID(of: Substring(token))
            inputIDs.append(Int32(tokenID))
        }
        
//        // Fill the remaining token id slots with padding tokens
//        let padding = ProtTokenizer.maxTokens - inputIDs.count
//        inputIDs += Array(repeating: Int32(0), count: padding)

//        guard inputIDs.count <= ProtTokenizer.maxTokens else {
//            fatalError("`inputIDs` array size isn't the right size.")
//        }
        let encodedInput = MLMultiArray(MLShapedArray(scalars: inputIDs, shape: [1, inputIDs.count]))
        print(encodedInput.shape)
        return encodedInput
    }

}
