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
    private var vocab: [Substring: Int] = [:]
    
    @Published var aaSequence: [String] = []
    @Published var tokenSequence: [String] = []
    @Published var tokenCount = 0

    public init(){
        vocab = self.loadVocabulary()
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

    private func getTokenID(of token: Substring) -> Int {
        let unkownTokenID = vocab[Substring("[UNK]")]!
        return vocab[token] ?? unkownTokenID
    }

    public func tokenize(protSequence: String) -> MLMultiArray {
        let cleanedAAString = replaceUnkownAminoAcids(protSequence: protSequence)
        var tokens = Array(cleanedAAString).map{String($0)}
        tokens = Array(tokens.prefix(ProtTokenizer.maxTokens - ProtTokenizer.overheadTokens))
        tokens.insert("[CLS]", at: 0)
        
        tokenSequence = tokens
        tokenCount = tokens.count
        
        tokens.append("[SEP]")
        // Fill the remaining slots with padding tokens
        let padding = ProtTokenizer.maxTokens - tokens.count
        tokens += Array(repeating: "[PAD]", count: padding)
        
        let tokenIDs = tokens.compactMap{vocab[Substring($0)]}

        return MLMultiArray(MLShapedArray(scalars: tokenIDs.compactMap{Int32($0)}, shape: [1, ProtTokenizer.maxTokens]))
    }

}
