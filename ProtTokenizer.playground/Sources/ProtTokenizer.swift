//
//  ProtTokenizer.swift
//
//
//  Created by Martin Castro on 4/4/23.
//

import Foundation
import CoreML

public class ProtTokenizer {
    private var lookupDictionary: [Substring: Int] = [:]

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
        //Bridge to NSString to access the APIs for replacing substrings
        var cleanedString = protSequence as NSString
        cleanedString = cleanedString.replacingOccurrences(of: "U", with: "X") as NSString
        cleanedString = cleanedString.replacingOccurrences(of: "Z", with: "X") as NSString
        cleanedString = cleanedString.replacingOccurrences(of: "O", with: "X") as NSString
        cleanedString = cleanedString.replacingOccurrences(of: "B", with: "X") as NSString
        return cleanedString as String
    }

    private func tokenID(of token: Substring) -> Int {
        let unkownTokenID = 25 // Corresponds to X!
        return lookupDictionary[token] ?? unkownTokenID
    }

    public func tokenize(protSequence: String) -> MLMultiArray {
        let cleanedAAString = replaceUnkownAminoAcids(protSequence: protSequence)
        let aminoAcidSeq = cleanedAAString.split(separator: " ")

        var inputIDs: [Int] = [2] // Start with a [CLS] token id
        for aa in aminoAcidSeq {
            let tokenID = tokenID(of: aa)
            inputIDs.append(tokenID)
        }
        inputIDs.append(3) //End with a [SEP] token id
        
        guard let input = try? MLMultiArray(inputIDs) else {
            fatalError("Issue making the MLMultiArray :(")
        }
        return input
    }

}
