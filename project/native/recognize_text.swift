// Local image-to-text recognition. No network or generative rewriting.
import Foundation
import Vision

struct TextLine: Codable {
    let text: String
    let confidence: Float
    let x: Double
    let y: Double
    let width: Double
    let height: Double
}
struct TextResult: Codable {
    let provider: String
    let languageCorrection: Bool
    let lines: [TextLine]
}

do {
    guard CommandLine.arguments.count == 2 else {
        throw NSError(domain: "airboard", code: 2, userInfo: [NSLocalizedDescriptionKey: "Pass one PNG file"])
    }
    let imageURL = URL(fileURLWithPath: CommandLine.arguments[1])
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = false
    request.recognitionLanguages = ["en-US"]
    request.minimumTextHeight = 0.01
    let handler = VNImageRequestHandler(url: imageURL, options: [:])
    try handler.perform([request])
    let lines = (request.results ?? []).compactMap { observation -> TextLine? in
        guard let text = observation.topCandidates(1).first else { return nil }
        let box = observation.boundingBox
        return TextLine(text: text.string, confidence: text.confidence,
                        x: box.minX, y: box.minY, width: box.width, height: box.height)
    }
    let result = TextResult(provider: "Apple Vision local OCR", languageCorrection: false, lines: lines)
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    let output = try encoder.encode(result)
    FileHandle.standardOutput.write(output)
} catch {
    FileHandle.standardError.write(Data("Text recognition failed: \(error.localizedDescription)\n".utf8))
    exit(1)
}
