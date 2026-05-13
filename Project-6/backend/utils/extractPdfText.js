import pdf from "pdf-parse";

export async function extractPdfText(buffer) {
  try {
    const data = await pdf(buffer);
    const extractedText = data.text?.trim() || "";

    if (extractedText.length < 10) {
      throw new Error(
        "The PDF does not contain readable text. It may be scanned or image-based and would need OCR."
      );
    }

    return extractedText;
  } catch (error) {
    throw new Error(error.message || "Failed to extract text from the PDF.");
  }
}