# **Logics-Parsing-VLM**

| ![1111](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/cBt0BJc308VHWf4JJibtJ.png) | ![2222](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/J0HqPp2h7xNoEsf4ll60a.png) |
|---|---|

## Overview

Logics-Parsing-VLM is a Gradio-based web application for parsing documents and images into structured HTML and Markdown formats using advanced Vision Language Models (VLMs). It supports PDF and image files, converting them into clean, logical representations while preserving elements like paragraphs, headings, tables, figures, and formulas. The app provides a user-friendly interface for uploading files, previewing pages, processing documents, and downloading results.

This repository hosts the source code for the demo application. For the original model and research, refer to the links below.

- Model Page: [Logics-Parsing on Hugging Face](https://huggingface.co/Logics-MLLM/Logics-Parsing)
- Original GitHub: [alibaba/Logics-Parsing](https://github.com/alibaba/Logics-Parsing)
- ArXiv Paper: [Logics-Parsing](https://arxiv.org/abs/2509.19760)

## Features

- Upload and process PDF or image files (PNG, JPEG, JPG).
- Preview individual pages with navigation controls.
- Select between two VLMs: Logics-Parsing and Gliese-OCR-7B-Post1.0.
- Generate structured HTML output for each page.
- Convert HTML to Markdown for rendering and source viewing.
- Download a single Markdown file containing all processed pages.
- Display processing time and handle errors gracefully.
- Example files support for quick testing (place samples in the `examples` directory).

## Requirements

- Python 3.8 or higher.
- CUDA-enabled GPU for optimal performance (falls back to CPU if unavailable).
- Dependencies listed in `requirements.txt` (create one based on the imports if needed).

Key libraries include:
- torch
- transformers
- gradio
- pillow (PIL)
- pymupdf (fitz)
- loguru
- html2text
- markdown
- click
- spaces (for GPU acceleration in Gradio)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/PRITHIVSAKTHIUR/Logics-Parsing-VLM.git
   cd Logics-Parsing-VLM
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   Note: If `requirements.txt` is not present, install the following:
   ```
   pip install torch transformers gradio pillow pymupdf loguru html2text markdown click spaces
   ```

3. (Optional) Create an `examples` directory and add sample PDF or image files for the examples section in the app.

## Usage

Run the application using the command-line interface:

```
python app.py
```

- The app will launch a local web server (typically at http://127.0.0.1:7860).
- Upload a PDF or image file.
- Select a model (default: Logics-Parsing).
- Click "Process" to parse all pages.
- Navigate pages using "Previous" and "Next" buttons.
- View results in tabs: Markdown Rendering, Markdown Source, Generated HTML.
- Download the full Markdown result and check processing time.

For debugging, use the `--debug` flag or modify the `demo.launch()` parameters.

## Models

- **Logics-Parsing**: A specialized VLM for document parsing, focusing on logical structure extraction.
- **Gliese-OCR-7B-Post1.0**: An OCR-focused model for high-accuracy text and structure recognition.

Models are loaded from Hugging Face Hub. Ensure internet access for the first run to download them.

## Limitations

- Processing large documents may require significant GPU memory.
- No internet access within the code execution environment for additional package installations.
- Unsupported file types will raise errors.
- Performance varies based on hardware; CPU mode is slower.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests for improvements, bug fixes, or new features.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- Built with Gradio for the UI.
- Utilizes Hugging Face Transformers for model handling.
- Thanks to the creators of [Logics-Parsing](https://huggingface.co/Logics-MLLM/Logics-Parsing) and [Gliese-OCR](https://huggingface.co/prithivMLmods/Gliese-OCR-7B-Post1.0) models.
