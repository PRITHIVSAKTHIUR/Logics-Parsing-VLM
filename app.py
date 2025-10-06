import os
import hashlib
import spaces
import re
import time
import click
import gradio as gr
from io import BytesIO
from PIL import Image
from loguru import logger
from pathlib import Path
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from transformers.image_utils import load_image
import fitz
import html2text
import markdown
import tempfile
from typing import Optional, Tuple, Dict, Any, List

pdf_suffixes = [".pdf"]
image_suffixes = [".png", ".jpeg", ".jpg"]
device = "cuda" if torch.cuda.is_available() else "cpu"

logger.info(f"Using device: {device}")

# Model 1: Logics-Parsing
MODEL_ID_1 = "Logics-MLLM/Logics-Parsing"
logger.info(f"Loading model 1: {MODEL_ID_1}")
processor_1 = AutoProcessor.from_pretrained(MODEL_ID_1, trust_remote_code=True)
model_1 = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID_1,
    trust_remote_code=True,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device).eval()
logger.info(f"Model '{MODEL_ID_1}' loaded successfully.")

# Model 2: Gliese-OCR-7B-Post1.0
MODEL_ID_2 = "prithivMLmods/Gliese-OCR-7B-Post1.0"
logger.info(f"Loading model 2: {MODEL_ID_2}")
processor_2 = AutoProcessor.from_pretrained(MODEL_ID_2, trust_remote_code=True)
model_2 = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID_2,
    trust_remote_code=True,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device).eval()
logger.info(f"Model '{MODEL_ID_2}' loaded successfully.")

# Model 3: olmOCR-7B-0825
MODEL_ID_3 = "allenai/olmOCR-7B-0825"
logger.info(f"Loading model 3: {MODEL_ID_3}")
processor_3 = AutoProcessor.from_pretrained(MODEL_ID_3, trust_remote_code=True)
model_3 = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID_3,
    trust_remote_code=True,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device).eval()
logger.info(f"Model '{MODEL_ID_3}' loaded successfully.")

@spaces.GPU
def parse_page(image: Image.Image, model_name: str) -> str:
    """
    Parses a single document page image using the selected model.
    """
    if model_name == "Logics-Parsing":
        current_processor, current_model = processor_1, model_1
    elif model_name == "Gliese-OCR-7B-Post1.0":
        current_processor, current_model = processor_2, model_2
    elif model_name == "olmOCR-7B-0825":
        current_processor, current_model = processor_3, model_3
    else:
        raise ValueError(f"Unknown model choice: {model_name}")

    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": "Parse this document page into a clean, structured HTML representation. Preserve the logical structure with appropriate tags for content blocks such as paragraphs (<p>), headings (<h1>-<h6>), tables (<table>), figures (<figure>), formulas (<formula>), and others. Include category tags, and filter out irrelevant elements like headers and footers."}]}]
    prompt_full = current_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = current_processor(text=[prompt_full], images=[image], return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        generated_ids = current_model.generate(**inputs, max_new_tokens=2048, temperature=0.1, top_p=0.9, do_sample=True, repetition_penalty=1.05)
    
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = current_processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return output_text

def convert_file_to_images(file_path: str, dpi: int = 200) -> List[Image.Image]:
    """
    Converts a PDF or image file into a list of PIL Images.
    """
    images = []
    file_ext = Path(file_path).suffix.lower()
    
    if file_ext in image_suffixes:
        images.append(Image.open(file_path).convert("RGB"))
        return images
        
    if file_ext not in pdf_suffixes:
        raise ValueError(f"Unsupported file type: {file_ext}")

    try:
        pdf_document = fitz.open(file_path)
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            images.append(Image.open(BytesIO(img_data)))
        pdf_document.close()
    except Exception as e:
        logger.error(f"Failed to convert PDF using PyMuPDF: {e}")
        raise
    return images

def get_initial_state() -> Dict[str, Any]:
    """Returns the default initial state for the application."""
    return {"pages": [], "total_pages": 0, "current_page_index": 0, "page_results": []}

def load_and_preview_file(file_path: Optional[str]) -> Tuple[Optional[Image.Image], str, Dict[str, Any]]:
    """
    Loads a file, converts all pages to images, and stores them in the state.
    """
    state = get_initial_state()
    if not file_path:
        return None, '<div class="page-info">No file loaded</div>', state

    try:
        pages = convert_file_to_images(file_path)
        if not pages:
            return None, '<div class="page-info">Could not load file</div>', state
        
        state["pages"] = pages
        state["total_pages"] = len(pages)
        page_info_html = f'<div class="page-info">Page 1 / {state["total_pages"]}</div>'
        return pages[0], page_info_html, state
    except Exception as e:
        logger.error(f"Failed to load and preview file: {e}")
        return None, '<div class="page-info">Failed to load preview</div>', state

async def process_all_pages(state: Dict[str, Any], model_choice: str):
    """
    Processes all pages stored in the state and updates the state with results.
    """
    if not state or not state["pages"]:
        error_msg = "<h3>Please upload a file first.</h3>"
        return error_msg, "", "", None, "Error: No file to process", state

    logger.info(f'Processing {state["total_pages"]} pages with model: {model_choice}')
    start_time = time.time()
    
    try:
        page_results = []
        for i, page_img in enumerate(state["pages"]):
            logger.info(f"Parsing page {i + 1}/{state['total_pages']}")
            html_result = parse_page(page_img, model_choice)
            page_results.append({'raw_html': html_result})
        
        state["page_results"] = page_results
        
        # Create a single markdown file for download with all content
        full_html_content = "\n\n".join([f'<!-- Page {i+1} -->\n{res["raw_html"]}' for i, res in enumerate(page_results)])
        full_markdown = html2text.html2text(full_html_content)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
            f.write(full_markdown)
            md_path = f.name
            
        parsing_time = time.time() - start_time
        cost_time_str = f'Total processing time: {parsing_time:.2f}s'
        
        # Display the results for the current page
        current_page_results = get_page_outputs(state)
        
        return *current_page_results, md_path, cost_time_str, state

    except Exception as e:
        logger.error(f"Parsing failed: {e}", exc_info=True)
        error_html = f"<h3>An error occurred during processing:</h3><p>{str(e)}</p>"
        return error_html, "", "", None, f"Error: {str(e)}", state

def navigate_page(direction: str, state: Dict[str, Any]):
    """
    Navigates to the previous or next page and updates the UI accordingly.
    """
    if not state or not state["pages"]:
        return None, '<div class="page-info">No file loaded</div>', *get_page_outputs(state), state

    current_index = state["current_page_index"]
    total_pages = state["total_pages"]
    
    if direction == "prev":
        new_index = max(0, current_index - 1)
    elif direction == "next":
        new_index = min(total_pages - 1, current_index + 1)
    else:
        new_index = current_index
        
    state["current_page_index"] = new_index
    
    image_preview = state["pages"][new_index]
    page_info_html = f'<div class="page-info">Page {new_index + 1} / {total_pages}</div>'
    
    page_outputs = get_page_outputs(state)
    
    return image_preview, page_info_html, *page_outputs, state

def get_page_outputs(state: Dict[str, Any]) -> Tuple[str, str, str]:
    """Helper to get displayable outputs for the current page."""
    if not state or not state.get("page_results"):
        return "<h3>Process the document to see results.</h3>", "", ""

    index = state["current_page_index"]
    result = state["page_results"][index]
    raw_html = result['raw_html']
    
    mmd_source = html2text.html2text(raw_html)
    mmd_render = markdown.markdown(mmd_source, extensions=['fenced_code', 'tables'])
    
    return mmd_render, mmd_source, raw_html

def clear_all():
    """Clears all UI components and resets the state."""
    return (
        None,
        None,
        "<h3>Results will be displayed here after processing.</h3>",
        "",
        "",
        None,
        "",
        '<div class="page-info">No file loaded</div>',
        get_initial_state()
    )

@click.command()
def main():
    """
    Sets up and launches the Gradio user interface for the Logics-Parsing app.
    """
    css = """
    .main-container { max-width: 1400px; margin: 0 auto; }
    .header-text { text-align: center; color: #2c3e50; margin-bottom: 20px; }
    .process-button { border: none !important; color: white !important; font-weight: bold !important; background-color: blue !important;}
    .process-button:hover { background-color: darkblue !important; transform: translateY(-2px) !important; box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important; }
    .page-info { text-align: center; padding: 8px 16px; border-radius: 20px; font-weight: bold; margin: 10px 0; }
    """
    with gr.Blocks(theme="bethecloud/storj_theme", css=css, title="Logics-Parsing Demo") as demo:
        app_state = gr.State(value=get_initial_state())

        gr.HTML("""
        <div class="header-text">
            <h1>📄 Multimodal: VLM Parsing</h1>
            <p style="font-size: 1.1em; color: #6b7280;">An advanced Vision Language Model to parse documents and images into clean Markdown(.md)</p>
            <div style="display: flex; justify-content: center; gap: 20px; margin: 15px 0;">
                <a href="https://huggingface.co/collections/prithivMLmods/mm-vlm-parsing-68e33e52bfb9ae60b50602dc" target="_blank" style="text-decoration: none; color: #2563eb; font-weight: 500;">🤗 Model Info</a>
                <a href="https://github.com/PRITHIVSAKTHIUR/VLM-Parsing" target="_blank" style="text-decoration: none; color: #2563eb; font-weight: 500;">💻 GitHub</a>
                <a href="https://huggingface.co/models?pipeline_tag=image-text-to-text&sort=trending" target="_blank" style="text-decoration: none; color: #2563eb; font-weight: 500;">📝 Multimodal VLMs</a>
            </div>
        </div>
        """)

        with gr.Row(elem_classes=["main-container"]):
            with gr.Column(scale=1):
                model_choice = gr.Dropdown(choices=["Logics-Parsing", "Gliese-OCR-7B-Post1.0", "olmOCR-7B-0825"], label="Select Model⚡️", value="Logics-Parsing")
                file_input = gr.File(label="Upload PDF or Image", file_types=[".pdf", ".jpg", ".jpeg", ".png"], type="filepath")
                image_preview = gr.Image(label="Preview", type="pil", interactive=False, height=280)
                
                with gr.Row():
                    prev_page_btn = gr.Button("◀ Previous", size="md")
                    page_info = gr.HTML('<div class="page-info">No file loaded</div>')
                    next_page_btn = gr.Button("Next ▶", size="md")

                example_root = "examples"
                if os.path.exists(example_root) and os.path.isdir(example_root):
                    example_files = [os.path.join(example_root, f) for f in os.listdir(example_root) if f.endswith(tuple(pdf_suffixes + image_suffixes))]
                    if example_files:
                        #with gr.Accordion("Open Examples⚙️", open=False):
                        #with gr.row():
                        gr.Examples(examples=example_files, inputs=file_input, examples_per_page=10)

                with gr.Accordion("Download Details🕧", open=False):
                    output_file = gr.File(label='Download Markdown Result', interactive=False)
                    cost_time = gr.Text(label='Time Cost', interactive=False)

                process_btn = gr.Button("🚀 Process", variant="primary", elem_classes=["process-button"], size="lg")
                clear_btn = gr.Button("🗑️ Clear All", variant="secondary")

            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("Markdown Rendering"):
                        mmd_html = gr.TextArea(lines=27, label='Markdown Rendering', show_copy_button=True, interactive=True)
                    with gr.Tab("Markdown Source"):
                        mmd = gr.TextArea(lines=27, show_copy_button=True, label="Markdown Source", interactive=True)
                    with gr.Tab("Generated HTML"):
                        raw_html = gr.TextArea(lines=27, show_copy_button=True, label="Generated HTML")

        # --- Event Handlers ---
        file_input.change(
            fn=load_and_preview_file, 
            inputs=file_input, 
            outputs=[image_preview, page_info, app_state],
            show_progress="full")
        
        process_btn.click(
            fn=process_all_pages, 
            inputs=[app_state, model_choice], 
            outputs=[mmd_html, mmd, raw_html, 
            output_file, cost_time, app_state], 
            concurrency_limit=15, 
            show_progress="full")

        prev_page_btn.click(
            fn=lambda s: navigate_page("prev", s), 
            inputs=app_state, outputs=[image_preview, 
            page_info, mmd_html, mmd, raw_html, app_state])
        
        next_page_btn.click(
            fn=lambda s: navigate_page("next", s), 
            inputs=app_state, outputs=[image_preview, 
            page_info, mmd_html, mmd, raw_html, app_state])

        clear_btn.click(
            fn=clear_all, 
            outputs=[file_input, image_preview, mmd_html, mmd, raw_html, 
                     output_file, cost_time, page_info, app_state])
        
    demo.queue().launch(debug=True, share=True, mcp_server=True, ssr_mode=False, show_error=True)

if __name__ == '__main__':
    if not os.path.exists("examples"):
        os.makedirs("examples")
        logger.info("Created 'examples' directory. Please add some sample PDF/image files there.")
    main()
