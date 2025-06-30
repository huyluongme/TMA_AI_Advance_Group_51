import gradio as gr
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from openai import OpenAI
import io
import os
from invoke_model import predict_image
from PIL import Image


load_dotenv()

# Access variables

ai_model = os.getenv("OPENAI_MODEL")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Constants
EXAMPLE_IMAGE_PATH = "./Example_Images/Example_Image.png"  # Path to an example image for development
IMAGE_SIZE = (256, 256)  # Size to resize images for the model
GUIDE_LINES = f"""
Upload an image of a diseased plant leaf. The image should be clear and focused on the leaf. The model works best with images size are {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]} pixels. 
The model will predict the disease and provide treatment information.
If the model is not confident, it will return "Unknown" and suggest trying another image.
You can click "What is the disease?" to get more information about the detected disease.
Use the "Save Report" button to download a report with the prediction and explanation.
Clear the inputs using the "Clear" button to start over.
"""

UPPER_THESHOLD = 80  # Confidence threshold for predictions
MIDDLE_THESHOLD = 50  # Confidence threshold for predictions


def resize_image(image):
    """
    Resize the image to the required size for the model.
    """

    if isinstance(image, Image.Image):
        return image.resize(IMAGE_SIZE)
    return image 

def reset_response_boxes():
    return (
        gr.update(value="", visible=True),
        gr.update(value="", visible=False)
    )

def get_disease_info(disease):
    if not disease:
        return "❗ No disease to explain."
    
    question = f"What is {disease} and how to treat it in plants?"
    if not ai_model:
        return "⚠️ OpenAI model not configured. Please set the OPENAI_MODEL environment variable."
    if not client:
        return "⚠️ OpenAI client not initialized. Please check your API key."
    if not question:
        return "❗ No disease name provided for explanation."

    try:
        response = client.chat.completions.create(
            model=ai_model,
            messages=[{"role": "user", "content": question}]
        )
        content = response.choices[0].message.content
        return (
            gr.update(value=content, visible=False),
            gr.update(value=content, visible=True)
        )
    except Exception as e:
        return f"⚠️ OpenAI error: {str(e)}"

def get_confidence_color(confidence, has_disease):
    """
    Returns a color based on confidence and detection status.
    - For no disease: low → orange, mid → yellow, high → green
    - For disease: high → orange, mid → yellow, low → green
    """
    if has_disease:
        # 100 → 0
        if confidence >= UPPER_THESHOLD:
            return "#ea580c"  # orange
        elif confidence >= MIDDLE_THESHOLD:
            return "#eab308"  # yellow
        else:
            return "#22c55e"  # green
    else:
        # 0 → 100
        if confidence < MIDDLE_THESHOLD:
            return "#ea580c"  # orange
        elif confidence < UPPER_THESHOLD:
            return "#eab308"  # yellow
        else:
            return "#22c55e"  # green

def create_confidence_circle(confidence: float, color: str):
    fig, ax = plt.subplots(figsize=(2, 2), subplot_kw={'projection': 'polar'})
    ax.barh(1, 2 * 3.1416 * (confidence / 100), left=0, height=0.5, color=color)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_rticks([])
    ax.spines['polar'].set_visible(False)
    ax.set_title(f"{confidence:.1f}%", va='bottom', fontsize=12, color='#111827')
    fig.patch.set_facecolor('#FFFFFF')
    return fig

def parse_predict(prediction):
    try:
        tree, disease = prediction.split("___")
    except ValueError:
        return "Unknown", "Unrecognized format"
    return tree.replace("_", " "), disease.replace("_", " ")

def predict_disease(image):
    if image is None:
        return "No Image", "", "⚠️ No image provided", create_confidence_circle(0, "#ea580c"), *[gr.update(interactive=False)]*2, "", ""

    image = resize_image(image)
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    buffered.seek(0)

    try:
        response = predict_image(buffered)
        if response == "N/A":
            return "Unknown", "Not confident enough to predict.", "⚠️ Prediction confidence too low", create_confidence_circle(0, "#ea580c"), gr.update(interactive=False), "", gr.update(interactive=False), ""

        prediction = response.get("prediction")
        confidence = float(response.get("confidence"))
        treeName, disease = parse_predict(prediction)

    except Exception as e:
        return f"Model error: {e}", "", "", "", gr.update(interactive=False), "", gr.update(interactive=False), ""

    has_disease = disease.lower() not in ["unknown", "healthy", "none", ""]

    color = get_confidence_color(confidence, has_disease=has_disease)
    circle = create_confidence_circle(confidence, color)
    prediction_label = f"🟢 **Detected Disease:** `{disease}`"

    if not has_disease:
        return treeName, "No specific disease detected. Plant appears healthy.", prediction_label, circle, gr.update(interactive=False), "", gr.update(interactive=False), ""

    return treeName, disease, prediction_label, circle, gr.update(interactive=True), "", gr.update(interactive=True), (disease if has_disease else "")

def save_result(prediction, explanation):
    filename = "disease_report.txt"
    content = f"{prediction}\n\n{explanation}"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    return filename

with gr.Blocks(title="🌿 CropCare AI", theme=gr.themes.Base().set(
    body_background_fill="#F3F4F6",     # light page background
    body_text_color="#111827",         # dark text
    block_background_fill="#FFFFFF",   # white cards/blocks
    button_primary_background_fill="#22c55e",  # green button
    button_primary_text_color="#ffffff",       # white text on buttons
    button_large_padding="16px 24px",  # horizontal space
)) as demo:
    gr.Markdown("<h1 style='text-align: center; color: #22c55e;'>🌿 CropCare AI</h1>")
    gr.Markdown("<p style='text-align: center; font-size: 16px;'>Upload an image of a diseased plant to detect the disease and get treatment info.</p>")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="📷 Upload Leaf Image", height=300)
            with gr.Row():
                clear_btn = gr.Button("🔁 Clear", variant="secondary")
                predict_btn = gr.Button("🔍 Predict", variant="primary")

            with gr.Column():
                if os.path.exists(EXAMPLE_IMAGE_PATH):
                    gr.Markdown("**Example Image:**")
                    gr.Image(value=EXAMPLE_IMAGE_PATH, label="Example Leaf Image", height=300)
                else:
                    gr.Markdown("**Example Image:** No example image available.")

            # Pre-requisites information
            gr.Markdown("### 📋 How to Use\n" + "\n".join(f"- {line}" for line in GUIDE_LINES.strip().split("\n")))

        with gr.Column(scale=1):
            gr.Markdown("### 🧾 Prediction Results")
            tree_name_box = gr.Textbox(label="🌳 Tree Name", interactive=False)
            disease_name_box = gr.Textbox(label="🦠 Disease Name", interactive=False)
            prediction_box = gr.Markdown()
            confidence_plot = gr.Plot(label="📊 Confidence Level")
            chatgpt_btn = gr.Button("🤖 What is the disease?", interactive=False)
            response_box = gr.Textbox(label="📘 Disease Explanation", lines=10, interactive=False, visible=True)
            response_markdown = gr.Markdown(visible=False)
            save_btn = gr.Button("💾 Save Report", interactive=False)

    disease_state = gr.State()

    predict_btn.click(
        fn=reset_response_boxes,
        inputs=[],
        outputs=[response_box, response_markdown]
    ).then(
        fn=predict_disease,
        inputs=image_input,
        outputs=[
            tree_name_box,
            disease_name_box,
            prediction_box,
            confidence_plot,
            chatgpt_btn,
            response_box,
            save_btn,
            disease_state
        ]
    )

    chatgpt_btn.click(
        fn=reset_response_boxes,
        inputs=[],
        outputs=[response_box, response_markdown]
    ).then(
        fn=get_disease_info,
        inputs=[disease_state],
        outputs=[response_box, response_markdown]
    )

    save_btn.click(
        fn=save_result,
        inputs=[prediction_box, response_box],
        outputs=gr.File(label="📥 Download Report")
    )

    clear_btn.click(
        fn=reset_response_boxes,
        inputs=[],
        outputs=[response_box, response_markdown]
    ).then(
        fn=lambda: (None, "", "", "", None, gr.update(interactive=False), "", gr.update(interactive=False)),
        inputs=None,
        outputs=[
            image_input,
            tree_name_box,
            disease_name_box,
            prediction_box,
            confidence_plot,
            chatgpt_btn,
            response_box,
            save_btn
        ]
    )

demo.launch()
