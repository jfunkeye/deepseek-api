from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import pytesseract
from io import BytesIO
from sympy import sympify, SympifyError
from transformers import BlipProcessor, BlipForQuestionAnswering
import torch

app = FastAPI()

# Allow CORS if your frontend is on different domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip2-flan-t5-xl").to(device)

def try_solve_math(expr: str) -> str:
    try:
        result = sympify(expr).evalf()
        return str(result)
    except (SympifyError, TypeError):
        return None

def generate_ai_answer(image: Image.Image, question: str) -> str:
    inputs = processor(image, question, return_tensors="pt").to(device)
    outputs = model.generate(**inputs)
    answer = processor.decode(outputs[0], skip_special_tokens=True)
    return answer

class TextQuestion(BaseModel):
    question: str

@app.post("/api/ask-text/")
async def ask_text(input: TextQuestion):
    question = input.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Empty question")

    # Try math first
    math_answer = try_solve_math(question)
    if math_answer:
        return {"result": f"Math answer: {math_answer}"}

    # No image, so just return question or some default
    # You can optionally implement a text-only AI model here
    # For now, just echo the question back
    return {"result": f"Received question: {question}. (No image input, AI text answering not implemented yet)"}

@app.post("/api/ask-image/")
async def ask_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Upload an image.")

    image_data = await file.read()
    image = Image.open(BytesIO(image_data)).convert("RGB")

    extracted_text = pytesseract.image_to_string(image).strip()

    if not extracted_text:
        return {"result": "No text detected in the image."}

    math_answer = try_solve_math(extracted_text)
    if math_answer:
        return {"result": f"Math answer: {math_answer}"}

    ai_answer = generate_ai_answer(image, extracted_text)
    return {"result": ai_answer}
