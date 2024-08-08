import fitz  # PyMuPDF
import openai# Step 1: Read the PDF file and extract its content using PyMuPDF
import tiktoken
from openai import OpenAI
api_key = ''
client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=api_key,
)
def read_pdf(file_path):
    text = ""
    with fitz.open(file_path) as doc:
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
    return text# Step 2: Interact with OpenAI's GPT to convert text to DITA XML
def convert_to_dita_xml(text, api_key):
    openai.api_key = api_key    # Creating a refined prompt for better DITA XML conversion
    prompt = (
        "Convert the following text to DITA XML. Ensure the output is well-formed XML and "
        "includes the necessary DITA tags such as <topic>, <title>, and <body>.\n\n"
        f"{text}"
    )
    try:
        response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=2000,  # Adjust based on your needs
            temperature=0.5,
        )
        # response = openai.ChatCompletion.create(
        #     model="gpt-4",
        #     messages=[
        #         {"role": "system", "content": "You are a helpful assistant."},
        #         {"role": "user", "content": prompt},
        #     ],
        #     max_tokens=2000,  # Adjust based on your needs
        #     temperature=0.5,
        # )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        print(f"Error during API call: {e}")
        return None# Step 3: Save the converted DITA XML to a specific location
def save_to_file(content, file_path):
    with open(file_path, 'w') as file:
        file.write(content)
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens
def main():
    pdf_path = r"C:\Users\ianta\Documents\product_cp_files_download_download_16211\94271 - mc_concentrate.pdf"  # Path to your PDF file
    output_path = r"C:\Users\ianta\Documents\Label Data Extraction"  # Path to save the DITA XML file
    openai_api_key = ""  # Your OpenAI API key
    pdf_content = read_pdf(pdf_path)    # Convert to DITA XML using OpenAI GPT
    dita_xml_content = convert_to_dita_xml(pdf_content, openai_api_key)
    tokensno = num_tokens_from_string(pdf_content, "cl100k_base")
    if dita_xml_content:
        # Save the converted content to a file
        save_to_file(dita_xml_content, output_path)
        print(f"Converted DITA XML file saved to {output_path}")
    else:
        print("Failed to convert PDF content to DITA XML.")
        print(tokensno)
if __name__ == "__main__":
    main()
    
    
    
 API code   
    
import os
import fitz  # PyMuPDF
import openai
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import tiktoken
# Initialize FastAPI app
app = FastAPI()
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Define Pydantic model for response
class DITAResponse(BaseModel):
    dita_xml: str
    token_count: int
# Function to read the PDF file and extract its content using PyMuPDF
def read_pdf(file_path: str) -> str:
    try:
        text = ""
        with fitz.open(file_path) as doc:
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text()
        return text
    except Exception as e:
        logger.error(f"Error reading PDF: {e}")
        raise HTTPException(status_code=500, detail="Failed to read PDF file")
# Function to interact with OpenAI's GPT to convert text to DITA XML
def convert_to_dita_xml(text: str) -> str:
    api_key = ""
    openai.api_key = api_key
    prompt = (
        "Convert the following text to DITA XML. Ensure the output is well-formed XML and "
        "includes the necessary DITA tags such as <topic>, <title>, and <body>.\n\n"
        f"{text}"
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=2000,
            temperature=0.5,
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        logger.error(f"Error during OpenAI API call: {e}")
        raise HTTPException(status_code=500, detail="Failed to convert PDF content to DITA XML")
# Function to calculate the number of tokens in a text string
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens
# API endpoint to upload PDF and get DITA XML
@app.post("/convert_pdf_to_dita", response_model=DITAResponse)
async def convert_pdf_to_dita(file: UploadFile = File(...)):
    try:
        # Save the uploaded file temporarily
        file_path = f"/tmp/{file.filename}"
        with open(file_path, "wb") as f:
            f.write(file.file.read())
        # Read the PDF content
        pdf_content = read_pdf(file_path)
        # Convert to DITA XML using OpenAI GPT
        dita_xml_content = convert_to_dita_xml(pdf_content)
        # Calculate the number of tokens
        tokensno = num_tokens_from_string(pdf_content, "cl100k_base")
        # Clean up the temporary file
        os.remove(file_path)
        return DITAResponse(dita_xml=dita_xml_content, token_count=tokensno)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred")
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
