import os
import json
import uuid
from collections import defaultdict
from tqdm import tqdm
from PIL import Image
import pandas as pd
import chainlit as cl
from langchain_community.embeddings.ollama import OllamaEmbeddings
from sentence_transformers import CrossEncoder
from qdrant_client.http.models import PointStruct, VectorParams
from qdrant_client import QdrantClient
from langchain_community.llms.ollama import Ollama
from chainlit import Message, on_chat_start, make_async
from chainlit.types import AskFileResponse
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.basic import chunk_elements
from semantic_text_splitter import TextSplitter
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM 
from transformers import AutoProcessor 
from excelProcess import getData , excelData , query_and_retrieve_parent

# Initialize models
semantic_model = OllamaEmbeddings(model="nomic-embed-text")
# cross_encoder_model = CrossEncoder("cross-encoder/stsb-roberta-base")
cross_encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
client = QdrantClient(url="http://localhost:6333")

collection_name = "clean1"

# ImageDIR='C:/sharan/ragllm/unstruc1/pdfOutput/demo/test'
ImageDIR='C:/sharan/ragllm/unstruc1/pdfOutput/demo/Image'

summary_directory = "C:/sharan/ragllm/unstruc1/pdfOutput/demo/ImagesSummariestest"
merged_output_file = "C:/sharan/ragllm/unstruc1/merged_output_file.json"


model_id = "microsoft/Phi-3.5-vision-instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map="cpu",
    trust_remote_code=True,
    _attn_implementation='eager'
)
processor = AutoProcessor.from_pretrained(
    model_id,
    trust_remote_code=True,
    num_crops=4
)


welcome_message = """Welcome to the RAG QA demo! To get started:
1. Upload a PDF or Xlsx
2. Ask a question about the file
"""

def preprocess_and_remove(pdf_file=None,xlsx_file=None):
    if pdf_file and xlsx_file is not None:
        filtered_df=getData(xlsx_file,sepData=None)

        # sepData = partition_pdf(
        #     filename=file.path,
        #     strategy='hi_res',
        #     extract_images_in_pdf=True,
        #     infer_table_structure=True,
        #     extract_image_block_types=['Image', 'Table'],
        #     extract_image_block_to_payload=False,
        #     extract_image_block_output_dir=ImageDIR
        # )

        # sepData = partition_pdf(
        # filename=pdf_file,
        # strategy='hi_res',
        # infer_table_structure=True,
        # )

        # # Initialize a dictionary to track the first image and table removal per page
        # page_removals = {}

        # # Iterate over the elements
        # for element in sepData:
        #     page_no = element.metadata.page_number
        #     element_type = type(element).__name__

        #     # Check if the page has already had an image or table removed
        #     if page_no not in page_removals:
        #         page_removals[page_no] = {'image': False, 'table': False}

        #     # Remove the first image or table if it hasn't been removed yet
        #     if element_type == 'Image' and not page_removals[page_no]['image']:
        #         sepData.remove(element)
        #         page_removals[page_no]['image'] = True
        #     elif element_type == 'Table' and not page_removals[page_no]['table']:
        #         sepData.remove(element)
        #         page_removals[page_no]['table'] = True

        # Initialize dictionaries to track the smallest image counts per page
        # smallest_figure = defaultdict(lambda: float('inf'))

        # Track the filenames to delete
        # files_to_delete = []

        # # Iterate over the files in the directory
        # for filename in os.listdir(ImageDIR):
        #     # Check if the file is an image or table with the naming convention
        #     if filename.startswith('figure-') or filename.startswith('table-'):
        #         # Split the filename to extract page number and image/table count
        #         parts = filename.split('-')
        #         page_number = parts[1]
        #         count = int(parts[2].split('.')[0])  # Remove file extension and convert to int

        #         # Determine if it's a figure or table
        #         if filename.startswith('figure-'):
        #             # Update the smallest figure count for the page
        #             if count < smallest_figure[page_number]:
        #                 smallest_figure[page_number] = count
        #                 # Store the filename to delete
        #                 files_to_delete.append((page_number, 'figure', filename))
        #         elif filename.startswith('table-'):
        #             # Store the filename to delete
        #             files_to_delete.append((page_number, 'table', filename))

        # # Delete the files with the smallest count for each page and all tables
        # for page_number, item_type, filename in files_to_delete:
        #     if item_type == 'figure' and smallest_figure[page_number] == int(filename.split('-')[2].split('.')[0]):
        #         file_path = os.path.join(ImageDIR, filename)
        #         os.remove(file_path)
        #         print(f"Deleted figure: {file_path}")
        #     elif item_type == 'table':
        #         file_path = os.path.join(ImageDIR, filename)
        #         os.remove(file_path)
        #         print(f"Deleted table: {file_path}")
    return filtered_df

def store_text_table(sepData):
    # Ensure the collection exists with the correct vector size
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=768, distance="Cosine")
    )

    # Extract and process tables
    html_tables = []
    for element in sepData:
        if "Table" in str(type(element)):
            html_content = getattr(element.metadata, "text_as_html", None)
            if html_content:
                html_tables.append((html_content, element.metadata.page_number))  # Store html content with page number

    dataframes = [df for html_content, _ in html_tables for df in pd.read_html(html_content)]

    def _preprocess_tables(tables):
        return ["\n".join([table.to_csv(index=False)]) for table in tables]

    processed_tables = _preprocess_tables(dataframes)

    sepData = [
    element for element in sepData
    if "unstructured.documents.elements.Table" not in str(type(element)) and
       "unstructured.documents.elements.Image" not in str(type(element))
    ]
    chunks = chunk_elements(sepData, new_after_n_chars=800, max_characters=1000)

    # Load text chunks into Qdrant
    for chunk in tqdm(chunks, desc="Loading text chunks"):
        vector = semantic_model.embed_query(chunk)
        point = PointStruct(id=str(uuid.uuid4()), vector=vector, payload={"text": chunk.text, "page_number": chunk.metadata.page_number})
        client.upsert(collection_name=collection_name, points=[point])

    # Load processed tables into Qdrant
    for i, tables in tqdm(enumerate(processed_tables), total=len(processed_tables), desc="Loading Tables"):
        vector = semantic_model.embed_query(tables)
        html_content, page_number = html_tables[i]  # Retrieve the page number
        point = PointStruct(id=str(uuid.uuid4()), vector=vector, payload={"text": html_content, "page_number": page_number})
        client.upsert(collection_name=collection_name, points=[point])


def generate_image_summaries(processor, model):
    summaries = {}
    all_files = [f for f in os.listdir(ImageDIR) if f.endswith('.jpg') or f.endswith('.png')]
    for filename in tqdm(all_files, desc="Processing images"):
        json_filename = os.path.join(summary_directory, f"{filename}.json")
        
        if os.path.exists(json_filename):
            print(f"Skipping {filename}, summary already exists.")
            continue

        image_path = os.path.join(ImageDIR, filename)
        image = Image.open(image_path)

        # Prepare the prompt
        placeholder = f"<|image_1|>"
        messages = [
            {
                "role": "user",
                "content": (
                    placeholder + 
                    "Summarize the key components and relationships depicted in the P&ID diagram, focusing on the main processes, equipment and instrumentation. "
                    "Highlight the flow of materials and signals, and identify any critical control loops or safety features. Provide a concise overview suitable for semantic retrieval applications. "
                    "Definitions: "
                    "DCS - Distributed Control System, "
                    "SIS - Safety Instrumented System, "
                    "PLC - Programmable Logic Controller, "
                    "HMI - Human Machine Interface, "
                    "PID - Proportional Integral Derivative Control Algorithm, "
                    "PIC - Pressure Indicating Controller, "
                    "FIC - Flow Indicating Controller, "
                    "LIC - Level Indicating Controller, "
                    "TIC - Temperature Indicating Controller, "
                    "RSP - Remote Setpoint, "
                    "P&ID - Piping and Instrument Diagram, "
                    "SP - Setpoint, "
                    "OP - Output, "
                    "PV - Process Variable."
                )
            }
        ]
        prompt = processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Process the image and generate the summary
        inputs = processor(prompt, image, return_tensors="pt")
        generation_args = {
            "max_new_tokens": 500,
            "temperature": 0.0,
            "do_sample": False,
        }
        generate_ids = model.generate(
            **inputs,
            eos_token_id=processor.tokenizer.eos_token_id,
            **generation_args
        )
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        # Store the summary in the dictionary
        summaries[filename] = response

        # Save each summary as a separate JSON file
        with open(json_filename, 'w') as json_file:
            json.dump({filename: response}, json_file)

    print("Summaries have been saved as individual JSON files.")
    return summaries

def merge_summaries():
    all_summaries = {}
    json_files = [f for f in os.listdir(summary_directory) if f.endswith('.json')]
    for json_file in json_files:
        json_path = os.path.join(summary_directory, json_file)
        with open(json_path, 'r') as f:
            summary = json.load(f)
            all_summaries.update(summary)

    with open(merged_output_file, 'w') as f:
        json.dump(all_summaries, f)

    print(f"Merged summaries have been written to {merged_output_file}")
    return all_summaries

import json
import os
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, PointStruct
from tqdm import tqdm

# Initialize the Qdrant client
client = QdrantClient(host='localhost', port=6333)  # Adjust host and port as needed

summary_directory = "C:/sharan/ragllm/unstruc1/pdfOutput/demo/ImagesSummariestest"

def create_and_vectorize_summaries(embeddings):
    # Define the vector size and distance metric
    vector_size = 768  # Example vector size, adjust as needed
    distance_metric = 'Cosine'  # You can also use 'Euclidean' or 'Dot'

    # Create the collection
    collection_name = 'ImageTest'
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=distance_metric)
    )

    print(f"Collection '{collection_name}' created successfully.")

    # Vectorize and store summaries
    for json_filename in tqdm(os.listdir(summary_directory), desc="Processing JSON files"):
        if json_filename.endswith('.json'):
            json_path = os.path.join(summary_directory, json_filename)
            
            with open(json_path, 'r') as json_file:
                data = json.load(json_file)
                for filename, summary in data.items():
                    # Extract PageNumber from the filename
                    parts = filename.split('-')
                    if len(parts) >= 3:
                        page_number = parts[1]  # Assuming the format is figure-PageNumber-CountNumber
                    else:
                        print(f"Filename format is incorrect: {filename}")
                        continue

                    vector = embeddings.embed_query(summary)
                    point_id = int(page_number)  # Use PageNumber as the ID
                    point = PointStruct(
                        id=point_id,
                        vector=vector,
                        payload={
                            "filename": filename,
                            "summary": summary
                        }
                    )
                    client.upsert(
                        collection_name=collection_name,
                        points=[point]
                    )

    print("Summaries and filenames have been vectorized and stored in Qdrant.")


def get_Image_summary_by_page_number(page_number):
    collection_name = 'ImageTest'
    
    # Retrieve the point with the given page_number as ID
    result = client.retrieve(
        collection_name=collection_name,
        ids=[int(page_number)]  # Ensure the ID is a string
    )
    
    if result:
        # Assuming the payload contains the summary
        summary_payload = result[0].payload.get('summary', 'Not found')
        return summary_payload
    else:
        return "Not found"





def query_llm(context_with_pages, question,excel_data,image_summary):

    instruction = (
        "You are a helpful assistant that responds to the user's question based on the context. "
        "Be short, concise and to the point when answering the question.You may also be provided with excel data. related to the control narrative"
        "You are also provided with Image Summary of the P&ID diagram.Use it to answer if the question is related to the image which is P&ID diagram"
        "You are a RAG LLM and context given to you are chunks retrieved via semantic search from a document. "
        "If the answer to the question is not available in the context provided, please say 'Information not available as a part of this Document.'"
    )
    
    # Extract context and page numbers
    # context = " ".join([text for text, _ in context_with_pages])
    # page_numbers = [page for _, page in context_with_pages if page is not None]
    
    prompt_template = (
        f"Instruction: {instruction}\n\n"
        f"Excel Data:\n{excel_data}\n\n"
        f"Image Summary:\n{image_summary}\n\n"
        f"Context:\n{context_with_pages}\n\n"
        f"Question: {question}\n\n"
    
        f"Answer:"
    )
    print(prompt_template)
    # model = Ollama(model="phi3:mini")
    # model = Ollama(model="phi3.5")

    model = Ollama(model="mistral:latest")
    # model = Ollama(model="gemma2:2b")
    response = model.invoke(prompt_template)
    print(response)
    print(type(response))
    
    return response

async_function = make_async(query_llm)

def VLMquery(image,query):
    image = image[0].path
    image = Image.open(image)
    placeholder = f"<|image_1|>"
    messages = [
            {
                "role": "user",
                "content": (
                    placeholder + 
                    "You are an expert in P&ID diagram and will explain the image based on user query."
                    "User query:"+query+
                    "Definitions: "
                    "DCS - Distributed Control System, "
                    "SIS - Safety Instrumented System, "
                    "PLC - Programmable Logic Controller, "
                    "HMI - Human Machine Interface, "
                    "PID - Proportional Integral Derivative Control Algorithm, "
                    "PIC - Pressure Indicating Controller, "
                    "FIC - Flow Indicating Controller, "
                    "LIC - Level Indicating Controller, "
                    "TIC - Temperature Indicating Controller, "
                    "RSP - Remote Setpoint, "
                    "P&ID - Piping and Instrument Diagram, "
                    "SP - Setpoint, "
                    "OP - Output, "
                    "PV - Process Variable."
                )
            }
        ]
    prompt = processor.tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
    )

    # Process the image and generate the summary
    inputs = processor(prompt, image, return_tensors="pt")
    generation_args = {
        "max_new_tokens": 500,
        "temperature": 0.0,
        "do_sample": False,
    }
    generate_ids = model.generate(
        **inputs,
        eos_token_id=processor.tokenizer.eos_token_id,
        **generation_args
    )
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(
        generate_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    print("Done")
    return response

VLM = make_async(VLMquery)


filtered_df = None
@cl.on_chat_start
async def start():
    global filtered_df
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content=welcome_message,
            accept=["application/pdf", "application/.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"],
            max_files=2,
            max_size_mb=20,
            timeout=3600
        ).send()
    
    print(files)
    pdf_file = None
    xlsx_file = None

    for file in files:
        if file.name.endswith('.PDF'):
            pdf_file = file.path
        elif file.name.endswith('.XLSX'):
            xlsx_file = file.path

    print(pdf_file,xlsx_file)
    if pdf_file and xlsx_file is not None:
        msg = cl.Message(content="Processing PDF and Excel files.")
    elif pdf_file:
        msg = cl.Message(content=f"Processing PDF file `{pdf_file}`...")
    elif xlsx_file:
        msg = cl.Message(content=f"Processing Excel file `{xlsx_file}`...")
    else:
        msg = cl.Message(content="No valid files provided.")
    
    await msg.send()

    process = make_async(preprocess_and_remove)
    storeTextTable = make_async(store_text_table)
    filtered_df=await process(pdf_file,xlsx_file)
    # sepData = await process(pdf_file,xlsx_file)
    # await storeTextTable(sepData)
    # print("DONE")

    # generate_image_summaries(processor,model)
    # generatesum = make_async(generate_image_summaries)
    # await generatesum(processor,model)
    
    # # # merge_summaries()
    # mergesum=make_async(merge_summaries)
    # await mergesum()

    # # # vectorize_and_store_summaries(semantic_model)
    # vectorizeImage = make_async(vectorize_and_store_summaries)
    # await vectorizeImage(semantic_model)

    # Additional processing logic can be added here
    msg.content = "processed. You can now ask questions!"
    await msg.update()


@cl.on_message
async def main(message: cl.Message):
    if message.elements:
        vlmResponse = await VLM(message.elements,message.content)
        await Message(content=vlmResponse).send()
    else:
        query_text = message.content
        context_with_page , page_number = query_and_retrieve_parent(query_text)
        excel = make_async(excelData)
        excel_data = await excel(query_text,filtered_df)
        Image_summary = make_async(get_Image_summary_by_page_number)
        image_summary = await Image_summary(page_number)
        ###
        llm_response = await async_function(context_with_page,query_text,excel_data,image_summary)
        ###
        print("got llm response")
        # msg.content = f"{llm_response}\n\nPage Number:{page_number}"
        # await msg.update()
        await Message(content=f"{llm_response}\n\nPage Number:{page_number}").send()

if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)