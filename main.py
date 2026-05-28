from unstructured.partition.pdf import partition_pdf

output_path = './content/'
file_path = output_path + 'attention.pdf'

#Extracting Data From PDF file

def partition_pdf_file(file_path):
    '''Partition a PDF file and return the elements.'''
    chunks = partition_pdf(
        filename=file_path,
        infer_table_structure=True,
        strategy='hi_res',
        extract_image_block_types=['Image'],
        extract_image_block_to_payload=True,
        chunking_strategy='by_title',
        max_characters=2000,
        combine_text_under_n_characters=500,
        new_after_n_characters=6000,

    )
    return chunks

chunks = partition_pdf_file(file_path)

#Processing The Extracted Data
def get_table(chunks):
    tables = []
    for chunk in chunks:
        for el in chunk.metadata.original_elements:
            if 'Table' in str(type(el)):
                print(el.to_dict())
                tables.append(el)

    return tables

tables = get_table(chunks) 


#Text and Images 

def save_texts(chunks):
    '''Extract text from the list of chunks.'''
    texts = [chunk for chunk in chunks if 'CompositeElement' in str(type(chunk))]
    return texts

texts = save_texts(chunks)

def get_image_base64(chunks):
    '''Get the base64 encoded images from the chunks.'''
    image_64 = []
    for chunk in chunks:
        chunk_el = chunk.metadata.original_elements
        for el in chunk_el:
            if 'Image' in str(type(el)):
                image_64.append(el.metadata.image_base64)

    return image_64

images = get_image_base64(chunks)

#Image and Text Summarization

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StructuredOutputParser

##Summarize Text
##create an openAi model 
from dotenv import load_dotenv
import os 
load_dotenv(override=True)
API_KEY = os.getenv('GEMINI_API_KEY')
model = ChatOpenAI(api_key = API_KEY , model='gemini-2.5-flash',base_url='https://generativelanguage.googleapis.com/v1beta2/models/gemini-2.5-flash:generateContent', temperature=0.7)

##Create the prompt template for summarization
prompt = """
Your are an assistant tasked with summarizing tables and text.
Give a concise summary of the table or text.

Respond only with the summary ,no additional comment.
Do not start your message by saying 'Here is a summary' or anythings like that.
Just give the summary as it is.
Table or text chunk:{element}

"""


prompt_tamplate = ChatPromptTemplate.from_template(prompt)

#Chain the prompt with the gemini model and output parser
chain_text = prompt_tamplate|model|StructuredOutputParser

text_summaries = chain_text.batch(texts,{'max_concurrency':3})
