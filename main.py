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





