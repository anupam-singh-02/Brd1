import frappe
import os
import pickle
import fitz  # PyMuPDF
# import nltk
import json
from enum import Enum
import pandas as pd
import platform
import subprocess
import tempfile
import langchain
# import aspose.words as aw
from sentence_transformers import SentenceTransformer, util
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain.memory import SimpleMemory
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_ollama import OllamaLLM
from langchain.schema import Document
from pydantic import BaseModel, Field, ValidationError
from typing import Literal
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from bs4 import BeautifulSoup
from . import utility
@frappe.whitelist()
def validate_document_checklist(file1_path, checkbox_list='false'):
    checkbox_list = False if checkbox_list == 'false' else True
    print("checklist-->", checkbox_list, type(checkbox_list))
    upload_folder1 = os.path.dirname(file1_path)
    file_name = os.path.basename(file1_path)
    try:
        html1 = utility.convert_docx_to_html(file1_path, upload_folder1, True)
    except Exception as e:
        frappe.msgprint(f'{str(e)}', indicator='error')
        return {'message': f'{str(e)}'}, 400
    # Clean up the HTML content if needed.
    html1 = utility.remove_empty_sections(html1)
    html_original_lines = utility.section_lines_with_section_using_bs4(html1)
    html_data1 = utility.split_html_sections_in_list(html_original_lines)
    updated_html_data1 = utility.split_html_after_p_tags(html_data1)
    updated_html_data1 = [item.replace('''class="[\'card_section\']"''', '''class="card_section"''') for item in updated_html_data1]
    file_html = "".join(updated_html_data1)
    class ClauseEnum(Enum):
        C1 = "The date of the agreement shall be clearly mentioned within the document and must not be backdated."
        C2 = "The term of this agreement shall be as specified in the contract, covering the total duration agreed upon by both parties."
        C3 = "The actual nature of support to be provided shall be clearly specified in the agreement. The client must explicitly define the type and scope of support required from the vendor to ensure clarity and mutual understanding."
        C4 = "Payment shall be made within a credit period of up to 30 days from the date of invoice. Applicable taxes will be charged extra as per the prevailing tax laws. Any delay in payment beyond the stipulated credit period may incur additional charges or penalties as agreed upon in the contract."
        C5 = "The vendor shall not provide the same services to other clients in order to avoid competition, except for existing clients and orders that are already in the pipeline at the time of this agreement. This clause does not restrict the vendor from fulfilling ongoing or prior commitments to current clients."
        C6 = "The agreement may only be terminated after a notice period of 60 days by either party. During this period, both parties shall continue to fulfill their obligations under the agreement until the termination is effective."
        C7 = "The services provided to the client shall remain confidential and shall not be shared with any other client. The vendor is prohibited from disclosing or using such services for other clients. This restriction applies up to 6 months after the termination of the contract."
        C8 = "The parties shall comply strictly with all applicable Indian laws. However, import/export compliance and legal obligations under UK or US laws shall not be applicable or enforceable under this agreement."
        C9 = "Only employee-related insurance, such as health insurance, accident insurance, or any other coverage directly benefiting employees, will be considered acceptable. However, liability insurance, which typically covers legal claims, damages, or third-party liabilities against the organization, is not acceptable. The cumulative leave of 1.75 i.e. 2 (two) days to be allowed during the month as per statutory requirement. Interest @ 2% per month will be charged if bills are not paid within 30 days from the date of the invoice"
        C10 = "The vendor agrees to indemnify, defend, and hold harmless the client, its affiliates, officers, employees, and agents from and against any third-party claims, damages, losses, liabilities, costs, and expenses (including reasonable attorney fees) arising out of or related to the vendor's negligence, willful misconduct, or breach of its obligations under this agreement. This indemnity shall not apply to the extent that the claim arises due to the client's negligence, misconduct, or breach of agreement."
        C11 = "During the term of this Agreement and for one year thereafter, neither party nor any of their vendors shall knowingly solicit, hire or engage or enter into any contract of employment or consultancy, whether on permanent or temporary basis with any of the other party's employees for one year following the termination of such employee's employment with the other party, without such other party's prior written consent. In case of such hire, the other party will be entitled to claim an amount of one annual salary of the said consultant from the hirer.Below point can be added if client suggest revision This restriction does not prohibit either of the parties from giving consideration to any application for employment submitted on an unsolicited basis or in response to a general advertisement or communication of employment opportunities."
        C12 = "In the event of any damage or loss incurred by the client due to the vendor's mistake, negligence, or breach of contract, the vendor shall be liable to compensate the client. The liability shall be limited to direct damages only, and in no event shall the vendor be liable for any indirect, incidental, or consequential damages, including but not limited to loss of profit, revenue, or data. The vendor's total liability shall not exceed the amount specified in the purchase order or the fees paid under this agreement, whichever is lower."
        C13 = "In the event the vendor fails to deliver the services within the specified time, a penalty in the form of service credits will be accepted. The penalty amount will be as specified in the purchase order, or as agreed upon, whichever is applicable. This penalty will be deducted from the total payable amount for the contract."
        C14 = "Nothing in this Agreement will be understood to preclude or limit Clover from providing software, materials, or services for itself or other clients, irrespective of the possible similarity of such software, materials or services to those which might be delivered to Client."
        C15 = "The provision of this service agreement which imposes an obligation after termination or expiration of the Agreement shall survive for a period of one (1) year post termination or expiration of this service agreement."
        C16 = """If any dispute arises between the Parties during the subsistence of this Agreement or thereafter, in connection with the validity, interpretation, implementation, or alleged breach of any provision of this Agreement, or regarding a question including the legitimacy of the termination of this Agreement by either Party, the Parties shall endeavor to settle such dispute amicably through mutual discussion. 
                The attempt to bring about an amicable settlement shall be deemed to have failed if, after reasonable efforts for not less than 30 (Thirty) days, either Party provides 15 (Fifteen) days’ written notice to the other Party invoking arbitration for the settlement of the dispute. The Arbitral Tribunal shall consist of three (3) Arbitrators: one appointed by each Party, and the third Arbitrator appointed by mutual consent of the two so appointed. The arbitration proceedings shall be governed by the Arbitration and Conciliation Act, 1996. The place of arbitration shall be Mumbai, India, and the proceedings shall be conducted in the English language. The Arbitrator’s award shall be substantiated in writing. The Arbitral Tribunal shall also determine the allocation of costs of the arbitration proceedings. The award shall be binding on the Parties subject to the applicable laws in force and shall be enforceable in any competent court of law. Exceptions to the Confidentiality and Non-Use Obligations Notwithstanding anything to the contrary contained in this Agreement, the obligations of confidentiality and non-use shall not apply to information that:
                1.1 was or becomes publicly available or known to the public through no breach of the Receiving Party’s or its Representatives' obligations under this Agreement;
                1.2 was or becomes known to the Receiving Party or its Representatives from sources other than the Disclosing Party or its Representatives, provided that such source is not known by the Receiving Party to be under a confidentiality obligation prohibiting disclosure; or
                1.3 is independently developed by the Receiving Party or on its behalf (including by its Representatives) without use of or reference to the Confidential Information of the Disclosing Party.
                """
        C17 = "All intellectual property rights in any work product, software, documentation, or deliverables provided under this Agreement shall remain the exclusive property of the originating party unless explicitly agreed otherwise in writing. Nothing in this Agreement shall be construed to transfer ownership of any pre-existing intellectual property between the parties."
        C18 = "Each party represents and warrants that it shall comply with all applicable anti-bribery and anti-corruption laws and regulations in connection with this Agreement. Neither party shall offer, give, solicit, or receive any improper payment or advantage, directly or indirectly, in connection with the performance of its obligations under this Agreement."
    
    class CheckboxClauseEnum(Enum):
        C9 = "Only employee-related insurance, such as health insurance, accident insurance, or any other coverage directly benefiting employees, will be considered acceptable. However, liability insurance, which typically covers legal claims, damages, or third-party liabilities against the organization, is not acceptable. The cumulative leave of 1.75 i.e. 2 (two) days to be allowed during the month as per statutory requirement. Interest @ 2% per month will be charged if bills are not paid within 30 days from the date of the invoice"
        C17 = "All intellectual property rights in any work product, software, documentation, or deliverables provided under this Agreement shall remain the exclusive property of the originating party unless explicitly agreed otherwise in writing. Nothing in this Agreement shall be construed to transfer ownership of any pre-existing intellectual property between the parties."
        C18 = "Each party represents and warrants that it shall comply with all applicable anti-bribery and anti-corruption laws and regulations in connection with this Agreement. Neither party shall offer, give, solicit, or receive any improper payment or advantage, directly or indirectly, in connection with the performance of its obligations under this Agreement."
    class ClauseResponse(BaseModel):
        clause_compliance: Literal['YES', 'NO'] = Field(..., description="Compliance status of the clause: 'YES' or 'NO'.")
        description: str = Field(..., description="Detailed description of the clause mentioned in the provided document.")
        source: str = Field(..., description="The actual text source of the clause from the provided document.")
    output_parser = PydanticOutputParser(pydantic_object=ClauseResponse)
    # Suppress the Warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # ollama_llm = True
    ollama_llm = False
    # Initialize the Sentence Transformer model
    model_name = 'all-MiniLM-L6-v2'
    model_path = frappe.get_site_path('private', 'models', 'all_MiniLM_L6_v2')
    
    print("model_path_new---->", frappe.get_site_path('private', 'models', 'all_MiniLM_L6_v2'))
    # model_path = "/home/rishabh-ubuntu/frappe-bench/sites/legal-doc-compare/private/models/all_MiniLM_L6_v2"
    # print("model_path_old---->", model_path)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    if not os.path.exists(model_path):
        frappe.msgprint(f"Downloading model '{model_name}'...")
        model = SentenceTransformer(model_name)  # Download the model
        model.save(model_path)  # Save the model to the specified path
        frappe.msgprint(f"Model saved to '{model_path}'")
    else:
        frappe.msgprint(f"Loading model from '{model_path}'...")
        model = SentenceTransformer(model_path, device='cpu')  # Load the model from the local path
    frappe.msgprint("Model loaded successfully.")
    # Modify Checklist Function
    def modify_checklist_html(html_content, result_data):
        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        # Process each result item
        for item in result_data:
            clause_id = item['clause_id']
            description = item['description']
            compliance = item['clause_compliance']
            # Update the span content
            span_tag = soup.find('span', {'id': f"span_{clause_id}"})
            if span_tag:
                span_tag.string = description
            # Update section class
            section_tag = soup.find('section', {'id': clause_id})
            if section_tag:
                # Get current classes or set to empty list
                classes = section_tag.get('class', [])
                # Add matched/removed class based on compliance
                if compliance == "YES":
                    classes.append("matched")
                else:
                    classes.append("removed")
                # Ensure unique classes
                section_tag['class'] = list(set(classes))
        # Return modified HTML
        modified_html = str(soup)
        # print(modified_html)
        return modified_html
    def modify_original_html(html_content, matched_section):
        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        # Process each result item
        for item in matched_section:
            section_id = item['section_id']
            section_tag = soup.find('section', {'id': section_id})
            if section_tag:
                # Get current classes or set to empty list
                classes = section_tag.get('class', [])
                # Add matched/removed class based on compliance
                classes.append("matched")
                section_tag['class'] = list(set(classes))
        # Return modified HTML
        modified_html = str(soup)
        # print(modified_html)
        return modified_html
    def extract_sections_with_id_and_content_from_html(data):
        soup = BeautifulSoup(data, 'html.parser')
        sections = []
        for section in soup.find_all('section'):
            section_id = section.get('id')
            content = ' '.join(section.stripped_strings)
            sections.append({"section_id": section_id, "section_content": content})
        return sections
    
    def get_similarity_score(para1, para2, threshold=0.60):
        embedding1 = model.encode(para1, convert_to_tensor=True)
        embedding2 = model.encode(para2, convert_to_tensor=True)
        cosine_score = util.cos_sim(embedding1, embedding2)
        is_similar = cosine_score >= threshold
        return {'is_similar': is_similar, 'cosine_score': cosine_score}
    # Validation Function in use
    def validate_checklist_using_groq_retrival(input_path):
        import nltk
        nltk.data.path.append('/home/rishabh-ubuntu/nltk_data')
    
        # Function to extract text from a PDF
        def extract_text_from_pdf(pdf_path):
            doc = fitz.open(pdf_path)
            text = ""
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text("text") + "\n"
            return text
        # Function to extract tables from a PDF
        def extract_tables_from_pdf(pdf_path):
            doc = fitz.open(pdf_path)
            table_data = []
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                tables = page.find_tables()
                for table in tables:
                    df = pd.DataFrame(table.extract())
                    table_data.append({"page": page_num + 1, "table": df})
            return table_data
        # Convert DOCX to PDF using Aspose.Words and return the PDF path for Mac/Windows
        def create_pdf(input_path):
            """Convert DOCX to PDF using Aspose.Words and return the temporary PDF file path."""
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
                temp_pdf_path = temp_pdf.name
            try:
                doc = aw.Document(input_path)
                doc.save(temp_pdf_path)
                return temp_pdf_path
            except Exception as e:
                frappe.msgprint(f"Error converting DOCX to PDF: {e}", indicator='error')
                return None
        # Convert DOCX to PDF using Aspose.Words and return the PDF path for Linux
        def create_pdf_in_linux(input_path):
            """
            Convert a DOCX file to PDF using LibreOffice in headless mode (Linux),
            return the PDF file path, and delete the file after use if needed.
            """
            # Ensure the output path is safe and unique
            output_dir = tempfile.mkdtemp()
            filename_without_ext = os.path.splitext(os.path.basename(input_path))[0]
            output_pdf_path = os.path.join(output_dir, f"{filename_without_ext}.pdf")
            try:
                # Convert DOCX to PDF
                result = subprocess.run(
                    ["libreoffice", "--headless", "--convert-to", "pdf", input_path, "--outdir", output_dir],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                # Ensure the PDF was created
                if not os.path.exists(output_pdf_path):
                    raise FileNotFoundError(f"PDF conversion failed: {output_pdf_path} not found.")
                # Return path to the converted PDF
                return output_pdf_path
            except subprocess.CalledProcessError as e:
                frappe.msgprint(f"LibreOffice conversion failed: {e.stderr}", indicator='error')
                raise
        def split_text_into_sentences(text):
            try:
                nltk.data.find("tokenizers/punkt")
            except LookupError:
                nltk.download("punkt", download_dir="/home/rishabh-ubuntu/nltk_data")
            return nltk.sent_tokenize(text)
        # Function to compute embeddings and save them
        def compute_and_save_embeddings(file_path, output_path, file_type="pdf"):
            if file_type == "pdf":
                text = extract_text_from_pdf(file_path)
                tables = extract_tables_from_pdf(file_path)
            else:
                system = platform.system()
                frappe.logger().debug(f'---> {system}')
                if system == "Linux":
                    pdf_path = create_pdf_in_linux(file_path)
                else:
                    pdf_path = create_pdf(file_path)
                text = extract_text_from_pdf(pdf_path)
                tables = extract_tables_from_pdf(pdf_path)
                if os.path.exists(pdf_path):
                    os.unlink(pdf_path)
            sentences = split_text_into_sentences(text)
            embeddings = model.encode(sentences, show_progress_bar=True)
            with open(output_path, "wb") as fOut:
                pickle.dump({'sentences': sentences, 'embeddings': embeddings, 'tables': tables}, fOut)
            return sentences, embeddings
        # not used---Initialize FAISS for fast retrieval
        def setup_vectorstore(sentences, embeddings):
            embedding_model = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": "cpu"})
            documents = [Document(page_content=sentence) for sentence in sentences]
            vectorstore = FAISS.from_documents(documents, embedding_model)
            return vectorstore
        # not used---Setup LLM with RetrievalChain and Memory
        def setup_llm_with_memory(vectorstore):
            memory = SimpleMemory()
            # llm = OllamaLLM(model="llama3.1:8b")
            # llm = OllamaLLM(model="llama3.3:70b-instruct-q2_K", num_ctx=32000)
            # llm = OllamaLLM(model="deepseek-r1:7b")
            groq_api_key = frappe.conf.get('groq_api_key')
            print(">>> groq_api_key", groq_api_key)
            if not groq_api_key:
                frappe.msgprint("Groq API key not configured.", indicator='error')
                raise frappe.ValidationError("Groq API key not configured.")
            llm = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=groq_api_key, temperature=0.0)
            # llm = OllamaLLM(model="mistral:latest")
            # llm = ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key=groq_api_key, temperature=0.0)
            # llm = ChatGroq(model_name="llama-3.2-1b-preview", groq_api_key=groq_api_key, temperature=0.0)
            retrieval_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever(), memory=memory)
            return retrieval_chain
        def get_context(question, sentences, embeddings, top_k=5):
            q_embedding = model.encode([question]).astype("float32")
            _, I = index.search(q_embedding, top_k)
            return "\n\n".join([sentences[i] for i in I[0]])
        
        def call_llm(context, question):
            prompt = prompt_template.format(context=context, question=question)
            return llm.invoke(prompt)
        def setup_llm(ollama_llm):
            if ollama_llm:
                print("ollama--->")
                result = OllamaLLM(model="llama3.1:8b", temperature=0.0)
            else:
                print("groq--->")
                groq_api_key = frappe.conf.get('groq_api_key')
                if not groq_api_key:
                    frappe.msgprint("Groq API key not configured.", indicator='error')
                    raise frappe.ValidationError("Groq API key not configured.")
                # result = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=groq_api_key, temperature=0.0) # llama-3.1-8b-instant
                result = ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key=groq_api_key, temperature=0.0) # llama-3.1-8b-instant
            return result
        # main usage
        output_path = frappe.get_site_path('private', 'embeddings', 'embeddings.pkl')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        file_type = os.path.splitext(input_path)[1].lstrip('.')
        sentences, embeddings = compute_and_save_embeddings(input_path, output_path, file_type)
        # vectorstore = setup_vectorstore(sentences, embeddings)
        # retrieval_chain = setup_llm_with_memory(vectorstore)
        embedding_model = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": "cpu"})
        vectorstore = FAISS.from_documents([Document(page_content=s) for s in sentences], embedding_model)
        index = vectorstore.index
        llm = setup_llm(ollama_llm)
        fixed_prompt = """
            You are a legal assistant. Use the provided context to answer the user's question as accurately as possible.
            Your task:
            - Find whether the clause described in the question exists in the context with the same or similar meaning.
            - If the clause is found, extract and return it.
            - Respond strictly in valid JSON format only — No explanation, no markdown, no comments.
            Format your response as a JSON code block using the exact structure below:
            
            {{
                "clause_compliance": "YES" or "NO",
                "description": "Detailed explanation of how the clause is or isn't addressed in the provided document.",
                "source": "The actual clause text from the document. If the clause is not found, write 'Clause not found.'"
            }}
            
            Instructions:
            Be accurate and concise in both description and source.
            Wrap YES/NO in double quotes.
            Context: {context}
            Question: {question}
            Answer:
        """
        
        # Create the template
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=fixed_prompt
        )
        print("checkbox_list groq--->", checkbox_list)
        if checkbox_list:
            legal_clauses = [key for key in CheckboxClauseEnum]
        else:
            legal_clauses = [key for key in ClauseEnum]
        result_data = []
        for idx, clause_enum in enumerate(legal_clauses):
            question = clause_enum.value
            # prompt = f"""Question: {clause}\n\n\n{fixed_prompt.format(format_instructions=output_parser.get_format_instructions())}"""
            # prompt = PromptTemplate(input_variables=["context", "question"], template=fixed_prompt)
            context = get_context(question, sentences, embeddings)
            response_data = call_llm(context, question)
            response = response_data if ollama_llm else response_data.content
            try:
                # parsed_response = output_parser.parse(response['result'])
                # json_data = json.loads(parsed_response.model_dump_json())
                json_data = json.loads(response)
                print(f">>>>>>>> json_data {int(idx) + 1}:", json_data)
                if json_data and isinstance(json_data, dict):
                    json_data["clause_id"] = clause_enum.name
                    # json_data["description"] = ""
                    # json_data["source"] = ""
                else:
                    json_data = {
                        "clause_compliance": "NO",
                        "clause_id": clause_enum.name,
                        "description": "Clause no mentationed",
                        "source": "Clause no mentationed",
                    }
                result_data.append(json_data)
            except Exception as e:
                json_data = {
                    "clause_compliance": "NO",
                    "clause_id": clause_enum.name,
                    "description": "Clause no mentationed",
                    "source": "Clause no mentationed",
                }
                result_data.append(json_data)
                frappe.logger().error(f"JSON Error: {e}")
                frappe.logger().debug(f"Raw Response: {response}")
        return result_data
    result_data = validate_checklist_using_groq_retrival(file1_path)
    print("checkbox_list---->", checkbox_list)
    print("result_data--->", result_data)
    if checkbox_list:
        return result_data
    html_content = f"""<section class="card_section" id="C1">
        <p style="font-size: 11pt; vertical-align: middle;">
            <span style="font-weight: bold; vertical-align: middle;">Date of Agreement :</span>
        </p>
        <p style="font-size: 11pt; vertical-align: middle;">
            <span style="vertical-align: middle;">{ClauseEnum.C1.value}</span>
        </p>
        <p style="border: 1px solid orange;padding:4px;font-size: 11pt; vertical-align: middle;">
            <span style="font-weight: bold; vertical-align: middle;">Description :</span>
            <span id="span_C1" style="vertical-align: middle;"></span>
        </p>
    </section>
    <section class="card_section" id="C2">
        <p style="font-size: 11pt; vertical-align: middle;">
            <span style="font-weight: bold; vertical-align: middle;">Agreement Period :</span>
        </p>
        <p style="font-size: 11pt; vertical-align: middle;">
            <span style="vertical-align: middle;">{ClauseEnum.C2.value}</span>
        </p>
        <p style="border: 1px solid orange;padding:4px;font-size: 11pt; vertical-align: middle;">
            <span style="font-weight: bold; vertical-align: middle;">Description :</span>
            <span id="span_C2" style="vertical-align: middle;"></span>
        </p>
    </section>
    <section class="card_section" id="C3">
        <p style="font-size: 11pt; vertical-align: middle;">
            <span style="font-weight: bold; vertical-align: middle;">Support Details :</span>
        </p>
        <p style="font-size: 11pt; vertical-align: middle;">
            <span style="vertical-align: middle;">{ClauseEnum.C3.value}</span>
        </p>
        <p style="border: 1px solid orange;padding:4px;font-size: 11pt; vertical-align: middle;">
            <span style="font-weight: bold; vertical-align: middle;">Description :</span>
            <span id="span_C3" style="vertical-align: middle;"></span>
        </p>
    </section>
    <section class="card_section" id="C4">
        <p style="font-size: 11pt; vertical-align: middle;">
            <span style="font-weight: bold; vertical-align: middle;">Payment Terms :</span>
        </p>
        <p style="font-size: 11pt; vertical-align: middle;">
            <span style="vertical-align: middle;">{ClauseEnum.C4.value}</span>
        </p>
        <p style="border: 1px solid orange;padding:4px;font-size: 11pt; vertical-align: middle;">
            <span style="font-weight: bold; vertical-align: middle;">Description :</span>
            <span id="span_C4" style="vertical-align: middle;"></span>
        </p>
    </section>
    <section class="card_section" id="C5">
        <p style="font-size: 11pt; vertical-align: middle;">
            <span style="font-weight: bold; vertical-align: middle;">Non-Competition Clause :</span>
        </p>
        <p style="font-size: 11pt; vertical-align: middle;">
            <span style="vertical-align: middle;">{ClauseEnum.C5.value}</span>
        </p>
        <p style="border: 1px solid orange;padding:4px;font-size: 11pt; vertical-align: middle;">
            <span style="font-weight: bold; vertical-align: middle;">Description :</span>
            <span id="span_C5" style="vertical-align: middle;"></span>
        </p>
    </section>
    <section class="card_section" id="C6">
        <p style="font-size: 11pt; vertical-align: middle;">
            <span style="font-weight: bold; vertical-align: middle;">Termination :</span>
        </p>
        <p style="font-size: 11pt; vertical-align: middle;">
            <span style="vertical-align: middle;">{ClauseEnum.C6.value}</span>
        </p>
        <p style="border: 1px solid orange;padding:4px;font-size: 11pt; vertical-align: middle;">
            <span style="font-weight: bold; vertical-align: middle;">Description :</span>
            <span id="span_C6" style="vertical-align: middle;"></span>
        </p>
    </section>
    <section class="card_section" id="C7">
        <p style="font-size: 11pt; vertical-align: middle;">
            <span style="font-weight: bold; vertical-align: middle;">Survival Clause :</span>
        </p>
        <p style="font-size: 11pt; vertical-align: middle;">
            <span style="vertical-align: middle;">{ClauseEnum.C7.value}</span>
        </p>
        <p style="border: 1px solid orange;padding:4px;font-size: 11pt; vertical-align: middle;">
            <span style="font-weight: bold; vertical-align: middle;">Description :</span>
            <span id="span_C7" style="vertical-align: middle;"></span>
        </p>
    </section>
    <section class="card_section" id="C8">
        <p style="font-size: 11pt; vertical-align: middle;">
            <span style="font-weight: bold; vertical-align: middle;">Compliance with Other Laws :</span>
        </p>
        <p style="font-size: 11pt; vertical-align: middle;">
            <span style="vertical-align: middle;">{ClauseEnum.C8.value}</span>
        </p>
        <p style="border: 1px solid orange;padding:4px;font-size: 11pt; vertical-align: middle;">
            <span style="font-weight: bold; vertical-align: middle;">Description :</span>
            <span id="span_C8" style="vertical-align: middle;"></span>
        </p>
    </section>
    <section class="card_section" id="C9">
        <p style="font-size: 11pt; vertical-align: middle;">
            <span style="font-weight: bold; vertical-align: middle;">Insurance :</span>
        </p>
        <p style="font-size: 11pt; vertical-align: middle;">
            <span style="vertical-align: middle;">{ClauseEnum.C9.value}</span>
        </p>
        <p style="border: 1px solid orange;padding:4px;font-size: 11pt; vertical-align: middle;">
            <span style="font-weight: bold; vertical-align: middle;">Description :</span>
            <span id="span_C9" style="vertical-align: middle;"></span>
        </p>
    </section>
    <section class="card_section" id="C10">
        <p style="font-size: 11pt; vertical-align: middle;">
            <span style="font-weight: bold; vertical-align: middle;">Indemnity :</span>
        </p>
        <p style="font-size: 11pt; vertical-align: middle;">
            <span style="vertical-align: middle;">{ClauseEnum.C10.value}</span>
        </p>
        <p style="border: 1px solid orange;padding:4px;font-size: 11pt; vertical-align: middle;">
            <span style="font-weight: bold; vertical-align: middle;">Description :</span>
            <span id="span_C10" style="vertical-align: middle;"></span>
        </p>
    </section>
    <section class="card_section" id="C11">
        <p style="font-size: 11pt; vertical-align: middle;">
            <span style="font-weight: bold; vertical-align: middle;">Non-solicitation :</span>
        </p>
        <p style="font-size: 11pt; vertical-align: middle;">
            <span style="vertical-align: middle;">{ClauseEnum.C11.value}</span>
        </p>
        <p style="border: 1px solid orange;padding:4px;font-size: 11pt; vertical-align: middle;">
            <span style="font-weight: bold; vertical-align: middle;">Description :</span>
            <span id="span_C11" style="vertical-align: middle;"></span>
        </p>
    </section>
    <section class="card_section" id="C12">
        <p style="font-size: 11pt; vertical-align: middle;">
            <span style="font-weight: bold; vertical-align: middle;">Liabilities :</span>
        </p>
        <p style="font-size: 11pt; vertical-align: middle;">
            <span style="vertical-align: middle;">{ClauseEnum.C12.value}</span>
        </p>
        <p style="border: 1px solid orange;padding:4px;font-size: 11pt; vertical-align: middle;">
            <span style="font-weight: bold; vertical-align: middle;">Description :</span>
            <span id="span_C12" style="vertical-align: middle;"></span>
        </p>
    </section>
    <section class="card_section" id="C13">
        <p style="font-size: 11pt; vertical-align: middle;">
            <span style="font-weight: bold; vertical-align: middle;">Penalty Clause :</span>
        </p>
        <p style="font-size: 11pt; vertical-align: middle;">
            <span style="vertical-align: middle;">{ClauseEnum.C13.value}</span>
        </p>
        <p style="border: 1px solid orange;padding:4px;font-size: 11pt; vertical-align: middle;">
            <span style="font-weight: bold; vertical-align: middle;">Description :</span>
            <span id="span_C13" style="vertical-align: middle;"></span>
        </p>
    </section>
    <section class="card_section" id="C14">
        <p style="font-size: 11pt; vertical-align: middle;">
            <span style="font-weight: bold; vertical-align: middle;"> Non Exclusive :</span>
        </p>
        <p style="font-size: 11pt; vertical-align: middle;">
            <span style="vertical-align: middle;">{ClauseEnum.C14.value}</span>
        </p>
        <p style="border: 1px solid orange;padding:4px;font-size: 11pt; vertical-align: middle;">
            <span style="font-weight: bold; vertical-align: middle;">Description :</span>
            <span id="span_C14" style="vertical-align: middle;"></span>
        </p>
    </section>
    <section class="card_section" id="C15">
        <p style="font-size: 11pt; vertical-align: middle;">
            <span style="font-weight: bold; vertical-align: middle;"> Survival :</span>
        </p>
        <p style="font-size: 11pt; vertical-align: middle;">
            <span style="vertical-align: middle;">{ClauseEnum.C15.value}</span>
        </p>
        <p style="border: 1px solid orange;padding:4px;font-size: 11pt; vertical-align: middle;">
            <span style="font-weight: bold; vertical-align: middle;">Description :</span>
            <span id="span_C15" style="vertical-align: middle;"></span>
        </p>
    </section>
    <section class="card_section" id="C16">
        <p style="font-size: 11pt; vertical-align: middle;">
            <span style="font-weight: bold; vertical-align: middle;"> Dispute Resolution :</span>
        </p>
        <p style="font-size: 11pt; vertical-align: middle;">
            <span style="vertical-align: middle;">{ClauseEnum.C16.value}</span>
        </p>
        <p style="border: 1px solid orange;padding:4px;font-size: 11pt; vertical-align: middle;">
            <span style="font-weight: bold; vertical-align: middle;">Description :</span>
            <span id="span_C16" style="vertical-align: middle;"></span>
        </p>
    </section>
    <section class="card_section" id="C17">
        <p style="font-size: 11pt; vertical-align: middle;">
            <span style="font-weight: bold; vertical-align: middle;"> Intellectual Property Rights :</span>
        </p>
        <p style="font-size: 11pt; vertical-align: middle;">
            <span style="vertical-align: middle;">{ClauseEnum.C17.value}</span>
        </p>
        <p style="border: 1px solid orange;padding:4px;font-size: 11pt; vertical-align: middle;">
            <span style="font-weight: bold; vertical-align: middle;">Description :</span>
            <span id="span_C17" style="vertical-align: middle;"></span>
        </p>
    </section>
    <section class="card_section" id="C18">
        <p style="font-size: 11pt; vertical-align: middle;">
            <span style="font-weight: bold; vertical-align: middle;"> Anti-Corruption Clause :</span>
        </p>
        <p style="font-size: 11pt; vertical-align: middle;">
            <span style="vertical-align: middle;">{ClauseEnum.C18.value}</span>
        </p>
        <p style="border: 1px solid orange;padding:4px;font-size: 11pt; vertical-align: middle;">
            <span style="font-weight: bold; vertical-align: middle;">Description :</span>
            <span id="span_C18" style="vertical-align: middle;"></span>
        </p>
    </section>
    """
    updated_checklist = modify_checklist_html(html_content, result_data)
    updated_original_txt = extract_sections_with_id_and_content_from_html(file_html)
    matched_section = []
    for data in result_data:
        # print(data)
        score_list = []
        if data['clause_compliance'] == "YES":
            checklist_content = data['source']
            print('content--', checklist_content)
            for item in updated_original_txt:
                original_content = item['section_content']
                res = get_similarity_score(checklist_content, original_content)
                if res['is_similar']:
                    score_list.append({'section_id': item['section_id'], 'cosine_score': res['cosine_score']})
            print('score', score_list)
            if score_list:
                best_match = max(score_list, key=lambda x: x['cosine_score'])
                matched_section.append({'clause_id': data['clause_id'], 'section_id': best_match['section_id']})
    print('matched----', matched_section)
    file_html = modify_original_html(file_html, matched_section)
    response = {
        'success': True,
        'message': 'Checklist updated successfully',
        'file_html': file_html,
        'file_name': file_name,
        'updated_checklist': updated_checklist,
        'matched_section': matched_section,
    }
    return response