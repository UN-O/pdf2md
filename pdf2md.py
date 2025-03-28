import concurrent.futures
import base64
import os
import json
import re
import time
import http.client
import requests

from PIL import Image, ImageDraw

from pathlib import Path
from io import BytesIO
from mistralai.models import OCRResponse

from mistralai import Mistral
from mistralai import DocumentURLChunk, ImageURLChunk, TextChunk


# 初始化 OpenAI 客戶端 (請務必填入正確的 API 金鑰)
from openai import OpenAI
import tiktoken 

# import API key from .env file
from dotenv import load_dotenv
load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")
UPLOADTHING_API_KEY = os.getenv("UPLOADTHING_API_KEY")

if not MISTRAL_API_KEY:
    raise ValueError("MISTRAL_API_KEY not found in environment variables")
if not OPEN_AI_API_KEY:
    raise ValueError("OPEN_AI_API_KEY not found in environment variables")

clientMistral = Mistral(api_key=MISTRAL_API_KEY)
client = OpenAI(api_key=OPEN_AI_API_KEY)

COST_RATES = {
    "gpt-4o": {"prompt": 2.5 / 1000000, "completion": 10 / 1000000},
    "gpt-4o-mini": {"prompt": 0.15 / 1000000, "completion": 0.6 / 1000000},
    "o3-mini": {"prompt": 1.1 / 1000000, "completion": 4.4 / 1000000},
}

global_openai_cost = 0.0  # 全局累計 OpenAI 花費

class Timer:
    def __init__(self, stage_name):
        self.stage_name = stage_name

    def __enter__(self):
        self.start = time.time()
        print(f"{self.stage_name} entering...")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end = time.time()
        self.elapsed = self.end - self.start
        print(f"{self.stage_name} (finished {self.elapsed:.2f} s)")

def tracked_openai_call(api_func, stage_name, model_name):
    global global_openai_cost
    rates = COST_RATES.get(model_name, {"prompt": 2.5 / 1000000, "completion": 10 / 1000000})
    with Timer(stage_name):
        response = api_func()  # 呼叫實際的 API 函式
    usage = response.usage
    prompt_tokens = usage.prompt_tokens
    completion_tokens = usage.completion_tokens
    total_tokens = prompt_tokens + completion_tokens
    cost = prompt_tokens * rates["prompt"] + completion_tokens * rates["completion"]
    global_openai_cost += cost
    print(f"{stage_name} usage(i/o): {prompt_tokens} /  {completion_tokens} tokens, ${cost:.6f}")
    return response

# 計算 token 數量的函式
def count_tokens(text, model="gpt-4o"):
    try:
        encoding = tiktoken.encoding_for_model(model)
    except Exception:
        return len(text.split())
    return len(encoding.encode(text))

def md_analysis(md_path, output_folder,  need_summary, need_preview, need_writing_analysis, requirement, model="o3-mini"):
    global global_openai_cost
    global_openai_cost = 0.0
    os.makedirs(output_folder, exist_ok=True)
    
    pdf_name = os.path.splitext(os.path.basename(md_path))[0]
    with open(md_path, "r", encoding="utf-8") as f:
        full_article_md = f.read()  
    
    # Start the timer
    overall_start = time.time()
    
    with Timer("////// STEP1: generate summary & preview ////// "):
        # 確保目標資料夾存在
        
        if need_summary:
            print("Generating summary...")
            summary_folder = os.path.join(output_folder, "論文整理")
            os.makedirs(summary_folder, exist_ok=True)
            summary = generate_summary(full_article_md, requirement, model="o3-mini")
            summary_path = os.path.join(summary_folder, f"{pdf_name}_summary.md")
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(summary)
            print(f"Summary file saved at: {summary_path}")

        if need_preview:
            print("Generating preview...")
            preview_folder = os.path.join(output_folder, "論文預覽")
            os.makedirs(preview_folder, exist_ok=True)
            preview = generate_preview(full_article_md, requirement, model="o3-mini")
            preview_path = os.path.join(preview_folder, f"{pdf_name}_preview.md")
            with open(preview_path, "w", encoding="utf-8") as f:
                f.write(preview)
            print(f"Preview file saved at: {preview_path}")
        
        if need_writing_analysis:
            print("Generating writing analysis...")
            writing_folder = os.path.join(output_folder, "寫作分析")
            os.makedirs(writing_folder, exist_ok=True)
            writing_analysis = generate_writing_analysis(full_article_md, model="o3-mini")
            writing_path = os.path.join(writing_folder, f"{pdf_name}_writing.md")
            with open(writing_path, "w", encoding="utf-8") as f:
                f.write(writing_analysis)
            print(f"Writing analysis file saved at: {writing_path}")
        
    print("\n\n")
    
    overall_elapsed = time.time() - overall_start
    token_count = count_tokens(full_article_md, model="gpt-3.5-turbo")
    
    # 統一在處理完畢後顯示所有統計資訊
    summary_message = (
        "==========================================\n"
        "PDF 處理完成！\n"
        f"總耗時：{overall_elapsed:.2f} 秒\n"
        f"累計 OpenAI API 花費：${global_openai_cost:.6f}\n"
        f"Output.md token 數：{token_count} tokens\n"
        "==========================================\n\n"
    )
    print(summary_message)
    
#####################################################
# OCR mistral
#####################################################
def pdf2md(pdf_path, output_folder,  need_summary, need_preview, need_writing_analysis, requirement):
    global global_openai_cost
    # 每次處理前先重置累計花費
    global_openai_cost = 0.0
    os.makedirs(output_folder, exist_ok=True)
    
    # Start the timer
    overall_start = time.time()
    
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    
    # Extract text and images from the PDF
    with Timer("////// STEP1: OCR pdf content & upload img ////// "):
        pdf_file = Path(pdf_path)
        full_article_md = ocr_mistral(pdf_file)
        
        md_path = os.path.join(output_folder, f"{pdf_name}.md")
        with open(md_path, "w", encoding="utf-8") as md_file:
            md_file.write(full_article_md)
        print(f"Conversion and refinement complete! Check the folder: {output_folder}")
        print(f"Markdown file saved at: {md_path}")
    print("\n\n")
    
    # Final step: generate summary from the complete Markdown (in Chinese)
    with Timer("////// STEP2: generate summary & preview ////// "):
        # 確保目標資料夾存在
        
        if need_summary:
            print("Generating summary...")
            summary_folder = os.path.join(output_folder, "論文整理")
            os.makedirs(summary_folder, exist_ok=True)
            summary = generate_summary(full_article_md, requirement, model="o3-mini")
            summary_path = os.path.join(summary_folder, f"{pdf_name}_summary.md")
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(summary)
            print(f"Summary file saved at: {summary_path}")

        if need_preview:
            print("Generating preview...")
            preview_folder = os.path.join(output_folder, "論文預覽")
            os.makedirs(preview_folder, exist_ok=True)
            preview = generate_preview(full_article_md, requirement, model="o3-mini")
            preview_path = os.path.join(preview_folder, f"{pdf_name}_preview.md")
            with open(preview_path, "w", encoding="utf-8") as f:
                f.write(preview)
            print(f"Preview file saved at: {preview_path}")
        
        if need_writing_analysis:
            print("Generating writing analysis...")
            writing_folder = os.path.join(output_folder, "寫作分析")
            os.makedirs(writing_folder, exist_ok=True)
            writing_analysis = generate_writing_analysis(full_article_md, model="o3-mini")
            writing_path = os.path.join(writing_folder, f"{pdf_name}_writing.md")
            with open(writing_path, "w", encoding="utf-8") as f:
                f.write(writing_analysis)
            print(f"Writing analysis file saved at: {writing_path}")
        
    print("\n\n")
    
    overall_elapsed = time.time() - overall_start
    token_count = count_tokens(full_article_md, model="gpt-3.5-turbo")
    
    # 統一在處理完畢後顯示所有統計資訊
    summary_message = (
        "==========================================\n"
        "PDF 處理完成！\n"
        f"總耗時：{overall_elapsed:.2f} 秒\n"
        f"累計 OpenAI API 花費：${global_openai_cost:.6f}\n"
        f"Output.md token 數：{token_count} tokens\n"
        "==========================================\n\n"
    )
    print(summary_message)

def ocr_mistral(pdf_file):
    uploaded_file = clientMistral.files.upload(
        file={
            "file_name": pdf_file.stem,
            "content": pdf_file.read_bytes(),
        },
        purpose="ocr",
    )

    signed_url = clientMistral.files.get_signed_url(file_id=uploaded_file.id, expiry=1)

    pdf_response = clientMistral.ocr.process(document=DocumentURLChunk(document_url=signed_url.url), model="mistral-ocr-latest", include_image_base64=True)
    full_article_md = get_combined_markdown(pdf_response)
    return full_article_md

def generate_image_caption_base64(image_base64: str, model="gpt-4o"):
    # Read the image and encode it as Base64
    if image_base64.startswith("data:"):
      image_base64 = image_base64.split(",", 1)[1]
    data_uri = f"data:image/jpeg;base64,{image_base64}"

    # English prompt for structured output
    prompt = (
        "Based on the content of the provided figure from a academic paper, generate a concise and appropriate caption in detail."
        "You should write down every detail information of the figure in caption"
        "Please output your result strictly as a JSON object in the following format:\n"
        '{"caption": "<your caption in pure text>", "title": "<your title>"}'
    )

    # Define the JSON schema for structured output
    json_schema = {
        "name": "image_description",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "caption": {
                    "type": "string",
                    "description": "A detail description of the figure content."
                },
                "title": {
                    "type": "string",
                    "description": "A short title summarizing the image."
                }
            },
            "required": ["caption", "title"],
            "additionalProperties": False
        }
    }

    def api_call():
        return client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a multimodal AI assistant that generates image captions and titles."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_uri}}
                    ],
                }
            ],
            temperature=0.7,
            response_format={
                "type": "json_schema",
                "json_schema": json_schema
            }
        )

    # Default result if API call fails or times out
    default_result = {"caption": "Timeout or error", "title": "Timeout or error"}
    response = tracked_openai_call(
        lambda: call_api_with_retry(api_call, timeout=15, retries=3, default_result=default_result),
        stage_name=f"        image caption generation by {model}",
        model_name=model
    )
    # If the response is already a dict (default), use it; else parse the structured JSON result
    if isinstance(response, dict):
        result = response
    else:
        try:
            raw_response = response.choices[0].message.content
            result = json.loads(raw_response)
        except (AttributeError, json.JSONDecodeError):
            result = {"caption": raw_response, "title": ""}

    return result

def upload_image_to_uploadthing_from_base64(img_name: str, image_base64: str) -> str:
    """
    利用 base64 字串上傳圖片到 UploadThing 並回傳檔案 URL。
    如果 base64 字串包含 data URI 前綴 (如 "data:image/jpeg;base64,"), 則將其移除。
    """
    # 移除 data URI 前綴（如果存在）
    if image_base64.startswith("data:"):
        image_base64 = image_base64.split(",", 1)[1]

    # 將 base64 字串轉換回二進位資料
    image_data = base64.b64decode(image_base64)

    # 建立 HTTPS 連線
    conn = http.client.HTTPSConnection("api.uploadthing.com")

    # 準備上傳的 payload
    prepare_payload = json.dumps({
        "fileName": img_name,
        "fileSize": len(image_data),
        "slug": "imageUploader",
        "fileType": "image/jpeg",  # 根據實際情況調整 MIME type
        "customId": "my-id-123",
        "contentDisposition": "inline",
        "acl": "public-read",
        "expiresIn": 3600
    })

    headers = {
        'X-Uploadthing-Api-Key': UPLOADTHING_API_KEY,
        'Content-Type': 'application/json'
    }

    # 發送準備上傳的請求
    conn.request("POST", "/v7/prepareUpload", prepare_payload, headers)
    res = conn.getresponse()
    prepare_data = res.read()
    prepare_response = json.loads(prepare_data.decode("utf-8"))

    # 準備實際上傳的 payload
    upload_payload = json.dumps({
        "files": [
            {
                "name": img_name,
                "size": len(image_data),
                "type": "image/jpeg",  # 根據實際情況調整 MIME type
                "customId": None,
                "content": image_base64  # 直接使用移除前綴後的 base64 編碼內容
            }
        ],
        "acl": "public-read",
        "metadata": None,
        "contentDisposition": "inline"
    })

    # 發送實際上傳的請求
    conn.request("POST", "/v6/uploadFiles", upload_payload, headers)
    res = conn.getresponse()
    upload_data = res.read()
    upload_response = json.loads(upload_data.decode("utf-8"))

    # 從回應中提取預簽名 URL 與其他上傳資訊
    upload_info = upload_response["data"][0]
    presigned_url = upload_info["url"]
    fields = upload_info["fields"]

    # 使用 BytesIO 建立檔案物件供上傳
    file_obj = BytesIO(image_data)
    file_obj.name = img_name  # 設定檔名

    files = {'file': (img_name, file_obj, 'image/jpeg')}
    response = requests.post(presigned_url, data=fields, files=files)

    if response.status_code == 204:
        return upload_info["fileUrl"]
    else:
        raise Exception("Image upload failed")

def replace_images_in_markdown(markdown_str: str, images_dict: dict) -> str:
    """
    將 Markdown 中的圖片標記，替換成上傳到 UploadThing 後取得的 URL。
    假設 images_dict 的結構為 {圖片名稱: base64 編碼字串}
    """
    for img_name, base64_str in images_dict.items():
        print(f"      Processing image {img_name}...")
        try:
            image_desc = generate_image_caption_base64(base64_str)
            caption = image_desc.get("caption", "No caption generated")
            title = image_desc.get("title", "No title generated")
            print(f"          Uploading image {title}...")
            url = upload_image_to_uploadthing_from_base64(img_name, base64_str)
            print(f"          Finish Uploading image {title}...")
            # 將 Markdown 中對應的圖片標記替換，例如將 ![img_name](img_name) 替換為 ![img_name](url)
            markdown_str = markdown_str.replace(f"![{img_name}]({img_name})", f"![{caption}]({url}) \nTitle:**{title}**\n")
        except Exception as e:
            print(f"上傳 {img_name} 失敗: {e}")
    return markdown_str

def get_combined_markdown(ocr_response: OCRResponse) -> str:
    markdowns: list[str] = []
    for page in ocr_response.pages:
        # 假設 page.images 的結構中，img.id 為圖片名稱，img.image_base64 為圖片的 base64 編碼字串
        image_data = {img.id: img.image_base64 for img in page.images}
        markdowns.append(replace_images_in_markdown(page.markdown, image_data))
    return "\n\n".join(markdowns)

def call_api_with_timeout(func, timeout=20):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(func)
        return future.result(timeout=timeout)

def call_api_with_retry(func, timeout=20, retries=3, default_result=None):
    """
    嘗試呼叫 func，如果在 timeout 秒內沒有成功（或發生例外），則重試最多 retries 次。
    若仍失敗，則回傳 default_result。
    """
    for attempt in range(retries + 1):
        try:
            result = call_api_with_timeout(func, timeout=timeout)
            return result
        except Exception as e:
            print(f"API 呼叫失敗，嘗試重新呼叫 (第 {attempt+1} 次)... 錯誤類型: {type(e).__name__}, 詳細: {e}")
    return default_result

def upload_image_to_uploadthing(image_path):
    # 讀取圖片檔案並轉換為 base64 編碼
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
        image_base64 = base64.b64encode(image_data).decode('utf-8')

    # 建立 HTTPS 連線
    conn = http.client.HTTPSConnection("api.uploadthing.com")

    # 設定準備上傳的 payload
    prepare_payload = json.dumps({
        "fileName": os.path.basename(image_path),  # 檔案名稱
        "fileSize": len(image_data),  # 檔案大小（以 byte 為單位）
        "slug": "imageUploader",
        "fileType": "image/png",  # 檔案類型
        "customId": "my-id-123",
        "contentDisposition": "inline",
        "acl": "public-read",
        "expiresIn": 3600
    })

    # 設定請求的 headers
    headers = {
        'X-Uploadthing-Api-Key': "sk_live_b57a064b9e74e14eacad5bd073ed33f42c3e115d5fe048f31d253430b6d958e4",
        'Content-Type': 'application/json'
    }

    # 發送準備上傳的 POST 請求
    conn.request("POST", "/v7/prepareUpload", prepare_payload, headers)

    # 取得回應
    res = conn.getresponse()
    prepare_data = res.read()

    # 解析準備上傳的回應
    prepare_response = json.loads(prepare_data.decode("utf-8"))

    # 設定實際上傳的 payload
    upload_payload = json.dumps({
        "files": [
            {
                "name": os.path.basename(image_path),  # 檔案名稱
                "size": len(image_data),  # 檔案大小（以 byte 為單位）
                "type": "image/png",  # 檔案類型
                "customId": None,
                "content": image_base64  # 圖片的 base64 編碼內容
            }
        ],
        "acl": "public-read",
        "metadata": None,
        "contentDisposition": "inline"
    })

    # 發送實際上傳的 POST 請求
    conn.request("POST", "/v6/uploadFiles", upload_payload, headers)

    # 取得回應
    res = conn.getresponse()
    upload_data = res.read()

    # 解析實際上傳的回應
    upload_response = json.loads(upload_data.decode("utf-8"))

    # 提取預簽名 URL 和其他必要的資訊
    upload_info = upload_response["data"][0]
    presigned_url = upload_info["url"]
    fields = upload_info["fields"]

    # 使用預簽名 URL 上傳圖片
    files = {'file': (image_path, open(image_path, 'rb'), 'image/png')}
    response = requests.post(presigned_url, data=fields, files=files)

    # 確認上傳成功
    if response.status_code == 204:
        return upload_info["fileUrl"]
    else:
        raise Exception("Image upload failed")

# prompting text
def generate_summary(full_md, requirement="",model="o3-mini"):
    prompt = f"""
    Based on the complete Markdown content provided below, please generate a structured and comprehensive summary in Markdown format. 使用繁體中文回答，每個分析內容至少一個段落。
    
    {f"請確保滿足特別需求:{requirement}" if len(requirement) > 0 else ""}

    Your summary should seamlessly integrate the following aspects into a coherent Markdown document with proper formatting (using H2, H3, blockquotes, etc.):

    - **Plain Language Title**: Provide a clear title (using H2 or H3) that encapsulates the article's main theme.
    - use quote block to summary the article in one sentence.
    - **Key Highlights**: Emphasize the most distinctive and eye-catching parts of the article. Use blockquotes where appropriate.
    - **Problem Awareness**: Describe the main research problem or question the article addresses.
    - **Article Structure**: Offer an overview of the article’s organization and main sections using appropriate headings.
    - **Research Methodology & Experimental Design**: Summarize the research methods and include a dedicated section discussing the experimental design. 情詳細描述實驗設計的細節，包含參與者背景、實驗執行方式、數據收集方法等
    - **Quantitative Data Extraction**: Identify and highlight any significant numerical data mentioned in the content (e.g., measurements, statistics, key figures), ensuring these numbers are clearly presented alongside the text. 並且用 quote block 來解釋這個數據對這篇研究的意義，至少擷取 1~5 個量化結果
    - **Key Findings & Conclusions**: Concisely summarize the core findings and conclusions.
    - **Limitations & Future Directions**: Discuss the study’s limitations and propose suggestions for future research.
    - **FAQ**: Develop a series of questions related to the field or the assumptions in the article—from basic to advanced. Each FAQ should be presented as its own toggle (collapsible) list. The answers should be derived from the article’s content and include any numerical data mentioned. 除了基本問題，至少提出 3 個進階專業質疑問題
    - **Critique**: Provide a commentary on the article’s strengths and weaknesses. 
    - **專有名詞 table**: 列出這篇文章中所有的專有名詞，並解釋其意義
    
    - **Plain-Language Version**: Write a plain-language version of the article in a research-paper style (at least 3000 Chinese characters). 請參考整篇文章，進行完整的論述，不要單純擷取摘要。
    
    - **background**: 介紹這個研究單位、這個學者的背景，哪個國家、哪間學校，他們過去做些什麼研究
    - **文獻回顧整理**: 將論文中「文獻回顧」、「introduction」中所有提及到的其他研究論述引用整理成 Markdown table，包含以下欄位
        - 論述內容(中文)，請用粗體標示關鍵詞與重點
        - 作者
        - 年代
        - 文章標題
        - 論文內引用這篇文獻時的原文內容
        - 跟這篇論文的重要程度關聯性分數 (0~5) 分，5 分代表跟這篇論文關聯性最高，或是不斷提及，0 分代表只有稍微提到


    Please output only the final Markdown formatted summary without any additional JSON objects, code markers, or extra explanations.

    論文的 Full Markdown content:
    ----------------
    {full_md}
    ----------------
    """

    def api_call():
        return client.chat.completions.create(
            model=model,
            reasoning_effort="high",
            messages=[
                {"role": "system", "content": "You are an expert in research article summarization, data analysis, and Markdown formatting."},
                {"role": "user", "content": prompt}
            ],
        )

    default_result = {"choices": [{"message": {"content": ""}}], "usage": {"prompt_tokens": 0, "completion_tokens": 0}}
    
    response = tracked_openai_call(lambda: call_api_with_retry(api_call, timeout=600, retries=3, default_result=default_result), stage_name=f"generate_summary by {model}", model_name=model)
    
    try:
        summary_md = response.choices[0].message.content.strip()
    except (AttributeError, json.JSONDecodeError):
        summary_md = ""
    
    return summary_md

def generate_writing_analysis(full_md, model="o3-mini"):
    prompt = f"""
    我正在學習如何撰寫英文論文，請你賞析以下別的學者完整的 Markdown 論文內容，擷取這篇文章寫得好的段落，分析寫作技巧，以及我可以參考或學習模仿的論述結構與文法。此外特別整理出使用得很好的用詞單字與文法，讓我可以模仿學習造句。
    ----------------
    **需要分析整理的完整 Markdown 內容：**
    {full_md}
    ----------------
    """

    def api_call():
        return client.chat.completions.create(
            model=model,
            reasoning_effort="high",
            messages=[
                {"role": "system", "content": "You are an expert in research article summarization, data analysis, and Markdown formatting."},
                {"role": "user", "content": prompt}
            ],
        )

    default_result = {"choices": [{"message": {"content": ""}}], "usage": {"prompt_tokens": 0, "completion_tokens": 0}}
    
    response = tracked_openai_call(lambda: call_api_with_retry(api_call, timeout=600, retries=3, default_result=default_result), stage_name=f"generate_preview by {model}", model_name=model)
    
    try:
        summary_md = response.choices[0].message.content.strip()
    except (AttributeError, json.JSONDecodeError):
        summary_md = ""
    
    return summary_md

def generate_preview(full_md, requirement="", model="o3-mini"):
    prompt = f"""
    根據以下完整的 Markdown 內容，請撰寫一篇 **完整、流暢、帶有口語化風格的學術文章**，而不是生硬的研究摘要。你的文章應該讓讀者覺得這是一個有深度的解說，而不是機械地羅列數據。
    
    {f"請確保滿足特別需求:{requirement}" if len(requirement) > 0 else ""}

    # **寫作風格**
    1. **口語化**：用「在這篇研究中，我們可以看到...」、「最有趣的發現是...」、「你可能會好奇...」「在這篇論文中，最重要的核心重點是... 、我們可以從這張圖片看到... 這個數據細節不重要，重點是...都很顯著... 、我們可以從中認識到說...」這樣的語氣，讓文章更自然易讀。
    2. **融會貫通**：不只是列數據，而是解釋「這個數據代表什麼？為什麼重要？」
    3. **整合數據與圖表**：數據和圖表必須自然融入文章，而不是孤立列出。
    4. **完整 Markdown 格式**：包含標題、表格、圖片、引用區塊等格式化內容。
    5. 文章結構寫的 H2 是參考用的，請自行更換用詞或刪減
    6. 可以在各個段落放入圖片來協助講解概念

    ---
    
    # **文章結構**
    在最一剛開始，寫我們今天要來看(什麼單位、什麼人)的論文，你可以預期在這篇文章中認識(什麼樣的知識)、或是在最一剛開始進行提問來引發閱讀的興趣
    ## **研究核心概念**
    - 以口語化方式解釋研究主題與有趣之處，例如：
      > 「這篇論文的核心概念是探索 XYZ 現象，簡單來說，就是看看...」

    ## **研究背景與問題**
    - 介紹研究的問題意識，並以輕鬆的方式解釋：

    ## **研究方法**
    - 介紹研究方法時，避免過於技術化，而是讓讀者理解實驗是怎麼做的：
      > 「研究團隊的方法很有意思，他們選擇了...」

    ## **重要數據與圖表解析**
    **這是最重要的部分，請確保數據、圖片與敘述結合，而不是單獨列出！**
    
    - 先用敘述引導數據，例如：
      > 「最讓人驚訝的是，當我們把 XYZ 變數提高時，結果竟然是...」
    
    - 插入數據表格範例(請注意這只是範例，如果要分析的文章內沒有表格就請直接略過)：
      
      | 這是 | 格式 | 範例 |
      |------|------|------|
      | 內容 | 內容 | 內容  |
      | 內容 | 內容 | 內容  |
      
      
    
    - 插入圖片範例：(請注意這只是範例，如果要分析的文章內沒有圖片就請直接略過)
      
      ![XYZ 變化趨勢](圖片連結)
      
    
    - **用口語方式解析圖片與數據** 這裡舉幾個範例，請以需要分析整理的完整 Markdown 內容來發展
      > 「從這張圖我們可以直覺地看到，當 X 增加時，Y 的變化很明顯。」
      > 「這個數據細節不重要，重點是它顯示了一個趨勢...」
      > 「這個發現其實蠻有趣的，因為我們原本預期的是...」

    ## **研究結論**
    - 不是單純總結，而是以「帶領讀者思考」的方式寫：
      > 「所以，我們可以從這個研究中學到什麼？最重要的結論是...」

    ## **這個研究的意義**
    - 這個研究對產業或學界有什麼影響？
      > 「這個發現可能改變我們對 XYZ 的理解，未來我們可能會在...」
    - 撰寫 **Takeaways**: Present a bullet-point list summarizing the main points.

    ## **限制與未來發展**
    - 討論研究限制時，可以用：
      > 「當然，這個研究不是完美的，還有幾個地方值得進一步探討...」
    
    ## **FAQ**: 
    - Develop a series of questions related to the field or the assumptions in the article—from basic to advanced. Each FAQ should be presented as its own toggle (collapsible) list. The answers should be derived from the article’s content and include any numerical data mentioned. 除了基本問題，至少提出 3 個進階專業質疑問題
    ---
    
    ## 延伸閱讀
    - 提供這篇論文主要參考的文獻，讓讀者可以進一步閱讀。並提供一段簡短的解說，這些文獻對這篇論文的重要性(提供了這篇論文怎樣的論點)。
    
    **請確保輸出是一篇完整、流暢、有邏輯的 Markdown 文章，而不是列表式的摘要！**
    
    ----------------
    **需要分析整理的完整 Markdown 內容：**
    {full_md}
    ----------------
    """

    def api_call():
        return client.chat.completions.create(
            model=model,
            reasoning_effort="high",
            messages=[
                {"role": "system", "content": "You are an expert in research article summarization, data analysis, and Markdown formatting."},
                {"role": "user", "content": prompt}
            ],
        )

    default_result = {"choices": [{"message": {"content": ""}}], "usage": {"prompt_tokens": 0, "completion_tokens": 0}}
    
    response = tracked_openai_call(lambda: call_api_with_retry(api_call, timeout=600, retries=3, default_result=default_result), stage_name=f"generate_preview by {model}", model_name=model)
    
    try:
        summary_md = response.choices[0].message.content.strip()
    except (AttributeError, json.JSONDecodeError):
        summary_md = ""
    
    return summary_md

def generate_image_caption(img_path, model="gpt-4o"):
    """
    Generate an image caption and title from the provided image.
    The API returns a JSON object in the following format:
      {"caption": <image caption>, "title": <image title>}
    If the API call fails (e.g. due to a timeout), it will retry once;
    if it still fails, a default value will be returned.
    """
    # Read the image and encode it as Base64
    with open(img_path, "rb") as f:
        image_data = f.read()
    image_base64 = base64.b64encode(image_data).decode("utf-8")
    
    # Determine the file extension and set MIME type accordingly (default to jpeg)
    ext = os.path.splitext(img_path)[1].strip(".").lower()
    if ext not in {"jpg", "jpeg", "png", "gif"}:
        ext = "jpeg"
    data_uri = f"data:image/{ext};base64,{image_base64}"
    
    # English prompt for structured output
    prompt = (
        "Based on the content of the provided figure from a academic paper, generate a concise and appropriate caption in detail"
        "and a short title. Please output your result strictly as a JSON object in the following format:\n"
        '{"caption": "<your caption>", "title": "<your title>"}'
    )
    
    # Define the JSON schema for structured output
    json_schema = {
        "name": "image_description",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "caption": {
                    "type": "string",
                    "description": "A concise description of the figure content."
                },
                "title": {
                    "type": "string",
                    "description": "A short title summarizing the image."
                }
            },
            "required": ["caption", "title"],
            "additionalProperties": False
        }
    }
    
    def api_call():
        return client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a multimodal AI assistant that generates image captions and titles."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_uri}}
                    ],
                }
            ],
            temperature=0.7,
            response_format={
                "type": "json_schema",
                "json_schema": json_schema
            }
        )
    
    # Default result if API call fails or times out
    default_result = {"caption": "Timeout or error", "title": "Timeout or error"}
    
    response = tracked_openai_call(
        lambda: call_api_with_retry(api_call, timeout=10, retries=3, default_result=default_result),
        stage_name=f"        image caption generation by {model}",
        model_name=model
    )
    
    # If the response is already a dict (default), use it; else parse the structured JSON result
    if isinstance(response, dict):
        result = response
    else:
        try:
            raw_response = response.choices[0].message.content
            result = json.loads(raw_response)
        except (AttributeError, json.JSONDecodeError):
            result = {"caption": raw_response, "title": ""}
    
    return result
