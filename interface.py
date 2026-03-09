import streamlit as st
import json
import re
import requests
import PyPDF2
import os
from io import BytesIO
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL", "http://34.70.252.201:8000")

# ──────────────────────────────────────────────
# CONFIG (Matching user snippet)
# ──────────────────────────────────────────────
MAX_INPUT_TOKENS = 4800   # tokens per chunk sent to the model
MAX_NEW_TOKENS = 1000       # max tokens the model generates per chunk

# ──────────────────────────────────────────────
# SYSTEM PROMPT BUILDER (Matching user snippet)
# ──────────────────────────────────────────────
def build_system_instruction(role: str) -> str:
    role_focus = (
        "You are advising the CLIENT (buyer). Focus on financial exposure, "
        "restrictions, loss of flexibility, termination risk, and IP limitations."
        if role == "Client"
        else
        "You are advising the CONTRACTOR (supplier). Focus on liability exposure, "
        "IP ownership risk, performance obligations, termination risk, and revenue uncertainty."
    )
    return f"""You are a Senior Legal Counsel performing an expert contract review.

{role_focus}

Return STRICT JSON only.

Tasks:
1. executive_summary
2. important_clauses
3. key_highlights
4. uncertainties_and_risks
"""

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def extract_json(text: str):
    """Pull the first JSON object out of a model response."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None

def chunk_text_approx(text: str) -> list[str]:
    """
    Since we don't have the tokenizer locally, we use a character-based 
    heuristic to approximate MAX_INPUT_TOKENS. 
    Llama 3 tokens are roughly 4 characters on average for English.
    """
    chars_per_chunk = MAX_INPUT_TOKENS * 4
    chunks = []
    for i in range(0, len(text), chars_per_chunk):
        chunks.append(text[i : i + chars_per_chunk])
    return chunks

# ──────────────────────────────────────────────
# INFERENCE (via Backend API)
# ──────────────────────────────────────────────
def generate_stream_api(chunk: str, role: str):
    """Send a request to the backend API for a streaming response."""
    
    system_instr = build_system_instruction(role)
    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_instr}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{chunk}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    payload = {
        "prompt": prompt,
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": 0.0,
        "top_p": 0.9
    }

    try:
        # Use stream=True to handle the StreamingResponse
        response = requests.post(f"{BACKEND_URL}/generate_stream", json=payload, stream=True, timeout=180)
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                yield decoded_line
                
    except Exception as e:
        yield f"Error connecting to backend: {str(e)}"

def analyze_contract_api(text: str, role: str) -> str:
    """Chunk the full contract and call the streaming API for each piece."""
    chunks = chunk_text_approx(text)
    
    full_output = ""
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, chunk in enumerate(chunks):
        status_text.text(f"Analyzing section {i + 1} of {len(chunks)} via Streaming API…")
        
        # Create a container for the streaming output of this chunk
        with st.container():
            st.markdown(f"### Section {i+1} Analysis")
            output_container = st.empty()
            chunk_output = ""
            
            for token in generate_stream_api(chunk, role):
                chunk_output += token
                # Update the container in real-time
                # Note: This gives immediate visual feedback
                output_container.markdown(chunk_output + "▌")
            
            output_container.markdown(chunk_output) # Final update without cursor
            full_output += chunk_output + "\n"
        
        progress_bar.progress((i + 1) / len(chunks))

    status_text.text("Analysis complete ✓")
    return full_output

# ──────────────────────────────────────────────
# STREAMLIT UI
# ──────────────────────────────────────────────
st.set_page_config(page_title="AI Legal Contract Analyzer", page_icon="⚖️", layout="wide")

st.title("⚖️ AI Legal Contract Analyzer")
st.caption(f"Remote Backend: {BACKEND_URL} (Streaming Enabled)")

role = st.radio(
    "Reviewing from the perspective of:",
    ["Client", "Contractor"],
    horizontal=True,
)

file = st.file_uploader(
    "Upload Contract (PDF or TXT)",
    type=["pdf", "txt"],
)

if file and st.button("Analyze Contract", type="primary"):

    # ── Read file ──────────────────────────────
    if file.type == "application/pdf":
        reader = PyPDF2.PdfReader(BytesIO(file.read()))
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
    else:
        text = file.read().decode("utf-8", errors="ignore")

    if not text.strip():
        st.error("Could not extract any text from the uploaded file.")
        st.stop()

    st.info(f"Contract loaded — {len(text):,} characters.")

    # ── Run analysis ───────────────────────────
    # We display results directly during analyze_contract_api now
    raw_result   = analyze_contract_api(text, role)
    json_result  = extract_json(raw_result)

    # ── Display processed results (Optional if streaming already showed it) ──
    if json_result:
        st.success("Analysis Parsed Successfully!")
        # Render each top-level key as a nice expandable section
        for section, content in json_result.items():
            label = section.replace("_", " ").title()
            with st.expander(f"📌 {label}", expanded=False):
                if isinstance(content, list):
                    for item in content:
                        st.markdown(f"- {item}")
                elif isinstance(content, dict):
                    for k, v in content.items():
                        st.markdown(f"**{k}:** {v}")
                else:
                    st.write(content)

        st.divider()
        with st.expander("Raw JSON"):
            st.json(json_result)
    else:
        st.warning("Could not parse full JSON from combined chunks. See raw outputs above.")

# Sidebar info
st.sidebar.title("System Info")
st.sidebar.markdown(f"**Backend URL:** `{BACKEND_URL}`")
try:
    health_resp = requests.get(f"{BACKEND_URL}/health", timeout=5)
    if health_resp.status_code == 200:
        st.sidebar.success("Backend Status: Online")
        st.sidebar.json(health_resp.json())
    else:
        st.sidebar.error(f"Backend Status: Error {health_resp.status_code}")
except:
    st.sidebar.error("Backend Status: Offline")
