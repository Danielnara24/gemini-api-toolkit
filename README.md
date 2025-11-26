# Gemini Automation Kit

A robust, lightweight Python wrapper for the [Google Gemini API](https://ai.google.dev/gemini-api/docs) (using the official `google-genai` SDK). 

This kit simplifies building automation tools by abstracting the complexity of managing multimodal inputs, tools, and response formatting. It provides a clean interface for handling videos, PDFs, "Thinking" models, and structured JSON responses.

**Now fully compatible with the Gemini 3 model family.**

## Features & Core Functions

This kit provides **three specialized functions**, each tailored to specific model generations and capabilities.

### 1. Multimodal & Structured Data (Gemini 3)
Use **`prompt_gemini_3`** for the `gemini-3-pro-preview` family. This function unlocks the newest API capabilities:
*   **Combined Modes:** Supports **Tools (Search/Code) and Structured Outputs** in a single request.
    *   *Example:* "Search for stock prices, calculate a ratio using Python, and return a JSON object."
*   **Thinking Level:** Granular control (`"low"` for speed, `"high"` for deep reasoning) rather than a simple on/off budget.
*   **Media Resolution:** Control token usage (`"low"`, `"medium"`, `"high"`) for Images, PDFs, and Videos to balance cost vs. detail.

### 2. Standard Text & Multimodal (Gemini 2.5)
Use **`prompt_gemini`** for `gemini-2.5-flash` and `gemini-2.5-pro`.
*   **General Purpose:** Ideal for chat, standard text generation, and file analysis.
*   **Unified Tooling:** Use Google Search, Code Execution, and URL Context simultaneously.
*   **Simple Thinking:** Supports boolean (`True`/`False`) configuration for standard thinking models.

### 3. Strict Structured Data (Gemini 2.5)
Use **`prompt_gemini_structured`** for `gemini-2.5-flash` and `gemini-2.5-pro`.
*   **Strict Schemas:** Enforce output formats (JSON or Enums) using Pydantic models.
*   *Limitation:* Due to API constraints on older models, this function **cannot** use Tools (Search/Code) simultaneously.

### Shared Wrapper Capabilities
All functions benefit from these built-in automations:
*   **Multimodal Simplified:** Pass local file paths for **Videos** or **PDFs** directly into the function. The script automatically handles MIME types, file data, and context limits.
*   **Automatic Citations:** When Google Search is enabled, the wrapper parses grounding metadata to automatically insert inline markdown citations (e.g., `[1](url)`) and append a formatted source list.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Danielnara24/gemini-automation-kit.git
    cd gemini-automation-kit
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

You must set your Gemini API key as an environment variable before running the scripts.

**Mac/Linux:**
```bash
export GEMINI_API_KEY="your_actual_api_key_here"
```

**Windows (Command Prompt):**
```cmd
set GEMINI_API_KEY="your_actual_api_key_here"
```

**Windows (PowerShell):**
```powershell
$env:GEMINI_API_KEY="your_actual_api_key_here"
```

## API Limits & Pricing

This tool relies on the Google Gemini API. Usage is subject to the rate limits and pricing of the tier you are using.

*   **Rate Limits:** Please consult the official documentation for the most current rate limits (RPM/TPM/RPD):  
    [**Google Gemini API Rate Limits**](https://ai.google.dev/gemini-api/docs/rate-limits)

*   **Pricing:** For details on costs associated with the paid tier:  
    [**Google Gemini API Pricing**](https://ai.google.dev/gemini-api/docs/pricing)

## Usage

Import the utility functions into your own Python scripts:

### 1. Basic Text & Google Search
Use `prompt_gemini` for standard interactions with 2.5 models. Set `google_search=True` to enable live web grounding.

```python
from gemini_utils import prompt_gemini

prompt = "What are the latest specs of the Steam Deck OLED vs the ROG Ally X?"

response, tokens = prompt_gemini(
    model="gemini-2.5-flash",
    prompt=prompt,
    google_search=True,  # Enables Search Tool
    thinking=True        # Enables Thinking
)

print(response)
# Output will include inline citations [1] and a source list.
```

### 2. Multimodal: Video & PDF
You don't need to manually upload files via the API; just pass the local file path.

```python
video_path = "./downloads/tutorial.mp4"

response, tokens = prompt_gemini(
    model="gemini-2.5-flash",
    prompt="Summarize the steps shown in this video and extract the code used.",
    video_attachment=video_path,
    code_execution=True # Allows the model to run code
)
```

### 3. Gemini 3: Agentic Workflow (Search + Code + JSON)
The new `prompt_gemini_3` function allows you to combine tools (Search/Code) with Structured Outputs.

```python
from pydantic import BaseModel
from gemini_utils import prompt_gemini_3

# 1. Define the desired output structure
class CryptoRatio(BaseModel):
    btc_price: float
    eth_price: float
    ratio: float
    summary: str

# 2. Run the agent
# The model will: 
#   A. Search Google for current prices (Knowledge cutoff is Jan 2025)
#   B. Write/Run Python code to calculate the exact ratio
#   C. Return the result as a strict JSON object
response_obj, tokens = prompt_gemini_3(
    prompt="Find current BTC and ETH prices and calculate the ETH/BTC ratio.",
    response_schema=CryptoRatio, 
    google_search=True,
    code_execution=True,
    thinking_level="high"
)

print(f"Ratio: {response_obj.ratio} | Summary: {response_obj.summary}")
```

## Disclaimer

This is an unofficial open-source utility and is **not affiliated with, endorsed by, or connected to Google**.

The code is provided "as is" to help developers interact with the Gemini API more easily. Users are responsible for their own API usage, costs, and adherence to Google's Terms of Service.

## License

[MIT](https://choosealicense.com/licenses/mit/)