import json
import ollama  


def create_user_prompt(jd_text: str) -> str:
    """
    Build the user prompt for the LLM, including a clear schema and a few-shot example.
    """
    return f"""
Analyze the following job description. Extract the following information and return only a valid JSON object:

"years_min": Minimum years of experience required (integer or null).
"years_max": Maximum years of experience required (integer or null).
"must_have_skills_raw": List of skills that are mandatory (e.g., "must have", "required").
"nice_to_have_skills_raw": List of skills that are preferred (e.g., "is a plus", "an advantage").
"must_have_certs_raw": List of mandatory certifications.
"nice_to_have_certs_raw": List of preferred certifications.
"role_focus_raw": A list of 2-3 keywords summarizing the core function of this role (e.g., "Financial Reporting", "Internal Control").

---
[EXAMPLE]
If the text is:
"We need a Senior Accountant with 5+ years experience. Must be proficient in MS Excel and SAP. HKICPA is required. A bonus if you know Tableau."

You must output:
{{
  "years_min": 5,
  "years_max": null,
  "must_have_skills_raw": ["MS Excel", "SAP"],
  "nice_to_have_skills_raw": ["Tableau"],
  "must_have_certs_raw": ["HKICPA"],
  "nice_to_have_certs_raw": [],
  "role_focus_raw": ["Senior Accountant"]
}}
---

[Job Description Text Here]
{jd_text}
"""


def extract_features_from_jd(jd_text: str, model_name: str = "qwen3:14b") -> str | None:
    """
    Call the Ollama server with a system+user prompt and request JSON output.
    Returns the raw JSON string emitted by the model, or None on failure.
    """
    user_prompt = create_user_prompt(jd_text)
    system_prompt = (
        "You are an expert AI assistant specialized in HR data extraction. "
        "You must follow the user's instructions precisely and only output a valid JSON object."
    )

    # Connect to Ollama server 
    ollama_server_url = "http://100.117.120.103:11434"

    try:
        print(f"Connecting to Ollama server at: {ollama_server_url} ...")
        client = ollama.Client(host=ollama_server_url)

        response = client.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            format="json",  # Ask Ollama to return JSON
        )

        llm_output_string = response["message"]["content"]
        return llm_output_string

    except Exception as e:
        print(f"Error calling Ollama API ({model_name} @ {ollama_server_url}): {e}")
        print("Please check:")
        print("  1) Tailscale is connected.")
        print("  2) The Ollama server host is running `ollama serve`.")
        print(f"  3) Model '{model_name}' has been pulled on the server (`ollama pull {model_name}`).")
        return None


def parse_llm_output(json_string: str | None) -> dict | None:
    """
    Parse the model's JSON string into a Python dict.
    Returns None if parsing fails or input is empty.
    """
    if not json_string:
        return None

    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON from LLM. Error: {e}")
        print(f"Raw LLM output (first 200 chars): {json_string[:200]}...")
        return None


if __name__ == "__main__":
    # Simple stress tests with varied JD styles
    test_jds = {
        "Test Case 1: Vague requirements (no explicit 'must have' / 'nice to have')": """
        We're looking for a Marketing Coordinator. You'll be responsible for social media, email campaigns, and event logistics. Strong writing skills are essential. Familiarity with HubSpot and Adobe Creative Suite is beneficial. A degree in Marketing or Communications is preferred.
        """,
        "Test Case 2: No experience requirement (Entry Level)": """
        Entry-level Junior Developer position. We will train the right candidate. Must be passionate about coding and eager to learn. Required: Basic understanding of JavaScript and HTML/CSS. Knowledge of React or Vue is a huge plus. We offer a great learning environment.
        """,
        "Test Case 3: Bulleted list / mixed formatting": """
        Job Title: Ops Manager
        What you'll do:
        - Oversee daily operations
        - Manage inventory
        - Ensure compliance
        Skills:
        *Leadership.
        *Excel (VLOOKUP, Pivot tables) - THIS IS REQUIRED.
        *SAP experience is an advantage.
        *We need someone with 5+ years in logistics.
        *PMP cert nice to have.
        """,
        "Test Case 4: Spelled-out years ('three years')": """
        Seeking a financial analyst. The ideal candidate will have at least three years of experience in financial modeling. Must be proficient in Excel. CFA (Chartered Financial Analyst) designation is highly desirable but not mandatory. Role involves forecasting and budgeting.
        """,
    }

    OLLAMA_MODEL_TO_USE = "qwen3:14b"

    for test_name, test_jd in test_jds.items():
        print("\n" + "=" * 50)
        print(f"Running: {test_name}")
        print(f"Using Ollama model: {OLLAMA_MODEL_TO_USE}")

        json_output = extract_features_from_jd(test_jd, model_name=OLLAMA_MODEL_TO_USE)

        if json_output:
            parsed = parse_llm_output(json_output)
            if parsed is not None:
                print("Parsed successfully:")
                print(json.dumps(parsed, indent=2, ensure_ascii=False))
            else:
                print("JSON parsing failed.")
                print(f"Raw output (first 200 chars): {json_output[:200]}...")
        else:
            print("API call failed (server may be unreachable).")

        print("=" * 50 + "\n")

    print("All stress tests completed.")
