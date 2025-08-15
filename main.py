import os
import json
from typing import Any, Dict, List, Optional

import streamlit as st

# If you're on openai>=1.0.0 (the modern SDK):
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# ---------------------------
# UI & App Configuration
# ---------------------------
st.set_page_config(page_title="Schema + Style Guide Extractor", layout="wide")
st.title("🧪 Schema & Style Guide Extractor")

st.caption(
    "Paste a Markdown/Text template and optionally include sample test cases. The app extracts: "
    "(1) a flat JSON Schema (all string fields), (2) concise guidelines, and (3) a detailed multi-bullet style guide derived from explicit instructions and sample tests."
)

# Model configuration (kept simple, no UI controls)
MODEL_NAME = "gpt-4.1"

# ---------------------------
# Helper: API client
# ---------------------------
def get_openai_client() -> Optional[Any]:
    """
    Creates the OpenAI client if the SDK is available and an API key is set.
    """
    return OpenAI()


# ---------------------------
# Prompts
# ---------------------------
SYSTEM_PROMPT = """
You are an expert technical writer and data modeler.

Task:
- Read the user's pasted template (plain text or Markdown) AND any included sample test cases.
- Output STRICT JSON ONLY (no code fences, no comments).
- The output MUST have three top-level keys: "schema", "guidelines", and "style_guide".
- Language & examples policy:
  - Use English for all generated/inferred narrative text ("guidelines" and the explanatory prose in "style_guide").
  - Preserve the original language for any text copied from the user's template (e.g., titles, rule snippets, step/expected texts, tags, domain terms). Do NOT translate copied content.
  - In "style_guide", include a final subsection titled "Examples" that lists ALL examples provided in the uploaded template verbatim, preserving their original language. Do not drop, merge, or summarize examples; include them one-to-one.
- "schema" MUST be a JSON Schema for an object with FLAT properties.
- All properties in the schema MUST have type "string". Do not use any other type. Do not nest objects or arrays.
- Each property MUST include a concise human-friendly description.
- "guidelines" MUST be a single English string summarizing overall test case generation guidelines derived from the template.
- "style_guide" MUST be a MARKDOWN STRING in English that captures the test case writing style. It MUST be detailed and derived from BOTH: (1) explicit instructions in the template, and (2) patterns observed in any sample test cases present. Focus on voice/tense, structure (e.g., Given/When/Then or Arrange-Act-Assert), naming conventions, assertion patterns, formatting (headings, bullets, numbering), step phrasing, and domain-specific terminology. Include a minimal pseudo-structure example as a small markdown section if applicable. End with a subsection titled "Examples" that enumerates all examples from the uploaded template verbatim (preserve original language). If nothing is available, return an empty string.

Return EXACTLY a JSON object conforming to this shape:
{
  "schema": {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "string",
    "type": "object",
    "properties": {
      "field_name": { "type": "string", "description": "human readable description" }
    },
    "required": ["optional", "list", "of", "fields"]
  },
  "guidelines": "concise overall guidelines text",
  "style_guide": "markdown describing the style in detail with headings and bullets as needed"
}

 Rules:
- Properties must be flat (no nested objects), and every property's type must be exactly "string".
- Keep descriptions and guidelines concise; make style_guide detailed but crisp.
- Avoid using triple-backtick code fences; standard markdown (headings, lists, bold/italics) is encouraged.
- If there are no obvious fields, return an empty properties object and an empty required list; guidelines and style_guide may be empty strings.
- Output VALID JSON ONLY.
""".strip()


def build_user_prompt(raw_text: str) -> str:
    return f"""
User Template (raw):
--------------------
{raw_text}
"""


# ---------------------------
# OpenAI Call
# ---------------------------
def extract_json_with_openai(
    client: Any,
    model: str,
    raw_text: str
) -> Dict[str, Any]:
    """
    Calls OpenAI chat completions with JSON-only response formatting.
    Falls back to plain parsing if response_format isn't supported by the model.
    """
    # Prefer models that support JSON mode. If the chosen model doesn’t, the SDK will error.
    response = None
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(raw_text)},
            ],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        # Fallback: try without response_format, then attempt to parse/repair.
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(raw_text)},
                ],
            )
            content = response.choices[0].message.content.strip()
            # Strip code fences if any
            if content.startswith("```"):
                content = content.split("```", 2)[1]
                # If there's a language hint like ```json
                if content.startswith("json"):
                    content = content[len("json"):].strip()
            return json.loads(content)
        except Exception as inner:
            raise RuntimeError(f"OpenAI parsing failed: {inner}") from inner


# ---------------------------
# Schema utils & rendering
# ---------------------------
def sanitize_to_string_schema(candidate: Dict[str, Any]) -> Dict[str, Any]:
    title = candidate.get("title") if isinstance(candidate, dict) else None
    if not isinstance(title, str) or not title.strip():
        title = "Extracted Schema"

    properties_in = {}
    if isinstance(candidate, dict):
        props = candidate.get("properties")
        if isinstance(props, dict):
            properties_in = props

    sanitized_properties: Dict[str, Dict[str, Any]] = {}
    for key, schema in properties_in.items():
        desc = None
        if isinstance(schema, dict):
            desc = schema.get("description")
        if desc is None:
            desc = ""
        sanitized_properties[str(key)] = {"type": "string", "description": str(desc)}

    required_in: List[str] = []
    if isinstance(candidate, dict):
        req = candidate.get("required")
        if isinstance(req, list):
            required_in = [str(x) for x in req if isinstance(x, str)]
    required_filtered = [k for k in required_in if k in sanitized_properties]

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": title,
        "type": "object",
        "properties": sanitized_properties,
        "required": required_filtered,
    }


def render_schema_table(schema: Dict[str, Any]) -> None:
    title = schema.get("title") or "Schema"
    st.subheader(f"📄 {title}")
    properties = schema.get("properties", {}) or {}
    if not isinstance(properties, dict) or not properties:
        st.info("No fields detected in schema.")
        return
    rows = []
    for field_name, field_schema in properties.items():
        if not isinstance(field_schema, dict):
            field_schema = {}
        description = field_schema.get("description") or "—"
        rows.append([field_name, "string", description])
    st.dataframe(
        {"Field": [r[0] for r in rows],
         "Type": [r[1] for r in rows],
         "Description": [r[2] for r in rows]},
        use_container_width=True,
        hide_index=True
    )


def sanitize_result_with_guidelines(raw_result: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(raw_result, dict):
        return {"schema": sanitize_to_string_schema({}), "guidelines": "", "style_guide": ""}
    candidate_schema: Any = raw_result.get("schema", raw_result)
    sanitized_schema = sanitize_to_string_schema(candidate_schema if isinstance(candidate_schema, dict) else {})
    guidelines_raw = raw_result.get("guidelines", "")
    guidelines_text = str(guidelines_raw) if guidelines_raw is not None else ""
    style_raw = raw_result.get("style_guide", "")
    style_markdown = str(style_raw) if style_raw is not None else ""
    return {"schema": sanitized_schema, "guidelines": guidelines_text, "style_guide": style_markdown}


# ---------------------------
# Main input area
# ---------------------------
pasted = st.text_area(
    "Paste template and optional sample test cases here",
    height=220,
    placeholder=(
        "# Example: Login test template\n"
        "Instructions: Use Given/When/Then. Present tense.\n\n"
        "# Sample tests (optional)\n"
        "Test: Successful login\n"
        "Given the user is on the login page...\n"
    ),
)

raw_text = ""
source = None
if pasted.strip():
    raw_text = pasted.strip()
    source = "pasted_text"

# ---------------------------
# Run extraction
# ---------------------------
run = st.button("🔎 Extract")

if run:
    if not raw_text:
        st.error("Please paste some template text.")
        st.stop()

    client = get_openai_client()
    if not client:
        st.stop()

    with st.spinner("Calling OpenAI and building JSON Schema + guidelines..."):
        try:
            result = extract_json_with_openai(
                client=client,
                model=MODEL_NAME,
                raw_text=raw_text
            )

            st.success("Extraction complete ✅")
            combined = sanitize_result_with_guidelines(result)
            schema = combined["schema"]
            guidelines = combined["guidelines"]
            style_guide = combined.get("style_guide", "")

            render_schema_table(schema)
            st.markdown("### 🧭 Guidelines")
            st.write(guidelines or "—")
            st.markdown("### 🖋️ Style guide")
            st.markdown(style_guide or "—")

            with st.expander("Show Full JSON (schema + guidelines + style)"):
                st.json(combined, expanded=False)

        except Exception as e:
            st.error(f"Failed to extract JSON: {e}")

