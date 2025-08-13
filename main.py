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
st.set_page_config(page_title="JSON Schema Extractor", layout="wide")
st.title("🧪 Schema Extractor")

st.caption(
    "Paste a Markdown or Text template. The app will extract a JSON Schema where all fields are strings, "
    "and display a table of fields with descriptions."
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
- Read the user's pasted template (plain text or Markdown).
- Output STRICT JSON ONLY (no code fences, no comments).
- The output MUST have two top-level keys: "schema" and "guidelines".
- "schema" MUST be a JSON Schema for an object with FLAT properties.
- All properties in the schema MUST have type "string". Do not use any other type. Do not nest objects or arrays.
- Each property MUST include a concise human-friendly description.
- "guidelines" MUST be a single string summarizing overall test case generation guidelines derived from the template.

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
  "guidelines": "concise overall guidelines text"
}

Rules:
- Properties must be flat (no nested objects), and every property's type must be exactly "string".
- Keep descriptions and the guidelines concise and useful.
- If there are no obvious fields, return an empty properties object and an empty required list; guidelines may be an empty string.
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
        return {"schema": sanitize_to_string_schema({}), "guidelines": ""}
    candidate_schema: Any = raw_result.get("schema", raw_result)
    sanitized_schema = sanitize_to_string_schema(candidate_schema if isinstance(candidate_schema, dict) else {})
    guidelines_raw = raw_result.get("guidelines", "")
    guidelines_text = str(guidelines_raw) if guidelines_raw is not None else ""
    return {"schema": sanitized_schema, "guidelines": guidelines_text}


# ---------------------------
# Main input area
# ---------------------------
pasted = st.text_area("Paste template content here", height=220, placeholder="# Example: Login test template\n...")

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

            render_schema_table(schema)
            st.markdown("### 🧭 Guidelines")
            st.write(guidelines or "—")

            with st.expander("Show Full JSON (schema + guidelines)"):
                st.json(combined, expanded=False)

        except Exception as e:
            st.error(f"Failed to extract JSON: {e}")

