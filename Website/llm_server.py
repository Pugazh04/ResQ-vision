from flask import Flask, jsonify
import google.generativeai as genai
import json
import os
import time
from threading import Lock

app = Flask(__name__)

# ================= MANUAL CORS FIX =================
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET')
    return response

# ================= CONFIG =================
genai.configure(api_key=os.getenv("<your api key>"))
model = genai.GenerativeModel("<your gemini model>")

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RESULT_JSON = os.path.join(BASE_DIR, "severity_output", "mask_img.json")

# ================= GLOBAL STATE =================
_llm_result = None
_llm_lock = Lock()
_llm_last_error = None

# ================= LLM CALL =================
def generate_insights():
    global _llm_result, _llm_last_error

    try:
        with open(RESULT_JSON, "r") as f:
            metrics = json.load(f)

        prompt = f"""
You are a disaster analysis expert.

Generate dashboard-ready insights.

Use EXACT headings:
Key Observations
Severity Interpretation

Use bullet points only.
Do not mention AI.

Severity Level: {metrics['severity_level']}
Spread Percentage: {metrics['spread_percentage']}%
"""

        response = model.generate_content(prompt)
        _llm_result = {"text": response.text}
        _llm_last_error = None
        print("[LLM] Successfully generated insights")

    except Exception as e:
        _llm_last_error = str(e)
        print(f"[LLM] Error: {e}")

# ================= ROUTE =================
@app.route("/llm-insights")
def llm_insights():
    global _llm_result

    with _llm_lock:
        # First call only
        if _llm_result is None and _llm_last_error is None:
            generate_insights()

        # If Gemini temporarily blocked
        if _llm_result is None and _llm_last_error:
            return jsonify({
                "text": (
                    "Key Observations\n"
                    "- Insights will be available shortly\n\n"
                    "Severity Interpretation\n"
                    "- LLM is temporarily rate-limited, please refresh later"
                )
            })

        return jsonify(_llm_result)

# ================= START =================
if __name__ == "__main__":
    print("[LLM] Server running at <your llm server url>")
    app.run(port=<your llm server port>, debug=False)
