import base64
import json
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

@app.route('/extract-data', methods=['POST'])
def extract_data():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']
    base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    prompt_text = """
    You are an intelligent assistant tasked with extracting structured information from a patient's treatment card image issued by SITA RATAN LEPROSY HOSPITAL, ANANDWAN.

    1. Use OCR to read printed text from the image.
    2. Extract the data into JSON format with the following keys:
    - book_number
    - registration_number
    - name
    - age
    - date_of_birth
    - sex
    - caste
    - mobile_number
    - aadhar_number
    - address
    - date_of_admission
    - mother's_name
    - relatives
    - blood_group
    - leprosy_type (either "MB" or "PB")
    - mdt_status (either "Cured", "Under MDT", or "Unknown")
    - deformity_status
    - duration_of_disease
    - previous_occupation

    3. If any field is empty or not visible in the image, assign its value as `null`.
    4. For multi-line fields like address or relatives, combine the lines into a single string.
    5. Return the final response in a clean JSON format only, with no extra commentary.

    Handle spelling inconsistencies (e.g., “pervious” should be treated as “previous”) and include them correctly in the JSON output.
    """

    completion = client.chat.completions.create(
        model="llama-3.2-90b-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        temperature=0.5,
        max_completion_tokens=1024,
        top_p=1,
        stream=False,
        response_format={"type": "json_object"},
    )

    structured_data = json.loads(completion.choices[0].message.content)

    # Optional: save to file
    with open("output.json", "w") as f:
        json.dump(structured_data, f, indent=4)

    return jsonify(structured_data)

if __name__ == '__main__':
    app.run(debug=True)
