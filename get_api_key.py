# ----------------------------------------------------------------
# CP423 Project - Get API Key
# ----------------------------------------------------------------
# Name:     Jordan Asmono
# ID:       210922810
# Email:    asmo2810@mylaurier.ca
# Date:     2026-08-05
# ----------------------------------------------------------------
# Imports
# ----------------------------------------------------------------
import requests
# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------
url = "https://openapi.data.uwaterloo.ca/v3/account/register"

data = {
    "email": "jordan.asmono@gmail.com",
    "project": "CP423: RAG System Final Project",
    "uri": "https://github.com/GamerSwagnamite/CP423-Project"
}

response = requests.post(url, data=data)

if response.status_code == 200:
    print("Success! Here is your API key:")
    print(response.json())  # The key will be inside this JSON response
else:
    print("Error:", response.status_code)
    print(response.text)