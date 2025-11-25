# Send post request for intiating quiz
import os
import requests
def intiate_quiz():
    payload = {
                "email": "23f1002487@ds.study.iitm.ac.in",
                "secret": "this-is-agni",
                "url": "https://tds-llm-analysis.s-anand.net/demo"
            }
    url = "http://localhost:8000/quiz/"
    response = requests.post(url, json=payload)
    return response.json()
if __name__ == "__main__":
    result = intiate_quiz()
    print(result)
    with open("log.log", "a") as f:
        f.write(str(result) + "\n")