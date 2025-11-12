"""
Test API Endpoints (FastAPI)
"""
import requests
import json


def test_root_endpoint():
    """Test root endpoint"""
    print("\n=== Testing Root Endpoint ===")
    response = requests.get("http://localhost:7860/")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 200


def test_health_endpoint():
    """Test health check endpoint"""
    print("\n=== Testing Health Endpoint ===")
    response = requests.get("http://localhost:7860/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 200


def test_quiz_endpoint_valid():
    """Test quiz endpoint with valid data"""
    print("\n=== Testing Quiz Endpoint (Valid) ===")
    payload = {
        "email": "23f1002487@ds.study.iitm.ac.in",
        "secret": "this-is-agni",
        "url": "https://tds-llm-analysis.s-anand.net/demo"
    }
    response = requests.post(
        "http://localhost:7860/quiz",
        json=payload,
        headers={'Content-Type': 'application/json'}
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 200


def test_quiz_endpoint_invalid_json():
    """Test quiz endpoint with invalid JSON"""
    print("\n=== Testing Quiz Endpoint (Invalid JSON) ===")
    response = requests.post(
        "http://localhost:7860/quiz",
        data="not json",
        headers={'Content-Type': 'text/plain'}
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
    # FastAPI returns 422 for invalid JSON/validation errors
    assert response.status_code == 422


def test_quiz_endpoint_invalid_secret():
    """Test quiz endpoint with invalid secret"""
    print("\n=== Testing Quiz Endpoint (Invalid Secret) ===")
    payload = {
        "email": "23f1002487@ds.study.iitm.ac.in",
        "secret": "wrong-secret",
        "url": "https://tds-llm-analysis.s-anand.net/demo"
    }
    response = requests.post(
        "http://localhost:7860/quiz",
        json=payload,
        headers={'Content-Type': 'application/json'}
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 403


def test_quiz_endpoint_missing_fields():
    """Test quiz endpoint with missing fields"""
    print("\n=== Testing Quiz Endpoint (Missing Fields) ===")
    payload = {
        "email": "23f1002487@ds.study.iitm.ac.in"
        # Missing secret and url
    }
    response = requests.post(
        "http://localhost:7860/quiz",
        json=payload,
        headers={'Content-Type': 'application/json'}
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    # FastAPI returns 422 for validation errors
    assert response.status_code == 422


def test_quiz_endpoint_invalid_email():
    """Test quiz endpoint with invalid email format"""
    print("\n=== Testing Quiz Endpoint (Invalid Email Format) ===")
    payload = {
        "email": "not-an-email",
        "secret": "this-is-agni",
        "url": "https://tds-llm-analysis.s-anand.net/demo"
    }
    response = requests.post(
        "http://localhost:7860/quiz",
        json=payload,
        headers={'Content-Type': 'application/json'}
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    # FastAPI validates email format
    assert response.status_code == 422


def test_api_docs():
    """Test that API docs are available"""
    print("\n=== Testing API Documentation ===")
    response = requests.get("http://localhost:7860/docs")
    print(f"Status: {response.status_code}")
    assert response.status_code == 200
    print("✓ Swagger UI available at /docs")


if __name__ == "__main__":
    print("Starting FastAPI Tests...")
    print("Make sure the server is running: uvicorn app:app --port 7860")
    
    try:
        test_root_endpoint()
        test_health_endpoint()
        test_quiz_endpoint_valid()
        test_quiz_endpoint_invalid_json()
        test_quiz_endpoint_invalid_secret()
        test_quiz_endpoint_missing_fields()
        test_quiz_endpoint_invalid_email()
        test_api_docs()
        
        print("\n" + "="*50)
        print("All tests passed! ✓")
        print("="*50)
        print("\n💡 Try the interactive API docs at http://localhost:7860/docs")
    except Exception as e:
        print(f"\nTest failed: {str(e)}")
        raise
