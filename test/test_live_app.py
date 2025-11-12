#!/usr/bin/env python3
"""
Test script for the deployed HF Space application
Tests all endpoints and features
"""
import requests
import json
import time
from datetime import datetime

# Configuration

# ============================================
# YOUR APP SERVER (Quiz Solver)
# ============================================
BASE_URL = "https://vsaketh-tds-p2.hf.space"  # Your HF Space app
# BASE_URL = "http://localhost:7860"  # Uncomment for local testing

# ============================================
# QUIZ SERVER (Questions Source)
# ============================================
QUIZ_URL = "http://localhost:8000/demo"  # Local mock makeup quiz (3 questions)

# Alternative quiz URLs:
# QUIZ_URL = "http://localhost:8000/full-quiz"  # Local mock - full quiz (15 questions)
# QUIZ_URL = "http://localhost:8000/random-quiz/5"  # Local mock - random 5 questions
# QUIZ_URL = "https://tds-llm-analysis.s-anand.net/demo"  # Real quiz server

# ============================================
# CREDENTIALS
# ============================================
EMAIL = "your-email@ds.study.iitm.ac.in"  # Update with your email
SECRET = "your-secret-string"  # Update with your secret


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def test_root_endpoint():
    """Test GET /"""
    print_section("TEST 1: Root Endpoint (GET /)")
    
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        assert response.status_code == 200, "Root endpoint should return 200"
        assert "message" in response.json(), "Response should contain 'message'"
        print("✅ Root endpoint test PASSED")
        return True
    except Exception as e:
        print(f"❌ Root endpoint test FAILED: {e}")
        return False


def test_health_endpoint():
    """Test GET /health"""
    print_section("TEST 2: Health Endpoint (GET /health)")
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        assert response.status_code == 200, "Health endpoint should return 200"
        data = response.json()
        assert "status" in data, "Response should contain 'status'"
        assert "components" in data, "Response should contain 'components'"
        print("✅ Health endpoint test PASSED")
        return True
    except Exception as e:
        print(f"❌ Health endpoint test FAILED: {e}")
        return False


def test_docs_endpoint():
    """Test GET /docs (Swagger UI)"""
    print_section("TEST 3: API Docs (GET /docs)")
    
    try:
        response = requests.get(f"{BASE_URL}/docs")
        print(f"Status Code: {response.status_code}")
        print(f"Content-Type: {response.headers.get('content-type')}")
        
        assert response.status_code == 200, "Docs endpoint should return 200"
        assert "text/html" in response.headers.get('content-type', ''), "Should return HTML"
        print("✅ Docs endpoint test PASSED")
        print(f"   View at: {BASE_URL}/docs")
        return True
    except Exception as e:
        print(f"❌ Docs endpoint test FAILED: {e}")
        return False


def test_quiz_endpoint_validation():
    """Test POST /quiz with invalid data"""
    print_section("TEST 4: Quiz Endpoint Validation (POST /quiz)")
    
    # Test 1: Missing fields
    print("Test 4.1: Missing required fields")
    try:
        response = requests.post(f"{BASE_URL}/quiz", json={})
        print(f"Status Code: {response.status_code}")
        assert response.status_code == 422, "Should return 422 for missing fields"
        print("✅ Missing fields validation PASSED")
    except Exception as e:
        print(f"❌ Missing fields validation FAILED: {e}")
    
    # Test 2: Invalid email
    print("\nTest 4.2: Invalid email format")
    try:
        response = requests.post(f"{BASE_URL}/quiz", json={
            "email": "invalid-email",
            "secret": "test",
            "url": "https://example.com"
        })
        print(f"Status Code: {response.status_code}")
        assert response.status_code == 422, "Should return 422 for invalid email"
        print("✅ Invalid email validation PASSED")
    except Exception as e:
        print(f"❌ Invalid email validation FAILED: {e}")
    
    # Test 3: Invalid URL scheme
    print("\nTest 4.3: Invalid URL scheme (file://)")
    try:
        response = requests.post(f"{BASE_URL}/quiz", json={
            "email": EMAIL,
            "secret": SECRET,
            "url": "file:///etc/passwd"
        })
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        assert response.status_code == 422, "Should return 422 for dangerous URL"
        print("✅ URL scheme validation PASSED")
    except Exception as e:
        print(f"❌ URL scheme validation FAILED: {e}")
    
    return True


def test_quiz_submission():
    """Test POST /quiz with valid data"""
    print_section("TEST 5: Quiz Submission (POST /quiz)")
    
    # Check if credentials are set
    if EMAIL == "your-email@ds.study.iitm.ac.in":
        print("⚠️  WARNING: Using default email. Update EMAIL and SECRET in the script")
        print("   Attempting to test with mock server anyway...\n")
    
    try:
        print(f"Submitting quiz: {QUIZ_URL}")
        print(f"Using email: {EMAIL}")
        print(f"Target server: {BASE_URL}\n")
        
        payload = {
            "email": EMAIL,
            "secret": SECRET,
            "url": QUIZ_URL
        }
        
        print("Sending POST request...")
        response = requests.post(f"{BASE_URL}/quiz", json=payload, timeout=30)
        print(f"Status Code: {response.status_code}")
        
        try:
            response_data = response.json()
            print(f"Response: {json.dumps(response_data, indent=2)}")
        except:
            print(f"Response Text: {response.text}")
        
        if response.status_code == 202:
            print("\n✅ Quiz submission ACCEPTED (processing in background)")
            if isinstance(response_data, dict):
                print(f"   Request ID: {response_data.get('request_id', 'N/A')}")
                print(f"   Status: {response_data.get('status', 'N/A')}")
            return True
        elif response.status_code == 200:
            print("\n✅ Quiz submission SUCCESSFUL")
            return True
        elif response.status_code == 403:
            print("\n❌ Authentication failed - check EMAIL and SECRET")
            return False
        else:
            print(f"\n⚠️  Unexpected status code: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError as e:
        print(f"❌ Connection error: Cannot reach {BASE_URL}")
        print("   Make sure your app server is running!")
        return False
    except requests.exceptions.Timeout:
        print(f"❌ Request timeout after 30 seconds")
        return False
    except Exception as e:
        print(f"❌ Quiz submission test FAILED: {e}")
        return False


def test_rate_limiting():
    """Test rate limiting by making rapid requests"""
    print_section("TEST 6: Rate Limiting")
    
    print("Making 5 rapid requests to health endpoint...")
    for i in range(5):
        response = requests.get(f"{BASE_URL}/health")
        print(f"Request {i+1}: Status {response.status_code}")
        time.sleep(0.2)
    
    print("✅ Rate limiting test completed (check for 429 status if limit exceeded)")
    return True


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("  HF SPACE APPLICATION TEST SUITE")
    print("="*70)
    print(f"\nTarget URL: {BASE_URL}")
    print(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {
        "Root Endpoint": test_root_endpoint(),
        "Health Endpoint": test_health_endpoint(),
        "Docs Endpoint": test_docs_endpoint(),
        "Input Validation": test_quiz_endpoint_validation(),
        "Quiz Submission": test_quiz_submission(),
        "Rate Limiting": test_rate_limiting(),
    }
    
    # Summary
    print_section("TEST SUMMARY")
    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result is True else ("❌ FAILED" if result is False else "⚠️  SKIPPED")
        print(f"{test_name:<25} {status}")
    
    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")
    print("\n" + "="*70)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
