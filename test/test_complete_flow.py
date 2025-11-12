#!/usr/bin/env python3
"""
Complete Flow Test - Tests the entire quiz solving pipeline
1. Mock quiz server provides questions
2. Your app solves the quiz
3. Mock quiz server validates answers
"""
import requests
import json
import time
from datetime import datetime

# Configuration
MOCK_QUIZ_SERVER = "http://localhost:8000"
YOUR_APP_SERVER = "http://localhost:7860"  # or your HF Space URL

# Test credentials
EMAIL = "test@example.com"
SECRET = "test-secret"


def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def test_mock_server_health():
    """Test if mock quiz server is running"""
    print_header("STEP 1: Check Mock Quiz Server")
    
    try:
        response = requests.get(f"{MOCK_QUIZ_SERVER}/health", timeout=5)
        data = response.json()
        print(f"✅ Mock server is healthy")
        print(f"   Total questions: {data['total_questions']}")
        print(f"   Total submissions: {data['total_submissions']}")
        print(f"   Endpoints: {', '.join(data['endpoints'])}")
        return True
    except Exception as e:
        print(f"❌ Mock server is not running: {e}")
        print(f"\nStart it with: python3 mock_quiz_server.py")
        return False


def test_get_quiz_questions():
    """Get quiz questions from mock server"""
    print_header("STEP 2: Get Quiz Questions")
    
    try:
        # Get demo quiz (HTML)
        response = requests.get(f"{MOCK_QUIZ_SERVER}/demo", timeout=5)
        print(f"✅ Demo quiz endpoint: {response.status_code}")
        print(f"   HTML length: {len(response.text)} characters")
        
        # Get questions as JSON
        response = requests.get(f"{MOCK_QUIZ_SERVER}/api/questions", timeout=5)
        data = response.json()
        print(f"✅ API questions endpoint: {response.status_code}")
        print(f"   Available questions: {data['total_questions']}")
        
        # Show first question
        if data['questions']:
            q = data['questions'][0]
            print(f"\n   Example Question:")
            print(f"   Q: {q['question']}")
            print(f"   Options: {len(q['options'])}")
            print(f"   Correct: {q['correct_answer']}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to get questions: {e}")
        return False


def test_submit_to_mock_server():
    """Submit answers directly to mock server"""
    print_header("STEP 3: Submit Answers to Mock Server")
    
    try:
        # Prepare sample answers (these are correct answers)
        submission = {
            "email": EMAIL,
            "quiz_url": f"{MOCK_QUIZ_SERVER}/demo",
            "answers": [
                {"question_id": 1, "answer": 1},  # Correct answer for Q1
                {"question_id": 2, "answer": 2},  # Correct answer for Q2
                {"question_id": 3, "answer": 1}   # Correct answer for Q3
            ]
        }
        
        print(f"Submitting {len(submission['answers'])} answers...")
        response = requests.post(
            f"{MOCK_QUIZ_SERVER}/submit",
            json=submission,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Submission successful!")
            print(f"\n   Results:")
            print(f"   - Total questions: {data['results']['total_questions']}")
            print(f"   - Correct answers: {data['results']['correct_answers']}")
            print(f"   - Score: {data['results']['score_percentage']}%")
            print(f"   - Grade: {data['results']['grade']}")
            
            print(f"\n   Detailed Results:")
            for result in data['detailed_results']:
                status = "✓" if result['is_correct'] else "✗"
                print(f"   {status} Q{result['question_id']}: {result['topic']}")
            
            return True
        else:
            print(f"❌ Submission failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to submit: {e}")
        return False


def test_get_submissions():
    """Get submission history"""
    print_header("STEP 4: Check Submission History")
    
    try:
        response = requests.get(f"{MOCK_QUIZ_SERVER}/submissions", timeout=5)
        data = response.json()
        
        print(f"✅ Total submissions: {data['total_submissions']}")
        
        if data['submissions']:
            latest = data['submissions'][-1]
            print(f"\n   Latest submission:")
            print(f"   - Email: {latest['email']}")
            print(f"   - Score: {latest['score_percentage']}%")
            print(f"   - Questions: {latest['total_questions']}")
            print(f"   - Correct: {latest['correct_answers']}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to get submissions: {e}")
        return False


def test_complete_integration():
    """Test complete integration with your app"""
    print_header("STEP 5: Test Integration with Your App")
    
    print("This would test the complete flow:")
    print("1. Your app receives POST /quiz request")
    print("2. Your app fetches questions from mock server")
    print("3. Your app uses LLM to solve questions")
    print("4. Your app submits answers to mock server")
    print("5. Mock server validates and scores")
    
    print("\n⚠️  To enable this test:")
    print("   1. Make sure your app is running on", YOUR_APP_SERVER)
    print("   2. Update credentials in this script")
    print("   3. Your app should POST results to mock server's /submit endpoint")
    
    # Uncomment to test with your actual app
    # try:
    #     payload = {
    #         "email": EMAIL,
    #         "secret": SECRET,
    #         "url": f"{MOCK_QUIZ_SERVER}/demo"
    #     }
    #     response = requests.post(f"{YOUR_APP_SERVER}/quiz", json=payload, timeout=30)
    #     print(f"Response: {response.status_code}")
    # except Exception as e:
    #     print(f"Error: {e}")


def main():
    """Run all tests"""
    print("\n" + "🎨 "*20)
    print("  COMPLETE FLOW TEST - Quiz Solving Pipeline")
    print("🎨 "*20)
    
    # Run tests
    tests = [
        test_mock_server_health,
        test_get_quiz_questions,
        test_submit_to_mock_server,
        test_get_submissions,
        test_complete_integration
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
        
        time.sleep(0.5)
    
    # Summary
    print_header("TEST SUMMARY")
    passed = sum(1 for r in results if r is True)
    total = len([r for r in results if r is not None])
    
    print(f"Tests Passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All tests passed!")
    else:
        print("⚠️  Some tests failed or skipped")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
