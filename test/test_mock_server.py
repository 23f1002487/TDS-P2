#!/usr/bin/env python3
"""
Test script to verify the mock quiz server works with your app
Run this after starting the mock server
"""
import requests
import json
import time

MOCK_SERVER_URL = "http://localhost:8000"

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def test_mock_server():
    """Test that the mock server is running"""
    print_section("Testing Mock Quiz Server")
    
    try:
        # Test health endpoint
        print("1. Testing /health endpoint...")
        response = requests.get(f"{MOCK_SERVER_URL}/health", timeout=5)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
        
        # Test API questions endpoint
        print("\n2. Testing /api/questions endpoint...")
        response = requests.get(f"{MOCK_SERVER_URL}/api/questions", timeout=5)
        data = response.json()
        print(f"   Status: {response.status_code}")
        print(f"   Total questions: {data['total_questions']}")
        print(f"   First question: {data['questions'][0]['question']}")
        
        # Test demo quiz (HTML)
        print("\n3. Testing /demo endpoint...")
        response = requests.get(f"{MOCK_SERVER_URL}/demo", timeout=5)
        print(f"   Status: {response.status_code}")
        print(f"   Content type: {response.headers.get('content-type')}")
        print(f"   HTML length: {len(response.text)} characters")
        
        print("\n✅ Mock server is working correctly!")
        print("\n📝 You can now test your app with these URLs:")
        print(f"   - Demo Quiz: {MOCK_SERVER_URL}/demo")
        print(f"   - Full Quiz: {MOCK_SERVER_URL}/full-quiz")
        print(f"   - API Docs: {MOCK_SERVER_URL}/docs")
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Cannot connect to mock server!")
        print(f"\nMake sure the mock server is running:")
        print("   python3 mock_quiz_server.py")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def test_quiz_questions():
    """Display sample quiz questions"""
    print_section("Sample Quiz Questions from Mock Server")
    
    try:
        response = requests.get(f"{MOCK_SERVER_URL}/api/questions", timeout=5)
        data = response.json()
        
        print(f"Total questions available: {data['total_questions']}\n")
        
        # Show first 3 questions
        for i, q in enumerate(data['questions'][:3], 1):
            print(f"Question {i}: {q['question']}")
            print(f"Topic: {q['topic']}")
            for idx, opt in enumerate(q['options']):
                marker = "✓" if idx == q['correct_answer'] else " "
                print(f"  {marker} {idx}. {opt}")
            print()
            
    except Exception as e:
        print(f"Error: {e}")


def test_with_your_app():
    """Show how to test with your actual quiz solver app"""
    print_section("Testing with Your Quiz Solver App")
    
    print("To test your app with the mock server:\n")
    
    print("1. Start the mock quiz server (in terminal 1):")
    print("   python3 mock_quiz_server.py\n")
    
    print("2. Update test_live_app.py to use the mock server:")
    print('   QUIZ_URL = "http://localhost:8000/demo"\n')
    
    print("3. Run your app tests (in terminal 2):")
    print("   python3 test_live_app.py\n")
    
    print("Or test your main app directly:")
    print("   python3 app.py\n")
    
    print("Then send a request to your app with the mock quiz URL.")


if __name__ == "__main__":
    print("\n🎨 Mock Quiz Server Tester\n")
    
    # Test if server is running
    if test_mock_server():
        # Show sample questions
        test_quiz_questions()
    
    # Show instructions
    test_with_your_app()
    
    print("\n" + "="*70)
    print("  Test Complete!")
    print("="*70 + "\n")
