#!/usr/bin/env python3
"""
Test AIPipe connectivity and API functionality
"""
import asyncio
import os
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN", "")
AIPIPE_BASE_URL = os.getenv("AIPIPE_BASE_URL", "https://aipipe.org/openai/v1")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


async def test_connection():
    """Test basic connection to AIPipe"""
    print_section("TEST 1: Basic Connection")
    
    if not AIPIPE_TOKEN or AIPIPE_TOKEN.startswith("your-"):
        print("❌ AIPIPE_TOKEN not set or invalid")
        print("   Please set AIPIPE_TOKEN in your .env file")
        return False
    
    print(f"Token: {AIPIPE_TOKEN[:20]}... ({len(AIPIPE_TOKEN)} chars)")
    print(f"Base URL: {AIPIPE_BASE_URL}")
    print(f"Model: {MODEL}")
    
    return True


async def test_simple_completion():
    """Test a simple chat completion"""
    print_section("TEST 2: Simple Chat Completion")
    
    try:
        client = AsyncOpenAI(
            api_key=AIPIPE_TOKEN,
            base_url=AIPIPE_BASE_URL
        )
        
        print("Sending request: 'What is 2+2?'")
        
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2+2? Answer with just the number."}
            ],
            temperature=0.1,
            max_tokens=10
        )
        
        answer = response.choices[0].message.content
        print(f"\n✅ Response received: {answer}")
        print(f"   Model used: {response.model}")
        print(f"   Tokens used: {response.usage.total_tokens}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print(f"   Type: {type(e).__name__}")
        
        # Print more details
        if hasattr(e, '__cause__'):
            print(f"   Cause: {e.__cause__}")
        if hasattr(e, 'response'):
            print(f"   Response: {e.response}")
        
        # Try a simple HTTP test
        print("\n   Testing basic HTTP connectivity...")
        try:
            import httpx
            response = httpx.get("https://aipipe.org")
            print(f"   AIPipe website reachable: {response.status_code}")
        except Exception as http_err:
            print(f"   Cannot reach aipipe.org: {http_err}")
        
        return False


async def test_json_response():
    """Test getting JSON response (like quiz solver uses)"""
    print_section("TEST 3: JSON Response Test")
    
    try:
        client = AsyncOpenAI(
            api_key=AIPIPE_TOKEN,
            base_url=AIPIPE_BASE_URL
        )
        
        prompt = """Given this data: [1, 2, 3, 4, 5]
        
Provide analysis in JSON format:
{
    "sum": "sum of numbers",
    "average": "average of numbers"
}

Respond with ONLY valid JSON."""
        
        print("Sending request for JSON response...")
        
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a data analyst. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=200
        )
        
        result = response.choices[0].message.content
        print(f"\n✅ Response received:")
        print(result)
        
        # Try to parse as JSON
        import json
        import re
        
        # Clean markdown code blocks if present
        cleaned = re.sub(r'```json\n?', '', result)
        cleaned = re.sub(r'```\n?', '', cleaned)
        cleaned = cleaned.strip()
        
        parsed = json.loads(cleaned)
        print(f"\n✅ Valid JSON parsed successfully:")
        print(f"   Sum: {parsed.get('sum')}")
        print(f"   Average: {parsed.get('average')}")
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"\n⚠️  JSON parse error: {e}")
        print(f"   Response was: {result}")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


async def test_multiple_models():
    """Test different model configurations"""
    print_section("TEST 4: Multiple Models Test")
    
    models_to_test = [
        "gpt-4o-mini",
        "gpt-3.5-turbo",
        "gpt-4o",
        "gpt-4",
        "openai/gpt-4o-mini",
        "openai/gpt-3.5-turbo",
        "openai/gpt-4o",
    ]
    
    results = {}
    
    for model in models_to_test:
        print(f"\nTesting model: {model}")
        try:
            client = AsyncOpenAI(
                api_key=AIPIPE_TOKEN,
                base_url=AIPIPE_BASE_URL
            )
            
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": "Say 'Hello'"}
                ],
                temperature=0.1,
                max_tokens=10
            )
            
            answer = response.choices[0].message.content
            print(f"  ✅ Success: {answer}")
            results[model] = "✅ Available"
            
        except Exception as e:
            error_msg = str(e)[:100]
            print(f"  ❌ Failed: {error_msg}")
            results[model] = f"❌ {error_msg}"
    
    print("\n" + "-"*70)
    print("Model Availability Summary:")
    for model, status in results.items():
        print(f"  {model:<30} {status}")
    
    return True


async def test_streaming():
    """Test streaming responses (optional)"""
    print_section("TEST 5: Streaming Test (Optional)")
    
    try:
        client = AsyncOpenAI(
            api_key=AIPIPE_TOKEN,
            base_url=AIPIPE_BASE_URL
        )
        
        print("Testing streaming response...")
        print("Response: ", end="", flush=True)
        
        stream = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "user", "content": "Count from 1 to 5"}
            ],
            temperature=0.1,
            max_tokens=50,
            stream=True
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
        
        print("\n\n✅ Streaming works!")
        return True
        
    except Exception as e:
        print(f"\n⚠️  Streaming not supported or error: {e}")
        return False


async def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("  AIPIPE CONNECTIVITY TEST SUITE")
    print("="*70)
    
    results = {}
    
    # Test 1: Connection check
    if not await test_connection():
        print("\n❌ Cannot proceed without valid AIPIPE_TOKEN")
        return
    
    # Test 2: Simple completion
    results["Simple Completion"] = await test_simple_completion()
    
    if not results["Simple Completion"]:
        print("\n❌ Basic API call failed. Check your token and base URL.")
        return
    
    # Test 3: JSON response
    results["JSON Response"] = await test_json_response()
    
    # Test 4: Multiple models
    results["Multiple Models"] = await test_multiple_models()
    
    # Test 5: Streaming (optional)
    results["Streaming"] = await test_streaming()
    
    # Summary
    print_section("TEST SUMMARY")
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<25} {status}")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! AIPipe is working correctly.")
    elif passed >= 2:
        print("\n⚠️  Core functionality working, some features may be limited.")
    else:
        print("\n❌ AIPipe connectivity issues detected.")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    asyncio.run(main())
