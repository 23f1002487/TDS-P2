#!/usr/bin/env python3
"""
Mock Quiz Server - FastAPI server that simulates a quiz platform
Provides makeup quiz questions for testing your quiz solver app
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
import random
from typing import List, Dict, Optional
from datetime import datetime

app = FastAPI(title="Mock Quiz Server", description="Simulated quiz platform for testing")

# Add CORS middleware to allow requests from your app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sample makeup quiz questions
MAKEUP_QUIZ_QUESTIONS = [
    {
        "id": 1,
        "question": "What is the primary purpose of primer in makeup application?",
        "options": [
            "To add color to the skin",
            "To create a smooth base and help makeup last longer",
            "To remove makeup at the end of the day",
            "To moisturize dry skin"
        ],
        "correct_answer": 1,
        "topic": "Makeup Basics"
    },
    {
        "id": 2,
        "question": "Which type of foundation provides the most coverage?",
        "options": [
            "BB cream",
            "Tinted moisturizer",
            "Full-coverage liquid foundation",
            "Setting powder"
        ],
        "correct_answer": 2,
        "topic": "Foundation"
    },
    {
        "id": 3,
        "question": "What is contouring primarily used for in makeup?",
        "options": [
            "To add shimmer and glow",
            "To define and enhance facial structure",
            "To cover blemishes",
            "To set the foundation"
        ],
        "correct_answer": 1,
        "topic": "Contouring"
    },
    {
        "id": 4,
        "question": "What does SPF stand for in makeup and skincare products?",
        "options": [
            "Skin Protection Formula",
            "Sun Protection Factor",
            "Special Pigment Foundation",
            "Smooth Polish Finish"
        ],
        "correct_answer": 1,
        "topic": "Skincare & Makeup"
    },
    {
        "id": 5,
        "question": "Which tool is best for applying liquid foundation for a flawless finish?",
        "options": [
            "Cotton pad",
            "Powder brush",
            "Beauty sponge or foundation brush",
            "Fingers only"
        ],
        "correct_answer": 2,
        "topic": "Makeup Tools"
    },
    {
        "id": 6,
        "question": "What is the purpose of setting spray in makeup?",
        "options": [
            "To remove makeup",
            "To help makeup last longer and prevent smudging",
            "To add extra coverage",
            "To cleanse the skin"
        ],
        "correct_answer": 1,
        "topic": "Makeup Setting"
    },
    {
        "id": 7,
        "question": "What color corrector neutralizes dark circles under the eyes?",
        "options": [
            "Green",
            "Purple",
            "Orange or peach",
            "Blue"
        ],
        "correct_answer": 2,
        "topic": "Color Correction"
    },
    {
        "id": 8,
        "question": "Which eyebrow product provides the most natural look?",
        "options": [
            "Eyebrow pomade",
            "Eyebrow powder",
            "Liquid eyebrow pen",
            "Eyebrow gel"
        ],
        "correct_answer": 1,
        "topic": "Eyebrow Makeup"
    },
    {
        "id": 9,
        "question": "What is baking in makeup terminology?",
        "options": [
            "Heating makeup to make it melt",
            "Applying loose powder and letting it sit before brushing off",
            "Using a heat tool to set foundation",
            "Mixing multiple makeup products together"
        ],
        "correct_answer": 1,
        "topic": "Advanced Techniques"
    },
    {
        "id": 10,
        "question": "Which type of eyeliner is best for beginners?",
        "options": [
            "Liquid eyeliner",
            "Gel eyeliner",
            "Pencil eyeliner",
            "Cake eyeliner"
        ],
        "correct_answer": 2,
        "topic": "Eye Makeup"
    },
    {
        "id": 11,
        "question": "What is the correct order for applying eye makeup?",
        "options": [
            "Mascara, eyeshadow, eyeliner",
            "Eyeshadow, eyeliner, mascara",
            "Eyeliner, mascara, eyeshadow",
            "Mascara, eyeliner, eyeshadow"
        ],
        "correct_answer": 1,
        "topic": "Eye Makeup Application"
    },
    {
        "id": 12,
        "question": "What does 'cut crease' refer to in eye makeup?",
        "options": [
            "A technique that creates a sharp line between lid and crease",
            "A type of eyeliner application",
            "Removing excess eyeshadow",
            "A mascara application method"
        ],
        "correct_answer": 0,
        "topic": "Eye Makeup Techniques"
    },
    {
        "id": 13,
        "question": "What is the main difference between highlighter and bronzer?",
        "options": [
            "Highlighter is matte, bronzer is shiny",
            "Highlighter adds glow to high points, bronzer adds warmth and dimension",
            "They are the same product with different names",
            "Highlighter is only for eyes, bronzer is for cheeks"
        ],
        "correct_answer": 1,
        "topic": "Face Makeup"
    },
    {
        "id": 14,
        "question": "What type of lip product provides the longest wear?",
        "options": [
            "Lip gloss",
            "Liquid lipstick",
            "Lip balm",
            "Tinted lip oil"
        ],
        "correct_answer": 1,
        "topic": "Lip Makeup"
    },
    {
        "id": 15,
        "question": "What is a makeup primer's main ingredient category?",
        "options": [
            "Water-based or silicone-based",
            "Oil-based only",
            "Alcohol-based",
            "Wax-based"
        ],
        "correct_answer": 0,
        "topic": "Makeup Chemistry"
    }
]

# Store quiz sessions and submissions
quiz_sessions: Dict[str, List[Dict]] = {}
quiz_submissions: List[Dict] = []


# Pydantic models for quiz submission
class QuizAnswer(BaseModel):
    """Model for a single quiz answer"""
    question_id: int
    answer: int = Field(..., ge=0, le=3, description="Answer index (0-3)")


class QuizSubmission(BaseModel):
    """Model for quiz submission from your app"""
    email: str
    quiz_url: str
    answers: List[QuizAnswer]
    timestamp: Optional[str] = None


@app.get("/", response_class=HTMLResponse)
async def root():
    """Display a simple quiz page"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Makeup Quiz - Test Platform</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 50px auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 {
                color: #e91e63;
                text-align: center;
            }
            .question {
                margin: 20px 0;
                padding: 20px;
                background: #fff9f9;
                border-left: 4px solid #e91e63;
                border-radius: 5px;
            }
            .options {
                margin: 10px 0;
            }
            .option {
                padding: 10px;
                margin: 5px 0;
                background: #f9f9f9;
                border: 1px solid #ddd;
                border-radius: 5px;
                cursor: pointer;
            }
            .option:hover {
                background: #ffe4f0;
            }
            button {
                background: #e91e63;
                color: white;
                border: none;
                padding: 12px 30px;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
                margin: 10px 5px;
            }
            button:hover {
                background: #c2185b;
            }
            .info {
                background: #e3f2fd;
                padding: 15px;
                border-radius: 5px;
                margin: 20px 0;
            }
            code {
                background: #f5f5f5;
                padding: 2px 6px;
                border-radius: 3px;
                font-family: monospace;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>💄 Makeup Quiz - Test Platform</h1>
            
            <div class="info">
                <h3>API Endpoints Available:</h3>
                <ul>
                    <li><code>GET /</code> - This page</li>
                    <li><code>GET /demo</code> - Demo quiz (3 questions)</li>
                    <li><code>GET /full-quiz</code> - Full quiz (15 questions)</li>
                    <li><code>GET /random-quiz/{num_questions}</code> - Custom number of questions</li>
                    <li><code>GET /api/questions</code> - Get all questions as JSON</li>
                    <li><code>GET /docs</code> - Interactive API documentation</li>
                </ul>
            </div>

            <div class="question">
                <h3>Sample Question:</h3>
                <p><strong>What is the primary purpose of primer in makeup application?</strong></p>
                <div class="options">
                    <div class="option">A) To add color to the skin</div>
                    <div class="option">B) To create a smooth base and help makeup last longer ✓</div>
                    <div class="option">C) To remove makeup at the end of the day</div>
                    <div class="option">D) To moisturize dry skin</div>
                </div>
            </div>

            <div style="text-align: center; margin-top: 30px;">
                <button onclick="window.location.href='/demo'">Start Demo Quiz</button>
                <button onclick="window.location.href='/full-quiz'">Full Quiz</button>
                <button onclick="window.location.href='/docs'">API Docs</button>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/demo", response_class=HTMLResponse)
async def demo_quiz():
    """Demo quiz with 3 questions"""
    questions = random.sample(MAKEUP_QUIZ_QUESTIONS, 3)
    return generate_quiz_html(questions, "Demo Quiz - 3 Questions")


@app.get("/full-quiz", response_class=HTMLResponse)
async def full_quiz():
    """Full quiz with all questions"""
    return generate_quiz_html(MAKEUP_QUIZ_QUESTIONS, "Full Makeup Quiz - 15 Questions")


@app.get("/random-quiz/{num_questions}", response_class=HTMLResponse)
async def random_quiz(num_questions: int):
    """Generate a quiz with specified number of questions"""
    if num_questions < 1 or num_questions > len(MAKEUP_QUIZ_QUESTIONS):
        raise HTTPException(status_code=400, detail=f"Number of questions must be between 1 and {len(MAKEUP_QUIZ_QUESTIONS)}")
    
    questions = random.sample(MAKEUP_QUIZ_QUESTIONS, num_questions)
    return generate_quiz_html(questions, f"Random Quiz - {num_questions} Questions")


@app.get("/api/questions")
async def get_all_questions():
    """Get all questions as JSON (useful for testing your solver)"""
    return {
        "total_questions": len(MAKEUP_QUIZ_QUESTIONS),
        "questions": MAKEUP_QUIZ_QUESTIONS
    }


def generate_quiz_html(questions: List[Dict], title: str) -> HTMLResponse:
    """Generate HTML for quiz questions"""
    questions_html = ""
    for idx, q in enumerate(questions, 1):
        options_html = ""
        for opt_idx, option in enumerate(q["options"]):
            options_html += f"""
                <div class="option">
                    <input type="radio" name="q{q['id']}" value="{opt_idx}" id="q{q['id']}_opt{opt_idx}">
                    <label for="q{q['id']}_opt{opt_idx}">{option}</label>
                </div>
            """
        
        questions_html += f"""
            <div class="question" id="question-{q['id']}">
                <h3>Question {idx}</h3>
                <p><strong>{q['question']}</strong></p>
                <div class="options">
                    {options_html}
                </div>
                <p style="color: #666; font-size: 0.9em;">Topic: {q['topic']}</p>
            </div>
        """
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                max-width: 900px;
                margin: 30px auto;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #e91e63;
                text-align: center;
                margin-bottom: 30px;
            }}
            .question {{
                margin: 25px 0;
                padding: 20px;
                background: #fff9f9;
                border-left: 4px solid #e91e63;
                border-radius: 5px;
            }}
            .question h3 {{
                color: #e91e63;
                margin-top: 0;
            }}
            .options {{
                margin: 15px 0;
            }}
            .option {{
                padding: 10px;
                margin: 8px 0;
                background: white;
                border: 2px solid #ddd;
                border-radius: 5px;
                cursor: pointer;
                display: flex;
                align-items: center;
            }}
            .option:hover {{
                background: #ffe4f0;
                border-color: #e91e63;
            }}
            .option input[type="radio"] {{
                margin-right: 10px;
                cursor: pointer;
            }}
            .option label {{
                cursor: pointer;
                flex-grow: 1;
            }}
            button {{
                background: #e91e63;
                color: white;
                border: none;
                padding: 12px 30px;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
                margin: 10px 5px;
            }}
            button:hover {{
                background: #c2185b;
            }}
            .button-container {{
                text-align: center;
                margin-top: 30px;
                padding-top: 20px;
                border-top: 2px solid #f0f0f0;
            }}
            .info-box {{
                background: #e3f2fd;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>💄 {title}</h1>
            
            <div class="info-box">
                <strong>Instructions:</strong> Select the best answer for each question.
                This is a test quiz for your makeup knowledge!
            </div>
            
            <form id="quizForm">
                {questions_html}
                
                <div class="button-container">
                    <button type="submit">Submit Quiz</button>
                    <button type="button" onclick="window.location.href='/'">Back to Home</button>
                </div>
            </form>
        </div>

        <script>
            document.getElementById('quizForm').addEventListener('submit', function(e) {{
                e.preventDefault();
                alert('Quiz submitted! In a real application, this would be scored.');
            }});
        </script>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)


@app.post("/submit")
async def submit_quiz(submission: QuizSubmission):
    """
    Endpoint to receive quiz answers from your app
    Validates and scores the submission
    """
    timestamp = submission.timestamp or datetime.now().isoformat()
    
    # Score the answers
    correct_count = 0
    total_questions = len(submission.answers)
    detailed_results = []
    
    for answer in submission.answers:
        # Find the question
        question = next((q for q in MAKEUP_QUIZ_QUESTIONS if q["id"] == answer.question_id), None)
        
        if question:
            is_correct = answer.answer == question["correct_answer"]
            if is_correct:
                correct_count += 1
            
            detailed_results.append({
                "question_id": answer.question_id,
                "question": question["question"],
                "submitted_answer": answer.answer,
                "correct_answer": question["correct_answer"],
                "is_correct": is_correct,
                "topic": question["topic"]
            })
        else:
            detailed_results.append({
                "question_id": answer.question_id,
                "error": "Question not found",
                "is_correct": False
            })
    
    # Calculate score
    score_percentage = (correct_count / total_questions * 100) if total_questions > 0 else 0
    
    # Store submission
    submission_record = {
        "email": submission.email,
        "quiz_url": submission.quiz_url,
        "timestamp": timestamp,
        "total_questions": total_questions,
        "correct_answers": correct_count,
        "score_percentage": score_percentage,
        "detailed_results": detailed_results
    }
    quiz_submissions.append(submission_record)
    
    # Return results
    return JSONResponse(
        status_code=200,
        content={
            "status": "success",
            "message": "Quiz submitted successfully",
            "results": {
                "total_questions": total_questions,
                "correct_answers": correct_count,
                "incorrect_answers": total_questions - correct_count,
                "score_percentage": round(score_percentage, 2),
                "grade": get_grade(score_percentage),
                "timestamp": timestamp
            },
            "detailed_results": detailed_results
        }
    )


@app.get("/submissions")
async def get_submissions():
    """Get all quiz submissions"""
    return {
        "total_submissions": len(quiz_submissions),
        "submissions": quiz_submissions
    }


@app.get("/submissions/latest")
async def get_latest_submission():
    """Get the most recent submission"""
    if not quiz_submissions:
        raise HTTPException(status_code=404, detail="No submissions found")
    
    return {
        "submission": quiz_submissions[-1]
    }


def get_grade(score: float) -> str:
    """Convert score percentage to letter grade"""
    if score >= 90:
        return "A (Excellent)"
    elif score >= 80:
        return "B (Good)"
    elif score >= 70:
        return "C (Average)"
    elif score >= 60:
        return "D (Below Average)"
    else:
        return "F (Fail)"


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "total_questions": len(MAKEUP_QUIZ_QUESTIONS),
        "total_submissions": len(quiz_submissions),
        "endpoints": [
            "/demo", "/full-quiz", "/random-quiz/{num}", 
            "/api/questions", "/submit", "/submissions"
        ]
    }


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*70)
    print("🎨 MAKEUP QUIZ SERVER - Starting...")
    print("="*70)
    print("\n📝 Available endpoints:")
    print("   - http://localhost:8000/          → Home page")
    print("   - http://localhost:8000/demo      → Demo quiz (3 questions)")
    print("   - http://localhost:8000/full-quiz → Full quiz (15 questions)")
    print("   - http://localhost:8000/docs      → Interactive API docs")
    print("\n" + "="*70 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
