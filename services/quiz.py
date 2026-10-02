"""
Quiz Generation Service
Generates structured MCQs with validated schemas using Mistral AI JSON mode.
"""

from typing import List, Dict, Any, Optional
from prompts.prompts import QUIZ_GENERATION_PROMPT
from services.llm import generate_structured_json, LLMError
from config import DEFAULT_MISTRAL_MODEL


class QuizValidationError(Exception):
    """Raised when the LLM output does not match the required Quiz schema."""
    pass


def validate_quiz_schema(quiz_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate that the generated quiz object contains all required fields and valid choices.

    Args:
        quiz_data: Dictionary parsed from LLM JSON output.

    Returns:
        Validated and normalized quiz dictionary.
    """
    if not isinstance(quiz_data, dict):
        raise QuizValidationError("Generated quiz is not a JSON object.")

    questions = quiz_data.get("questions")
    if not isinstance(questions, list) or len(questions) == 0:
        raise QuizValidationError("Quiz output is missing a non-empty 'questions' list.")

    validated_questions = []
    for idx, q in enumerate(questions, start=1):
        if not isinstance(q, dict):
            continue

        q_text = q.get("question", "").strip()
        options = q.get("options", [])
        correct_ans = q.get("correct_answer", "").strip()
        explanation = q.get("explanation", "").strip()

        if not q_text or not isinstance(options, list) or len(options) < 2:
            continue

        # Ensure 4 options format
        normalized_options = [str(opt).strip() for opt in options]

        validated_questions.append({
            "id": idx,
            "question": q_text,
            "options": normalized_options,
            "correct_answer": correct_ans,
            "explanation": explanation or "No explanation provided."
        })

    if not validated_questions:
        raise QuizValidationError("No valid questions could be extracted from the LLM response.")

    return {
        "topic": quiz_data.get("topic", "General Study"),
        "difficulty": quiz_data.get("difficulty", "Medium"),
        "questions": validated_questions
    }


def generate_study_quiz(
    topic_or_context: str,
    num_questions: int = 5,
    difficulty: str = "Medium",
    model: str = DEFAULT_MISTRAL_MODEL,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate a validated MCQ quiz from topic or study material context.

    Args:
        topic_or_context: Topic name or document text excerpt.
        num_questions: Number of questions (e.g. 5, 10).
        difficulty: 'Easy', 'Medium', or 'Hard'.
        model: Mistral model name.
        api_key: Mistral API key.

    Returns:
        Dict matching validated quiz schema.
    """
    prompt_text = QUIZ_GENERATION_PROMPT.format(
        topic=topic_or_context[:1500],
        num_questions=num_questions,
        difficulty=difficulty
    )

    messages = [
        {"role": "system", "content": "You are a specialized examination engine that outputs valid JSON only."},
        {"role": "user", "content": prompt_text}
    ]

    try:
        raw_json = generate_structured_json(
            messages=messages,
            model=model,
            temperature=0.2,
            max_tokens=2500,
            api_key=api_key
        )
        return validate_quiz_schema(raw_json)
    except Exception as e:
        raise LLMError(f"Failed to generate structured quiz: {str(e)}")
