from __future__ import annotations

import json
import logging
from typing import Any, Optional, Tuple

from .client import create_openrouter_client
from ..core import retry_with_backoff, CircuitBreaker
from ..data import get_source_coordinates

logger = logging.getLogger(__name__)

# Circuit breaker for coordinate ranking LLM API
_ranker_circuit_breaker = CircuitBreaker(
    name="openrouter_ranker",
    failure_threshold=5,
    recovery_timeout=60.0,
)


# JSON Schema for structured output
COORDINATE_RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "coordinate_ranking",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Concise 3-sentence summary of the news article"
                },
                "x_explanation": {
                    "type": "string",
                    "description": "2-3 sentence explanation for the X-axis political bias classification"
                },
                "x": {
                    "type": "number",
                    "description": "Political bias score from -100 (left) to +100 (right)"
                },
                "y_explanation": {
                    "type": "string",
                    "description": "2-3 sentence explanation for the Y-axis trustworthiness classification"
                },
                "y": {
                    "type": "number",
                    "description": "Trustworthiness score from -100 (opinion) to +100 (factual)"
                }
            },
            "required": ["summary", "x_explanation", "x", "y_explanation", "y"],
            "additionalProperties": False
        }
    }
}


class CoordinateRanker:
    """Assigns political bias (x) and trustworthiness (y) coordinates to articles."""

    def __init__(
        self,
        openrouter_api_key: str,
        model: str = "google/gemini-3-flash-preview",
        base_url: str = "https://openrouter.ai/api/v1",
    ):
        self._client: Any = create_openrouter_client(api_key=openrouter_api_key, base_url=base_url)
        self._model = model

    def rank_article(
        self,
        title: str,
        content: str,
        source: Optional[str] = None,
    ) -> Tuple[float, float, str, str]:
        """
        Assign (x, y) coordinates to an article based on political bias and trustworthiness.

        Args:
            title: Article title
            content: Article content (will be truncated to 4000 chars)
            source: Publisher/source name

        Returns:
            Tuple of (x, y, x_explanation, y_explanation) where:
                x: Political bias (-100=left, 0=neutral, +100=right)
                y: Trustworthiness (-100=opinion, +100=factual)
                x_explanation: Explanation for x coordinate
                y_explanation: Explanation for y coordinate
        """
        # Get source coordinates as starting point
        source_coords = None
        if source:
            source_coords = get_source_coordinates(source)

        # If no source coordinates found, use neutral defaults
        if source_coords is None:
            logger.warning(f"Source coordinates not found for '{source}', using defaults (0, 50)")
            source_coords = (0.0, 50.0)

        # Calculate constrained window (±15 units)
        min_x = max(source_coords[0] - 15, -100)
        max_x = min(source_coords[0] + 15, 100)
        min_y = max(source_coords[1] - 15, -100)
        max_y = min(source_coords[1] + 15, 100)

        # Truncate content to fit API limits
        truncated_content = content[:4000] if content else ""

        # Build the prompt with JSON schema
        system_prompt = f"""You are an expert news analyst tasked with evaluating a provided news article based on political bias and trustworthiness. You will assign an (x, y) coordinate pair to each article, where:
• X-Axis (Political Bias) ranges from -100 (Extreme Left-leaning) through 0 (Neutral) to +100 (Extreme Right-leaning).
• Y-Axis (Trustworthiness) ranges from -100 (Highly opinionated, unreliable) to +100 (Highly factual, reliable).

You will be provided with:
• The article content.
• The article's source coordinates (x, y).
• A maximum allowable distance from these coordinates (creating a bounding box that your answer must fall within).

Output Structure:

Your output must strictly follow this JSON schema:
{{
  "summary": "string - Provide a concise, approximately 3-sentence, high-quality journalistic summary of the news article. Highlight key facts, events, and how they are framed.",
  "x_explanation": "string - Clearly articulate why you have classified the article at the chosen point on the X-axis. Be highly specific, referring explicitly to particular aspects of the article (tone, framing, language, focus areas) that influenced your evaluation. Be concise. This should be no more than 2-3 sentences.",
  "x": float,
  "y_explanation": "string - Clearly articulate why you have classified the article at the chosen point on the Y-axis. Provide precise explanations regarding the factual accuracy, reliability of sources, clarity in differentiating facts from opinions, and adherence to journalistic standards. Be concise. This should be no more than 2-3 sentences.",
  "y": float
}}

Guidelines for Your Evaluation:

Step 1 - Summary:
• Read and succinctly summarize the article, capturing the essence and framing without unnecessary detail.

Step 2 - Evaluate Political Bias (X-Axis):
• Carefully analyze the article for political orientation. Consider explicit or implicit cues like choice of language, emphasis, and the framing of events.
• Discuss specific elements or sections that directly influenced your assessment.

Step 3 - Evaluate Trustworthiness (Y-Axis):
• Assess the factual accuracy, quality of sourcing, and clarity of separation between facts and opinion.
• Identify explicit aspects of the article that uphold or diminish its trustworthiness.

Step 4 - Coordinate Determination:
• Ensure your chosen coordinates logically align with your detailed analyses.
• Confirm your coordinates fit within the provided allowable range.

Do not provide an overview beyond the requested summary and explanations; instead, focus directly and precisely on elements connecting explicitly to your coordinate classifications. Remember, be concise. The summary should be no more than 3 sentences. The x_explanation and y_explanation should be no more than 2-3 sentences.

Article title: {title}

Article content: {truncated_content}

Source: {source or "Unknown"}

Source coordinates: {source_coords}

Minimum x-coordinate allowed: {min_x}
Maximum x-coordinate allowed: {max_x}
Minimum y-coordinate allowed: {min_y}
Maximum y-coordinate allowed: {max_y}"""

        # Check circuit breaker before making request
        if not _ranker_circuit_breaker.allow_request():
            logger.warning("Ranker circuit breaker is open - falling back to source coordinates")
            return (
                float(source_coords[0]/100),
                float(source_coords[1]/100),
                f"Circuit breaker open - using source baseline for {source or 'unknown source'}",
                f"Circuit breaker open - using source baseline for {source or 'unknown source'}",
            )

        try:
            response = self._call_llm_with_retry(system_prompt)

            # Parse response
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response from LLM")

            data = json.loads(content)

            # Extract and validate fields
            x = float(data.get("x", source_coords[0]))
            y = float(data.get("y", source_coords[1]))
            x_explanation = data.get("x_explanation", "")
            y_explanation = data.get("y_explanation", "")

            # Clamp coordinates to allowed range
            x = max(min(x, max_x), min_x)
            y = max(min(y, max_y), min_y)

            logger.info(
                f"Article ranked: x={x:.1f}, y={y:.1f}, source={source}, source_coords={source_coords}"
            )

            _ranker_circuit_breaker.record_success()
            return (float(x/100), float(y/100), x_explanation, y_explanation)

        except Exception as exc:
            _ranker_circuit_breaker.record_failure()
            logger.error(f"Error ranking article coordinates: {exc}")
            # Fall back to source coordinates
            logger.warning(f"Falling back to source coordinates: {source_coords}")
            return (
                float(source_coords[0]/100),
                float(source_coords[1]/100),
                f"Using source baseline for {source or 'unknown source'}",
                f"Using source baseline for {source or 'unknown source'}",
            )

    @retry_with_backoff(max_retries=3, base_delay=1.0, max_delay=30.0)
    def _call_llm_with_retry(self, system_prompt: str):
        """Make LLM call with retry logic."""
        return self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": "Analyze the article and respond in valid JSON format with the exact schema specified.",
                },
            ],
            temperature=0.3,
            max_tokens=1000,
            response_format=COORDINATE_RESPONSE_SCHEMA,
        )
