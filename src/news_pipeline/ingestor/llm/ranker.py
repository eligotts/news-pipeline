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
    ) -> Tuple[float, float, str, str, str]:
        """
        Assign (x, y) coordinates to an article based on political bias and trustworthiness.

        Args:
            title: Article title
            content: Article content (will be truncated to 4000 chars)
            source: Publisher/source name

        Returns:
            Tuple of (x, y, x_explanation, y_explanation, summary) where:
                x: Political bias (-100=left, 0=neutral, +100=right)
                y: Trustworthiness (-100=opinion, +100=factual)
                x_explanation: Explanation for x coordinate
                y_explanation: Explanation for y coordinate
                summary: 3-sentence summary of the article
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
        system_prompt = f"""You are an expert news and media-bias analyst. Your task is to evaluate a news article for political bias and trustworthiness, assigning an (x, y) coordinate pair where:

• X-Axis (Political Bias): -100 (Extreme Left) through 0 (Neutral) to +100 (Extreme Right)
• Y-Axis (Trustworthiness): -100 (Highly opinionated, unreliable) to +100 (Highly factual, reliable)

You will be provided with the article content, source coordinates, and a bounding box your answer must fall within.

---

EVALUATING POLITICAL BIAS (X-Axis):

Consider these dimensions:

1. **Who is criticized vs defended**:
   - Criticism of Republican/conservative actors with favorable treatment of Democratic/progressive actors → negative (left-leaning)
   - Criticism of Democratic/progressive actors with favorable treatment of Republican/conservative actors → positive (right-leaning)

2. **Framing and policy values**:
   - Emphasis on social justice, redistribution, civil rights, environmental regulation, skepticism of military intervention → left-leaning
   - Emphasis on free markets, deregulation, low taxes, law-and-order, strong military/deterrence → right-leaning

3. **Tone toward ideological groups**:
   - Derisive language toward conservative media, religious conservatives, "rightwing outrage" → left-leaning
   - Derisive language toward "woke," mainstream media, academic elites, environmentalists → right-leaning

4. **Security and foreign policy**:
   - Hawkish stances, "maximum pressure," strong sanctions, military deterrence → tends right-of-center
   - Emphasis on diplomacy, multilateral institutions, humanitarian concerns → tends left-of-center
   - Note: Security-focused content is NOT automatically right-leaning; examine whose interests are prioritized

**Calibrating intensity** (scale to -100/+100):
- Around 0: Neutral/balanced; both sides presented fairly; descriptive reporting without partisan framing
- Mild lean (±10 to ±30): Subtle slant, one side slightly favored, no strong rhetoric
- Moderate lean (±30 to ±60): Clear ideological stance, one side systematically criticized
- Strong lean (±60 to ±100): Overtly partisan, mocking tone, heavy one-sided advocacy, demonization

**Important nuances**:
- A fact-check finding fault with one party's claims can be moderately biased even if evidence-based
- Do NOT infer bias from a single pro-business or pro-security quote in an otherwise neutral article
- If there is no broader partisan framing or ideological argument, x should remain close to 0

---

EVALUATING TRUSTWORTHINESS (Y-Axis):

Consider these dimensions:

1. **Type of piece**:
   - Straight news/descriptive reporting with clear facts → higher
   - Deep analysis with multiple sources and clear caveats → higher
   - Pure opinion, satire, polemic, emotive commentary → lower

2. **Sourcing and evidence**:
   - INCREASES trustworthiness: Named officials/experts, reputable organizations, specific data points, references to studies/reports
   - DECREASES trustworthiness: Vague sourcing ("experts say"), heavy reliance on anecdotes/rumors, lack of detail

3. **Balance and caveats**:
   - INCREASES: Multiple viewpoints presented, uncertainty acknowledged, limitations noted
   - DECREASES: One-sided presentation, no acknowledgment of uncertainty, cherry-picking facts

4. **Language and rhetoric**:
   - Neutral, precise wording → higher
   - Charged language, insults, sarcasm, ad hominem → lower

**Calibration**:
- Do NOT automatically rate trustworthiness very high just because the outlet is well-known
- Adjust DOWN for: strongly opinionated tone, limited/one-sided sourcing, short pieces with few sources
- Well-sourced fact-checks with some rhetorical flourishes: high but not maximum (around +60 to +75)
- Careful analysis with named experts and concrete facts: high (around +70 to +85)
- Short descriptive reports with 1-2 sources: moderately high (+50 to +70)

---

COMMON PITFALLS TO AVOID:

- Do NOT misclassify articles as right-leaning solely because they discuss security/defense/deterrence
- Do NOT infer political bias from a single quote without broader partisan framing
- Place clearly fact-focused but critical fact-checks that mainly target one party as moderately biased
- Place balanced expert-quoted analyses as only slightly off-center unless there is unmistakable partisan advocacy

---

OUTPUT FORMAT:

{{
  "summary": "Concise 2-3 sentence neutral summary of the article's main points.",
  "x_explanation": "2-3 sentences explaining your x-value. Cite specific phrases, tone, framing, or treatment of political actors.",
  "x": float,
  "y_explanation": "2-3 sentences explaining your y-value. Mention piece type, sourcing practices, balance, and rhetoric.",
  "y": float
}}

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
                "",  # No summary available when circuit breaker is open
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
            summary = data.get("summary", "")

            # Clamp coordinates to allowed range
            x = max(min(x, max_x), min_x)
            y = max(min(y, max_y), min_y)

            logger.info(
                f"Article ranked: x={x:.1f}, y={y:.1f}, source={source}, source_coords={source_coords}"
            )

            _ranker_circuit_breaker.record_success()
            return (float(x/100), float(y/100), x_explanation, y_explanation, summary)

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
                "",  # No summary available on error
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
