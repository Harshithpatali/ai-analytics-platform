"""
AI business insights service.
"""

import os

from groq import Groq


class AIInsightsService:
    """
    Generate AI-powered insights.
    """

    @staticmethod
    def generate_insights(
        dataset_summary,
        model_metrics
    ):
        """
        Generate business insights.
        """

        api_key = os.getenv(
            "GROQ_API_KEY"
        )

        if not api_key:

            return (
                "GROQ_API_KEY not configured."
            )

        client = Groq(
            api_key=api_key
        )

        prompt = f"""
        You are an expert AI business analyst.

        Dataset Summary:
        {dataset_summary}

        Model Metrics:
        {model_metrics}

        Generate:
        1. Business insights
        2. Risk analysis
        3. Opportunities
        4. Recommendations
        5. Executive summary
        """

        response = ( client.chat.completions.create( model="llama-3.3-70b-versatile", messages=[ { "role": "user", "content": prompt } ], temperature=0.3, max_tokens=1024 ) )
        return (
            response
            .choices[0]
            .message.content
        )