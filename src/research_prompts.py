#!/usr/bin/env python3
"""
LLM prompt templates for the Research Agent.
"""

BRAND_URL_ANALYSIS_PROMPT = """You are a brand analyst. Analyze the following website HTML content and extract key brand information.

Website URL: {url}

HTML Content (trimmed):
{html_content}

Extract the following in JSON format:
{{
  "products": ["list of main products or services offered"],
  "style": "overall brand aesthetic/style description (e.g. modern, minimalist, luxury, playful)",
  "colors": ["list of primary brand colors as hex codes"],
  "fonts": ["list of font families used"],
  "target_audience": "who the brand appears to target",
  "brand_voice": "description of the brand's tone and voice",
  "key_messaging": ["list of key marketing messages or taglines"],
  "raw_insights": "2-3 paragraph summary of the brand's positioning, unique value proposition, and visual identity"
}}

Return ONLY valid JSON, no markdown."""

VIDEO_UNDERSTANDING_PROMPT = """You are a video content analyst specializing in social media marketing.
Analyze this video in detail and extract the following information.

Return your analysis as JSON:
{{
  "hook_type": "type of hook used (question, statement, visual, sound, controversy, curiosity_gap, pattern_interrupt)",
  "hook_text": "the exact text/words used in the opening hook (first 3 seconds)",
  "hook_effectiveness": "rating 1-10 and brief explanation",
  "num_scenes": number of distinct scenes/cuts,
  "duration": estimated duration in seconds,
  "pacing": "slow/medium/fast - description of editing rhythm",
  "characters": [
    {{
      "age_range": "estimated age range (e.g. 25-34)",
      "gender": "perceived gender",
      "description": "brief description of appearance, clothing, role in video"
    }}
  ],
  "textures": ["list of notable textures/materials visible (e.g. silk, denim, concrete, wood)"],
  "objects": ["list of key objects/props visible"],
  "colors": ["dominant color palette in the video"],
  "transcription": "full text transcription of any speech or on-screen text",
  "music_mood": "description of background music mood and energy",
  "cta": "call to action if present",
  "content_format": "tutorial/showcase/behind-the-scenes/transformation/day-in-life/testimonial/trend/unboxing/comparison/storytelling",
  "success_factors": ["list of 3-5 factors that make this content engaging"],
  "improvement_suggestions": ["list of 1-3 things that could be improved"]
}}

Return ONLY valid JSON, no markdown."""

COMPETITIVE_SUCCESS_PROMPT = """You are a social media strategist analyzing competitor content performance.

Brand: {brand_username} ({brand_followers:,} followers)
Competitor: {competitor_username} ({competitor_followers:,} followers)

Here are the top performing posts/reels from the competitor:

{top_content_summary}

And here are the top performing posts/reels from the brand:

{brand_content_summary}

Analyze the competitor's content strategy and provide insights:

1. What content themes/topics perform best for the competitor?
2. What posting patterns (format, length, style) drive highest engagement?
3. How does the competitor's approach differ from the brand's?
4. What specific tactics could the brand adopt from the competitor?
5. What content gaps exist that the brand could exploit?

Return your analysis as JSON:
{{
  "success_themes": ["list of top 3-5 content themes that drive engagement"],
  "format_insights": "what formats/styles work best",
  "differentiators": ["how competitor differs from brand"],
  "actionable_recommendations": ["specific tactics to adopt"],
  "content_gaps": ["opportunities the brand is missing"],
  "summary": "2-3 paragraph executive summary of competitive analysis"
}}

Return ONLY valid JSON, no markdown."""

FINANCIAL_EXTRACTION_PROMPT = """You are a financial analyst. Extract key financial metrics from the following web content about {company_name}.

Content:
{web_content}

Extract whatever financial data is available and return as JSON:
{{
  "revenue": "most recent annual revenue (e.g. '$1.5B')",
  "revenue_yoy_growth": "year-over-year revenue growth percentage",
  "active_subscribers": "number of active subscribers/customers if available",
  "market_cap": "current market capitalization",
  "stock_price": "current stock price if public",
  "key_metrics": [
    {{"metric": "metric name", "value": "metric value", "period": "time period"}}
  ],
  "recent_highlights": ["list of notable recent business developments"],
  "data_date": "approximate date of the financial data",
  "source_quality": "high/medium/low - confidence in data accuracy"
}}

If a field is not available in the content, set it to null.
Return ONLY valid JSON, no markdown."""
