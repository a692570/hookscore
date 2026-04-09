"""Basic usage example for hookscore."""

from hookscore import score_text, analyze, format_report

text = """
We spent three years building what everyone said was impossible.
Then we launched on a Tuesday. By Friday, we had 10,000 users.
The investors who passed? They called back.
"""

results = score_text(
    text,
    api_key="your-api-key-here",  # or set HOOKSCORE_API_KEY env var
    # api_base="https://api.openai.com/v1",  # default
    # model="gpt-4o-mini",  # default
)

report = analyze(results)
print(format_report(report))
