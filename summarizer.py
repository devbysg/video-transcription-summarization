
from langchain_core.prompts import ChatPromptTemplate


MIN_TRANSCRIPT_LENGTH = 200


def create_summary_chain(llm):

    prompt = ChatPromptTemplate([
        (
            "system",
            "You are an expert video summarization assistant."
        ),
        (
            "user",
            """Summarize this transcript.

Give:
1. Main topic
2. Key points
3. Important details
4. Final takeaway

Transcript:
{transcript}
"""
        )
    ])

    return prompt | llm


def summarize_transcript(transcript: str, llm):

    transcript = transcript.strip()

    # Case 1: Transcript is shorter than 200 characters
    if len(transcript) < MIN_TRANSCRIPT_LENGTH:

        return {
            "transcript": transcript,
            "summary": "Too small to summarise."
        }

    # Case 2: Transcript is 200 characters or longer
    chain = create_summary_chain(llm)

    result = chain.invoke({
        "transcript": transcript
    })

    return {
        "transcript": None,
        "summary": result.content
    }


if __name__ == "__main__":

    from dotenv import load_dotenv
    from langchain_groq import ChatGroq

    load_dotenv()

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0
    )

    test_transcript = """
    Data analytics plays an important role in modern organizations.
    Data analysts collect information from different sources, clean
    the data, analyze historical trends, identify patterns, and create
    reports that help business teams make informed decisions. SQL is
    commonly used to retrieve and transform data from databases, while
    Power BI can be used to create interactive dashboards and track
    business KPIs. A good analyst also needs to understand the business
    problem before selecting metrics and interpreting the results.
    """

    result = summarize_transcript(test_transcript, llm)

    print("\n--- RESULT ---")

    if result["transcript"] is not None:
        print("Transcript:")
        print(result["transcript"])

    print("\nSummary:")
    print(result["summary"])
