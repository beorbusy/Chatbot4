# good ai model but slow on high data , works best when it is in passage.
from transformers import pipeline

# Step 1: Load the BERT-based question-answering model
nlp = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

# Step 2: Define a 4-line passage (database) about Kailash Yatra
passage = """
The Kailash Manasarovar Yatra is a spiritual journey organized for pilgrims.
The yatra involves trekking in high altitudes, and physical fitness is important.
Mount Kailash and Lake Manasarovar are the main attractions of this journey.
Due to the sacred nature of the sites, some areas may have restrictions on activities like photography.
sadhguru is a yogi and mystic.
sadhguru is not scaduled for any yatra this year.
sadhguru will not be joing with us
sadhguru will not be coming
we will be going to sadhguru sri barhma ashram.
"""

# Step 3: Function to handle user query and get answer from BERT model, including confidence score
def ask_bot(query, passage):
    result = nlp(question=query, context=passage)

    # Extract the answer and confidence score
    answer = result['answer']
    confidence = result['score'] * 100  # Convert to percentage

    return answer, confidence

# Step 4: Main loop to take user input and respond with answer and confidence level
while True:
    user_query = input("Ask me anything about Kailash Yatra: ")  # Get user input

    if user_query.lower() in ['exit', 'quit']:
        print("Goodbye!")
        break

    # Get the AI-generated response and confidence
    response, confidence = ask_bot(user_query, passage)

    # Print the response along with the confidence level
    print(f"Bot: {response} (Confidence: {confidence:.2f}%)")
