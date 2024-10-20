
import json
import warnings
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from fuzzywuzzy import fuzz, process
from flask import Flask, request, render_template
from pyngrok import ngrok  # Import the ngrok module

# Suppress warnings
warnings.filterwarnings("ignore")

# Load the pre-trained Question-Answering model
nlp = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

# File names for the JSON databases
json_file = "database.json"
blacklist_file = "blacklist.json"

# --- Functions to Load and Save JSON ---
def load_json_database(filename):
    try:
        with open(filename, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"{filename} not found. Creating a new one.")
        return {}

def save_json_database(database, filename):
    with open(filename, "w") as file:
        json.dump(database, file, indent=4)
    print(f"Database saved to {filename}")

# Load the databases at the start
database = load_json_database(json_file)
blacklist = load_json_database(blacklist_file)

# --- Core Functions ---
def current_yatra(query):
    query_lower = query.lower()
    if "kailash" in query_lower or "kailasa" in query_lower:
        return "kailash_manasarovar"
    elif "himalaya" in query_lower or "himalayas" in query_lower:
        return "himalayas"
    elif "south" in query_lower or "southern" in query_lower:
        return "southern_sojourn"
    elif "kashi" in query_lower or "kashi krama" in query_lower:
        return "kashi_krama"
    else:
        return "common_questions"

def is_blacklisted(query, answer):
    blacklisted_answers = blacklist.get(query, [])
    return answer in blacklisted_answers

def fuzzy_answer(query, yatra):
    questions = [item.get("question", "") for item in database.get(yatra, [])]
    if questions:
        all_matches = process.extract(query, questions, scorer=fuzz.partial_ratio, limit=5)
        for match, score in all_matches:
            if score > 80:
                for item in database[yatra]:
                    if item.get("question") == match:
                        answer = item.get("answer", item.get("context", "No answer available"))
                        if not is_blacklisted(query, answer):
                            return answer, score
        return None, None
    return None, None

# --- Semantic Search AI Answer ---
def generate_embeddings(model, database):
    embeddings = {}
    for category, qas in database.items():
        embeddings[category] = []
        for qa in qas:
            question = qa["question"]
            question_embedding = model.encode(question, convert_to_tensor=True)
            context = qa.get("context", "No context available")

            embeddings[category].append({
                "question": question,
                "context": context,
                "embedding": question_embedding
            })
    return embeddings

def find_best_match(query, embeddings, model):
    query_embedding = model.encode(query, convert_to_tensor=True)
    best_context = ""
    best_score = -1

    for category, qa_list in embeddings.items():
        for qa in qa_list:
            similarity = util.pytorch_cos_sim(query_embedding, qa["embedding"])[0][0].item()
            if similarity > best_score:
                if not is_blacklisted(query, qa["context"]):
                    best_score = similarity
                    best_context = qa["context"]

    return best_context if best_score > 0.2 else "I'm sorry, I don't have an answer for that."

# --- Handle User-provided Answers ---
def save_to_database(question, context, yatra=None):
    yatra = yatra if yatra else "common_questions"
    if yatra in database:
        database[yatra].append({"question": question, "context": context})
    else:
        database[yatra] = [{"question": question, "context": context}]
    save_json_database(database, json_file)

# --- Handle User-provided Answers ---
def user_answer(query, yatra):
    print(f"No answer found for '{query}'.")
    user_answer = input(f"Please provide an answer: ")
    save_to_database(query, user_answer, yatra)
    return user_answer

# --- Handle Feedback ---
def handle_feedback(user_query, main_answer, yatra, feedback):
    if feedback == "like":
        print("Thanks for the feedback!")
        save_to_database(user_query, main_answer, yatra)
    elif feedback == "dislike":
        print("Sorry to hear that. We'll note your feedback.")
        handle_dislike(user_query, main_answer)
    else:
        print("No feedback provided, considered as skip.")
def handle_dislike(query, main_answer):
    if query not in blacklist:
        blacklist[query] = []
    if main_answer not in blacklist[query]:
        blacklist[query].append(main_answer)
    save_json_database(blacklist, blacklist_file)

# --- Flask App ---
app = Flask(__name__)

conversation_history = []

@app.route('/')
def home():
    return render_template('index.html', query='', answer='', yatra='', conversation=[])




# Global variable to store the current yatra
current_yatra_name = "common_questions"

@app.route('/ask', methods=['POST'])
def ask():
    global current_yatra_name  # Access the global yatra variable
    user_query = request.form['query']

    # If a new yatra is found in the query, update the current yatra
    detected_yatra = current_yatra(user_query)
    if detected_yatra != "common_questions":  # Only update if a specific yatra is found
        current_yatra_name = detected_yatra

    # Precompute embeddings for performance
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = generate_embeddings(model, database)

    # Step 1: Try Fuzzy Answer
    fuzzy_answer_result, fuzzy_score = fuzzy_answer(user_query, current_yatra_name)

    # Step 2: If no fuzzy match, try AI answer
    ai_answer_result = None
    if not fuzzy_answer_result:
        ai_answer_result = find_best_match(user_query, embeddings, model)

    # Step 3: Choose main answer (AI or fuzzy)
    main_answer = fuzzy_answer_result if fuzzy_answer_result else ai_answer_result

    # Step 4: If no answer, ask the user for an answer
    if not main_answer:
        main_answer = user_answer(user_query, current_yatra_name)

    # Append the user query and answer to the conversation history
    conversation_history.append({"question": user_query, "answer": main_answer})

    return render_template('index.html', user_query=user_query, main_answer=main_answer, current_yatra_name=current_yatra_name, conversation=conversation_history)



@app.route('/better-answer', methods=['POST'])
def better_answer():
    user_query = request.form['query']
    better_answer = request.form['better_answer']

    # Save the better answer under the current yatra (current_yatra_name)
    save_to_database(user_query, better_answer, current_yatra_name)

    # Append the better answer to the conversation history
    conversation_history.append({"question": user_query, "answer": better_answer})

    return ('', 204)  # Respond with no content to avoid page reload


@app.route('/feedback', methods=['POST'])
def feedback():
    user_query = request.form['query']
    main_answer = request.form['answer']
    yatra = request.form['yatra']
    feedback = request.form['feedback']
    handle_feedback(user_query, main_answer, yatra, feedback)

    # Append feedback acknowledgment to the conversation history
    conversation_history.append({"question": user_query, "answer": "Thank you for your feedback!"})

    return ('', 204)  # Respond with no content to avoid page reload


# --- Start Flask App with ngrok ---
if __name__ == "__main__":
    # Open an ngrok tunnel to the Flask app
    public_url = ngrok.connect(5000)
    print(" * ngrok tunnel: ", public_url)
    app.run(port=5000)
