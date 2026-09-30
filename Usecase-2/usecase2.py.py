import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline

student_data = """
Student Name: Rahul
Student ID: VT1001
Department: Computer Science and Engineering
Year: 3rd Year
CGPA: 8.7
Attendance: 92%
Email: rahul@example.com

Courses:
Data Structures and Algorithms
Database Management Systems
Artificial Intelligence
Computer Networks

Monday:
DSA - 9:00 AM
DBMS - 11:00 AM
AI - 2:00 PM

Tuesday:
Computer Networks - 10:00 AM
DSA - 2:00 PM

Wednesday:
AI - 9:00 AM
DBMS - 1:00 PM

Thursday:
DSA - 10:00 AM
Computer Networks - 2:00 PM

Friday:
DBMS - 9:00 AM
AI - 11:00 AM
"""

campus_data = """
CampusIQ is an AI-powered predictive campus space optimization
and classroom intelligence system.

CampusIQ manages classrooms, laboratories, buildings,
departments, timetables, occupancy and campus spaces.

Block A Room 101 has a capacity of 60 students.

Block A Room 102 has a capacity of 40 students.

Block B Room 201 has a capacity of 80 students.

CampusIQ analyzes classroom utilization and identifies
overcrowded and underutilized classrooms.

CampusIQ can predict future classroom space requirements
using historical occupancy, student strength and timetable data.

Students can use the CampusIQ chatbot to ask questions about
their attendance, courses, timetable and campus facilities.
"""

documents = [
    student_data,
    campus_data
]

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

embeddings = embedding_model.encode(
    documents,
    convert_to_numpy=True
)

vector_database = faiss.IndexFlatL2(
    embeddings.shape[1]
)

vector_database.add(
    embeddings.astype("float32")
)

llm = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct",
    max_new_tokens=150,
    temperature=0.2,
    do_sample=True
)

def retrieve(question):

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True
    )

    distances, indices = vector_database.search(
        question_embedding.astype("float32"),
        2
    )

    context = []

    for i in indices[0]:
        context.append(documents[i])

    return "\n\n".join(context)

def ask_campusiq(question):

    context = retrieve(question)

    prompt = f"""
You are CampusIQ, an AI assistant for a college.

Answer the user's question using ONLY the information
provided in the context.

Do not invent information.

Context:
{context}

Question:
{question}

Answer:
"""

    result = llm(prompt)

    answer = result[0]["generated_text"]

    if "Answer:" in answer:
        answer = answer.split("Answer:")[-1].strip()

    return answer

print("=" * 55)
print("           CAMPUSIQ AI CHATBOT")
print("=" * 55)
print("RAG + LLM chatbot is ready!")
print("Ask your questions directly.")
print("Type 'exit' to stop.")
print("=" * 55)

while True:

    question = input("\nYou: ")

    if question.lower().strip() == "exit":
        print("CampusIQ: Goodbye!")
        break

    answer = ask_campusiq(question)

    print("\nCampusIQ:", answer)
