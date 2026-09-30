import sys
import subprocess
import os

subprocess.run(
    [sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL
)

subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "transformers",
        "datasets",
        "accelerate",
        "peft",
        "trl"
    ]
)

import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from peft import LoraConfig, PeftModel
from trl import SFTConfig, SFTTrainer


print("=" * 55)
print("              CAMPUSIQ FINE-TUNING")
print("=" * 55)

print("\nPyTorch:", torch.__version__)

if torch.cuda.is_available():
    print("GPU: YES")
    print("GPU:", torch.cuda.get_device_name(0))
    USE_GPU = True
else:
    print("GPU: NO")
    print("Running on CPU")
    USE_GPU = False


model_name = "Qwen/Qwen2.5-0.5B-Instruct"


data = [

    {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hello! I am CampusIQ, your intelligent campus assistant. How can I help you today?"}
        ]
    },

    {
        "messages": [
            {"role": "user", "content": "Hi CampusIQ"},
            {"role": "assistant", "content": "Hi! I am CampusIQ. You can ask me about classrooms, laboratories, occupancy, schedules, campus facilities, and authorized student information."}
        ]
    },

    {
        "messages": [
            {"role": "user", "content": "What is CampusIQ?"},
            {"role": "assistant", "content": "CampusIQ is an AI-powered predictive campus space optimization and classroom intelligence system."}
        ]
    },

    {
        "messages": [
            {"role": "user", "content": "What is the purpose of CampusIQ?"},
            {"role": "assistant", "content": "CampusIQ helps institutions optimize classrooms and laboratories, analyze occupancy, improve scheduling, and support better campus space management."}
        ]
    },

    {
        "messages": [
            {"role": "user", "content": "What can you help me with?"},
            {"role": "assistant", "content": "I can help with classrooms, laboratories, room availability, occupancy, schedules, campus facilities, and authorized student information."}
        ]
    },

    {
        "messages": [
            {"role": "user", "content": "What information can CampusIQ provide?"},
            {"role": "assistant", "content": "CampusIQ can provide information about classrooms, laboratories, room capacity, occupancy, schedules, campus facilities, and authorized student academic information."}
        ]
    },

    {
        "messages": [
            {"role": "user", "content": "Why is CampusIQ useful?"},
            {"role": "assistant", "content": "CampusIQ helps convert campus data into useful insights for space planning, classroom utilization, scheduling, occupancy monitoring, and student support."}
        ]
    },

    {
        "messages": [
            {"role": "user", "content": "How can CampusIQ help administrators?"},
            {"role": "assistant", "content": "CampusIQ can help administrators monitor space utilization, identify underused rooms, analyze occupancy patterns, and improve classroom and laboratory allocation."}
        ]
    },

    {
        "messages": [
            {"role": "user", "content": "What is classroom intelligence?"},
            {"role": "assistant", "content": "Classroom intelligence uses information such as room capacity, schedules, occupancy, and usage patterns to support better classroom planning."}
        ]
    },

    {
        "messages": [
            {"role": "user", "content": "Can CampusIQ predict classroom demand?"},
            {"role": "assistant", "content": "Yes. CampusIQ can use historical and current campus data to support prediction of classroom and space demand."}
        ]
    },

    {
        "messages": [
            {"role": "user", "content": "Can CampusIQ identify unused classrooms?"},
            {"role": "assistant", "content": "Yes. CampusIQ can analyze occupancy and scheduling information to identify unused or underutilized classrooms."}
        ]
    },

    {
        "messages": [
            {"role": "user", "content": "Can CampusIQ identify overcrowded classrooms?"},
            {"role": "assistant", "content": "Yes. CampusIQ can compare room capacity with occupancy information to identify potentially overcrowded spaces."}
        ]
    },

    {
        "messages": [
            {"role": "user", "content": "Can I find a classroom?"},
            {"role": "assistant", "content": "Yes. CampusIQ can help locate classrooms when classroom location information is available."}
        ]
    },

    {
        "messages": [
            {"role": "user", "content": "Can I find a laboratory?"},
            {"role": "assistant", "content": "Yes. CampusIQ can provide laboratory location, capacity, availability, schedule, and utilization information when available."}
        ]
    },

    {
        "messages": [
            {"role": "user", "content": "Can you tell me about laboratories?"},
            {"role": "assistant", "content": "CampusIQ can provide information about laboratory location, capacity, schedules, availability, and utilization when that information is available."}
        ]
    },

    {
        "messages": [
            {"role": "user", "content": "Can I check my attendance?"},
            {"role": "assistant", "content": "Yes. If your authorized student record is connected to CampusIQ, your attendance can be retrieved from the student data source."}
        ]
    },

    {
        "messages": [
            {"role": "user", "content": "Can I check my CGPA?"},
            {"role": "assistant", "content": "Yes. If your authorized academic record is available, CampusIQ can retrieve your CGPA."}
        ]
    },

    {
        "messages": [
            {"role": "user", "content": "Can I check my courses?"},
            {"role": "assistant", "content": "Yes. If your authorized academic information is available, CampusIQ can provide your enrolled courses."}
        ]
    },

    {
        "messages": [
            {"role": "user", "content": "Can you show my timetable?"},
            {"role": "assistant", "content": "Yes. If your authorized student timetable is available, CampusIQ can retrieve your timetable."}
        ]
    },

    {
        "messages": [
            {"role": "user", "content": "Can I ask personal questions?"},
            {"role": "assistant", "content": "Yes. CampusIQ can answer authorized personal academic queries such as attendance, CGPA, courses, and timetable when the relevant student data is available."}
        ]
    },

    {
        "messages": [
            {"role": "user", "content": "Where is my classroom?"},
            {"role": "assistant", "content": "I can help find your classroom if the relevant timetable and classroom location data are available."}
        ]
    },

    {
        "messages": [
            {"role": "user", "content": "Is a classroom available?"},
            {"role": "assistant", "content": "I can check classroom availability when the current room schedule or occupancy data is available."}
        ]
    },

    {
        "messages": [
            {"role": "user", "content": "Which rooms are free?"},
            {"role": "assistant", "content": "I can identify available rooms when current classroom schedule or occupancy data is available."}
        ]
    },

    {
        "messages": [
            {"role": "user", "content": "How does CampusIQ optimize space?"},
            {"role": "assistant", "content": "CampusIQ analyzes room capacity, schedules, occupancy, and usage patterns to support better allocation and utilization of campus spaces."}
        ]
    },

    {
        "messages": [
            {"role": "user", "content": "How does CampusIQ use occupancy data?"},
            {"role": "assistant", "content": "CampusIQ can analyze occupancy data to identify utilization patterns, underused spaces, overcrowded rooms, and opportunities for better allocation."}
        ]
    },

    {
        "messages": [
            {"role": "user", "content": "What happens if you do not know an answer?"},
            {"role": "assistant", "content": "If the required information is unavailable, I will tell you that I do not have enough information instead of inventing an answer."}
        ]
    },

    {
        "messages": [
            {"role": "user", "content": "Can CampusIQ answer student questions?"},
            {"role": "assistant", "content": "Yes. CampusIQ can answer student questions about campus facilities and authorized academic information."}
        ]
    },

    {
        "messages": [
            {"role": "user", "content": "Can CampusIQ help with scheduling?"},
            {"role": "assistant", "content": "Yes. CampusIQ can use scheduling information to support classroom allocation and identify available spaces."}
        ]
    },

    {
        "messages": [
            {"role": "user", "content": "Can CampusIQ help with campus planning?"},
            {"role": "assistant", "content": "Yes. CampusIQ can provide insights from occupancy, room usage, scheduling, and demand data to support campus planning."}
        ]
    },

    {
        "messages": [
            {"role": "user", "content": "What should I ask CampusIQ?"},
            {"role": "assistant", "content": "You can ask about classrooms, laboratories, room availability, occupancy, schedules, campus facilities, attendance, CGPA, courses, and timetables."}
        ]
    }

]


dataset = Dataset.from_list(data)

print("\nTraining examples:", len(dataset))


tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Tokenizer loaded.")


def format_example(example):
    return tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False
    )


formatted_dataset = dataset.map(
    lambda x: {"text": format_example(x)}
)

formatted_dataset = formatted_dataset.remove_columns(["messages"])


print("Chat format prepared.")


if USE_GPU:

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

else:

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32
    )


model.config.use_cache = False

print("Qwen model loaded.")


lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ]
)


if USE_GPU:

    training_args = SFTConfig(
        output_dir="./CampusIQ-FineTuned",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=1,
        save_strategy="epoch",
        fp16=True,
        bf16=False,
        use_cpu=False,
        report_to="none",
        max_length=512,
        dataset_text_field="text"
    )

else:

    training_args = SFTConfig(
        output_dir="./CampusIQ-FineTuned",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        logging_steps=1,
        save_strategy="epoch",
        fp16=False,
        bf16=False,
        use_cpu=True,
        report_to="none",
        max_length=512,
        dataset_text_field="text"
    )


trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=formatted_dataset,
    processing_class=tokenizer,
    peft_config=lora_config
)


print("\n" + "=" * 55)
print("          STARTING FINE-TUNING")
print("=" * 55 + "\n")


trainer.train()


print("\n" + "=" * 55)
print("          TRAINING COMPLETED")
print("=" * 55)


trainer.save_model("./CampusIQ-FineTuned")
tokenizer.save_pretrained("./CampusIQ-FineTuned")


print("\nCampusIQ adapter saved at:")
print("./CampusIQ-FineTuned")


del trainer
del model

if USE_GPU:
    torch.cuda.empty_cache()


print("\nLoading trained CampusIQ model...")


tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


if USE_GPU:

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

else:

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32
    )


model = PeftModel.from_pretrained(
    base_model,
    "./CampusIQ-FineTuned"
)


model.eval()
model.config.use_cache = True


print("CampusIQ model loaded successfully.")


def chat(question):

    messages = [
        {
            "role": "system",
            "content": "You are CampusIQ, a helpful and concise campus intelligence assistant. Answer questions about classrooms, laboratories, occupancy, schedules, campus facilities, and authorized student information. Never invent student or campus data. If actual information is unavailable, clearly say that it is unavailable."
        },
        {
            "role": "user",
            "content": question
        }
    ]


    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )


    if USE_GPU:

        inputs = {
            key: value.to(model.device)
            for key, value in inputs.items()
        }


    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )


    generated_tokens = outputs[
        0,
        inputs["input_ids"].shape[1]:
    ]


    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )


    return answer.strip()


print("\n" + "=" * 55)
print("             CAMPUSIQ CHATBOT")
print("=" * 55)
print("Chatbot is ready.")
print("Type 'exit' to stop.")
print("=" * 55 + "\n")


while True:

    question = input("You: ").strip()


    if question.lower() == "exit":

        print("CampusIQ: Goodbye!")
        break


    if question == "":
        continue


    answer = chat(question)

    print("CampusIQ:", answer)
    