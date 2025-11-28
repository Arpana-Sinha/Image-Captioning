import torch
import gradio as gr
from transformers import pipeline

answer = pipeline("question-answering", model="deepset/roberta-base-squad2")

def read_file(file_obj):
    try:
        with open(file_obj.name, 'r', encoding='utf-8') as file:
            context = file.read()
            return context
    except Exception as e:
        return f"An error occurred: {e}"

def response(file, question):
    context = read_file(file)
    ans = answer(question = question, context = context)
    return ans["answer"]

gr.close_all()

demo = gr.Interface(fn = response,
                    inputs = [gr.File(label="Upload your file"),
                              gr.Textbox(label="Enter the question",lines = 1)],
                    outputs = [gr.Textbox(label="The answer to your question" )],
                    title = "Document Q & A",
                    description = "THIS APPLICATION WILL BE USED TO ANSER QUESTIONS BASED ON CONTEXT PROVIDED.")

demo.launch()