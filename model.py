"""from langchain_community.llms import Ollama

oll = Ollama()
print(oll.invoke("Who is Superman?"))"""

import ollama

res = ollama.chat(model="llama2", messages=[
{
    "role": "user",
    "content": "why is english hard?" 
 }    
 ])

print(res)