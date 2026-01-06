from dotenv import load_dotenv
import os
from agenthandler import run_agent
from fastapi import FastAPI
from pydantic import BaseModel,Field
from dotenv import load_dotenv
import os
import uvicorn
from typing import Optional

app = FastAPI(title='Simple FastAPI App',version='1.0.0')

@app.get("/") #endpoint is the root which is the forward slash
def root():
    return {'Message':'Welcome to my FastAPI Application'}

class userinputmodel(BaseModel):
    user_input: str = Field(...,example="Explain the Nigerian tax policy on VAT.")

#
@app.post("/invoke_agent")
async def invoke_agent(user_input: userinputmodel):
    
    answer=run_agent(user_input=user_input.user_input)
    return {"answer":answer}




if __name__ == '__main__':
    uvicorn.run(app,host=os.getenv("host"),port=int(os.getenv("port")))