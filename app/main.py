from dotenv import load_dotenv
import os
from agenthandler import run_agent
from fastapi import FastAPI
from pydantic import BaseModel,Field
from dotenv import load_dotenv
import os
import uvicorn
from typing import Optional
from agenthandler import get_conversation_history

app = FastAPI(title='Simple FastAPI App',version='1.0.0')

@app.get("/") #endpoint is the root which is the forward slash
def root():
    return {'Message':'Welcome to my FastAPI Application'}

class userinputmodel(BaseModel):
    user_input: str = Field(...,example="Explain the Nigerian tax policy on VAT.")
    session_id: str = Field(...,example="session_12345")

#
@app.post("/invoke_agent")
async def invoke_agent(payload: userinputmodel):
    
    answer=run_agent(user_input=payload.user_input,thread_id=payload.session_id)
    return {"answer":answer}

class SessionRequest(BaseModel):
    session_id: str = Field(..., example="session_12345")


@app.post("/conversation_history")
async def conversation_history(payload: SessionRequest):
    history = get_conversation_history(payload.session_id)
    return {
        "session_id": payload.session_id,
        "history": history
    }



# if __name__ == '__main__':
#     uvicorn.run(app,host=os.getenv("host"),port=int(os.getenv("port")))