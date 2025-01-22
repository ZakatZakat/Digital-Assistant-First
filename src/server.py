from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from generation import get_inference


app = FastAPI()

class UserCreate(BaseModel):
    message: str
    user_id: int

@app.post("/message/")
async def message(request: UserCreate):
    return {
        "user_id": request.user_id,
        "message": get_inference(
            query=request.message,
            model="llama3.2",
            return_answer=True,  # ответ будет возвращён как поле message
        ),
    }

# запустить локально llama app
if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8080, log_level="info")