from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import os
from fastapi.staticfiles import StaticFiles
import base64

app = FastAPI(title="Image Generator API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Model for the request body
class ImageRequest(BaseModel):
    prompt: str
    aspect_ratio: str = "1:1"  # Default to square

# DeepInfra API endpoint
DEEPINFRA_API_URL = "https://api.deepinfra.com/v1/inference/stabilityai/sd3.5-medium"
DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY", "")  # Set your API key as an environment variable

@app.post("/generate-image")
async def generate_image(request: ImageRequest):
    if not DEEPINFRA_API_KEY:
        raise HTTPException(status_code=500, detail="API key not configured")
    
    headers = {
        "Authorization": f"Bearer {DEEPINFRA_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "input": {
            "prompt": request.prompt,
            "aspect_ratio": request.aspect_ratio,
            "num_inference_steps": 30  # Adjust for speed vs. quality
        }
    }
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                DEEPINFRA_API_URL,
                json=payload,
                headers=headers
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Error from image model: {response.text}"
                )
            
            result = response.json()
            # DeepInfra returns base64 encoded images
            return {"image": result["output"][0]}
            
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Request to image model timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating image: {str(e)}")

# Serve the HTML frontend
@app.get("/")
async def serve_frontend():
    with open("static/index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)

# Import this after the route definition to avoid circular imports
from fastapi.responses import HTMLResponse

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)