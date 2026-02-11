from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field

class ScriptSegment(BaseModel):
    text_es: str = Field(description="The narrative text for this segment in Spanish")
    image_query: str = Field(description="English search query for the image")
    image_prompt: str = Field(description="English generative prompt for the image")
    note: str = Field(default="", description="Director's note or context")
    image_selection: Optional[Dict[str, Any]] = None

class TimelineItem(BaseModel):
    prompt: str
    start: float
    end: float

class VideoPlan(BaseModel):
    brief: str
    target_seconds: int = 60
    title_es: str = "Video personalizado"
    youtube_title_es: str = "Video personalizado"
    script_es: Optional[str] = None
    segments: List[ScriptSegment] = Field(default_factory=list)
    
    # State tracking
    seleccionar_imagenes: bool = False
    render_done: bool = False
    topic_finalized: bool = False
    topic_finalized_at: Optional[str] = None
    
    # Source info
    topic_source: Optional[str] = None
    topic_file: Optional[str] = None
    
    # Extra data for rendering
    texto_en: Optional[str] = None # Legacy?
    texto_es: Optional[str] = None # Full text
    prompts: List[str] = Field(default_factory=list) # List of image prompts
    timeline: List[TimelineItem] = Field(default_factory=list)
    dur_est: float = 0.0
    ids: List[str] = Field(default_factory=list)
    usar_video_base: bool = False

    class Config:
        extra = "allow" # Allow extra fields for backward compatibility or future expansion
