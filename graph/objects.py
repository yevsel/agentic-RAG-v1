from pydantic import BaseModel,Field

class GradeDocumentsObject(BaseModel):
    """We want to check if our retrieved docs is relevant to our question"""
    
    binary_score: str = Field("Documents are relevant to the question 'yes' or 'no'")
    