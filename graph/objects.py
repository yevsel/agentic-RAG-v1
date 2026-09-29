from pydantic import BaseModel,Field

class GradeDocumentsObject(BaseModel):
    """We want to check if our retrieved docs is relevant to our question"""
    binary_score: str = Field("Documents are relevant to the question 'yes' or 'no'")
    
class HallucinationGraderObject(BaseModel):
    """We want to check if LLM generated answer is grounded in the document by giving it a binary score"""
    binary_score: bool = Field("Answer is grounded in the facts, 'yes' or 'no'")

class AnswerGraderObject(BaseModel):
    """We want to check if LLM generated answer answers our question"""
    binary_score: bool = Field("Answer addresses the question, 'yes' or 'no'")

class IntentClassifierObject(BaseModel):
    """Route the use based on the question to know whether we fetch data or ignore"""
    datasource: str = Field(..., description="Given a user question, route to 'web_search' or 'vectorstore'")