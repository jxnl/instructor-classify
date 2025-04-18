# Examples

This page provides examples of common classification tasks and advanced usage patterns.

## Intent Classification

```python
from instructor_classify import Classifier, ClassificationDefinition
import instructor
from openai import OpenAI

# Intent classification definition
yaml_content = """
system_message: |
  You are an expert intent classifier for a customer service system.
  
label_definitions:
  - label: account_issue
    description: The user has a problem with their account access or settings.
    examples:
      examples_positive:
        - "I can't log into my account"
        - "How do I change my password?"
      examples_negative:
        - "When will my order arrive?"
        - "I want to return this product"
  
  - label: billing_question
    description: The user has a question about billing, charges, or payments.
    examples:
      examples_positive:
        - "Why was I charged twice?"
        - "When will my subscription renew?"
      examples_negative:
        - "How do I track my order?"
        - "The product is broken"
        
  - label: product_support
    description: The user needs help with using a product or has a technical issue.
    examples:
      examples_positive:
        - "My device won't turn on"
        - "How do I reset the software?"
      examples_negative:
        - "I want to cancel my order"
        - "Can I change my shipping address?"
"""

# Save to file
with open("intent_classifier.yaml", "w") as f:
    f.write(yaml_content)

# Load and use
definition = ClassificationDefinition.from_yaml("intent_classifier.yaml")
client = instructor.from_openai(OpenAI())
classifier = Classifier(definition).with_client(client).with_model("gpt-3.5-turbo")

# Test with example queries
queries = [
    "I forgot my password and now I can't get into my account",
    "You charged me twice for my last order",
    "The app keeps crashing whenever I try to upload photos",
    "When will my order arrive?",
]

for query in queries:
    result = classifier.predict(query)
    print(f"Query: '{query}'")
    print(f"Intent: {result.label}")
    print("---")
```

## Multi-Label Classification

```python
from instructor_classify import Classifier, ClassificationDefinition
import instructor
from openai import OpenAI
import yaml

# Content classifier for multiple labels
yaml_content = """
system_message: |
  You are an expert content classifier for a content moderation system.
  Content may belong to multiple categories simultaneously.
  
label_definitions:
  - label: politics
    description: Content related to government, policies, elections, or political figures.
  
  - label: business
    description: Content related to companies, markets, finance, or the economy.
  
  - label: technology
    description: Content related to digital products, software, hardware, or scientific innovation.
  
  - label: entertainment
    description: Content related to movies, music, celebrities, or leisure activities.
    
  - label: sports
    description: Content related to athletics, games, competitions, or sporting events.
"""

with open("content_classifier.yaml", "w") as f:
    f.write(yaml_content)

# Load and use
definition = ClassificationDefinition.from_yaml("content_classifier.yaml")
client = instructor.from_openai(OpenAI())
classifier = Classifier(definition).with_client(client).with_model("gpt-4o-mini")

# Example articles to classify
articles = [
    "Apple announces record profits as new iPhone sales exceed expectations.",
    "Senate passes new tech regulation bill aimed at social media companies.",
    "Hollywood actors strike over streaming revenue and AI concerns.",
    "Tech giant releases new AI tools for small business accounting.",
]

for article in articles:
    result = classifier.predict_multi(article)
    print(f"Article: '{article}'")
    print(f"Categories: {result.labels}")
    print("---")
```

## Working with Raw Completions

```python
from instructor_classify.classify import Classifier
from instructor_classify.schema import ClassificationDefinition
import instructor
from openai import OpenAI

# Load your classification definition
definition = ClassificationDefinition.from_yaml("prompt.yaml")
client = instructor.from_openai(OpenAI())
classifier = Classifier(definition).with_client(client).with_model("gpt-3.5-turbo")

# Get both the structured output and the raw completion
text = "What is the capital of France?"
result, completion = classifier.predict_with_completion(text)

print(f"Structured result: {result.label}")
print(f"Confidence: {completion.choices[0].finish_reason}")
print(f"Model: {completion.model}")
print(f"Usage: {completion.usage.total_tokens} tokens")

# You can analyze the completion for additional insights
response_text = completion.choices[0].message.content
print(f"Raw response: {response_text}")
```

## Asynchronous Batch Processing

```python
from instructor_classify.classify import AsyncClassifier
from instructor_classify.schema import ClassificationDefinition
import instructor
from openai import AsyncOpenAI
import asyncio
import time

async def classify_large_dataset():
    # Load your classification definition
    definition = ClassificationDefinition.from_yaml("prompt.yaml")
    client = instructor.from_openai_aclient(AsyncOpenAI())
    classifier = AsyncClassifier(definition).with_client(client).with_model("gpt-3.5-turbo")
    
    # Sample large dataset
    dataset = [
        "How do I reset my password?",
        "I'd like to cancel my subscription",
        "What are your business hours?",
        "The product I received is damaged",
        "Do you ship internationally?",
        # ... imagine hundreds more items
    ]
    
    # Process in batches with concurrency control
    start_time = time.time()
    results = await classifier.batch_predict(dataset, n_jobs=10)
    end_time = time.time()
    
    # Analyze results
    label_counts = {}
    for result in results:
        label_counts[result.label] = label_counts.get(result.label, 0) + 1
    
    print(f"Processed {len(dataset)} items in {end_time - start_time:.2f} seconds")
    print(f"Distribution: {label_counts}")
    
    return results

# Run the async function
if __name__ == "__main__":
    asyncio.run(classify_large_dataset())
```

## Using Different LLM Providers

```python
from instructor_classify.classify import Classifier
from instructor_classify.schema import ClassificationDefinition
import instructor

# Define a function to get results from different providers
def compare_providers(text, definition):
    results = {}
    
    # OpenAI
    try:
        from openai import OpenAI
        openai_client = instructor.from_openai(OpenAI())
        openai_classifier = (
            Classifier(definition)
            .with_client(openai_client)
            .with_model("gpt-3.5-turbo")
        )
        results["openai"] = openai_classifier.predict(text).label
    except Exception as e:
        results["openai"] = f"Error: {str(e)}"
    
    # Anthropic
    try:
        from anthropic import Anthropic
        anthropic_client = instructor.from_anthropic(Anthropic())
        anthropic_classifier = (
            Classifier(definition)
            .with_client(anthropic_client)
            .with_model("claude-3-haiku-20240307")
        )
        results["anthropic"] = anthropic_classifier.predict(text).label
    except Exception as e:
        results["anthropic"] = f"Error: {str(e)}"
    
    # Google
    try:
        import google.generativeai as genai
        google_client = instructor.from_gemini(genai)
        google_classifier = (
            Classifier(definition)
            .with_client(google_client)
            .with_model("gemini-pro")
        )
        results["google"] = google_classifier.predict(text).label
    except Exception as e:
        results["google"] = f"Error: {str(e)}"
    
    return results

# Example usage
definition = ClassificationDefinition.from_yaml("prompt.yaml")
text = "What is the best way to learn programming?"

results = compare_providers(text, definition)
for provider, label in results.items():
    print(f"{provider}: {label}")
```

## Integration with Web Frameworks

### FastAPI Example

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from instructor_classify.classify import Classifier
from instructor_classify.schema import ClassificationDefinition
import instructor
from openai import OpenAI
import os

app = FastAPI(title="Text Classification API")

# Load classifier on startup
@app.on_event("startup")
async def startup_event():
    global classifier
    
    # Load classification definition
    definition = ClassificationDefinition.from_yaml("prompt.yaml")
    
    # Create classifier
    client = instructor.from_openai(OpenAI(api_key=os.environ.get("OPENAI_API_KEY")))
    classifier = (
        Classifier(definition)
        .with_client(client)
        .with_model("gpt-3.5-turbo")
    )

class ClassificationRequest(BaseModel):
    text: str

class ClassificationResponse(BaseModel):
    text: str
    label: str

@app.post("/classify", response_model=ClassificationResponse)
async def classify_text(request: ClassificationRequest):
    try:
        result = classifier.predict(request.text)
        return ClassificationResponse(
            text=request.text,
            label=result.label
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run with: uvicorn app:app --reload
```

## Running Evaluations From Code

```python
from instructor_classify.eval_harness.unified_eval import UnifiedEvaluator
import yaml

# Create configuration programmatically
eval_config = {
    "models": ["gpt-3.5-turbo", "gpt-4o-mini"],
    "eval_sets": ["datasets/evalset_single.yaml"],
    "bootstrap_samples": 1000,
    "confidence_level": 0.95
}

# Save to file
with open("dynamic_eval_config.yaml", "w") as f:
    yaml.dump(eval_config, f)

# Run evaluation
evaluator = UnifiedEvaluator("dynamic_eval_config.yaml")
evaluator.prepare()
results = evaluator.run()

# Access evaluation results programmatically
for model_name, model_results in results.items():
    print(f"Model: {model_name}")
    for eval_set_name, metrics in model_results.items():
        print(f"  Eval Set: {eval_set_name}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1 Score: {metrics['macro_f1']:.4f}")
```