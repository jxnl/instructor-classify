from instructor_classify.classify import Classifier
from instructor_classify.schema import ClassificationDefinition
import instructor
from openai import OpenAI

def main():
    # Initialize OpenAI client
    client = OpenAI()
    
    # Wrap the client with instructor
    instructor_client = instructor.from_openai(client)
    
    # Load our classification definition from the YAML file
    definition = ClassificationDefinition.from_yaml("prompt.yaml")
    
    # Create the classifier
    classifier = (
        Classifier(definition)
        .with_client(instructor_client)
        .with_model("gpt-3.5-turbo")
    )
    
    # Example texts to classify
    texts = [
        "What's the difference between Python 2 and Python 3?",
        "Book a team lunch for next Tuesday at noon.",
        "Can you show me how to implement a REST API in Flask?",
        "When should I use asynchronous programming patterns?",
        "How do I schedule my code to run at specific times?"
    ]
    
    # Classify each text
    print("Classifying texts...")
    results = classifier.batch_predict(texts)
    
    # Print results
    print("\nResults:")
    for text, result in zip(texts, results):
        print(f"\nText: {text}")
        print(f"Classification: {result.label}")
        print("-" * 80)

if __name__ == "__main__":
    main()
