# Programmatic Definition

While YAML is convenient for version control and sharing, you can also define your classification schema programmatically in Python code. This page shows examples of creating `ClassificationDefinition` objects directly, with increasing levels of complexity.

## Basic Example: Direct Object Creation

At its simplest, you can create a classification schema with just labels:

```python
from instructor_classify.schema import ClassificationDefinition, LabelDefinition
from instructor_classify.classify import Classifier
import instructor
from openai import OpenAI

# Create label definitions
spam_label = LabelDefinition(
    label="spam",
    description="Unsolicited messages trying to sell products or scam recipients"
)

not_spam_label = LabelDefinition(
    label="not_spam",
    description="Legitimate messages with relevant content for the recipient"
)

# Create classification definition
classification_def = ClassificationDefinition(
    system_message="You are an expert spam detection system.",
    label_definitions=[spam_label, not_spam_label]
)

# Use the classification definition with a classifier
client = instructor.from_openai(OpenAI())
classifier = (
    Classifier(classification_def)
    .with_client(client)
    .with_model("gpt-3.5-turbo")
)

# Make a prediction
result = classifier.predict("CLICK HERE NOW to claim your FREE iPhone 15 Pro Max! You've been selected!")
print(f"Classification: {result.label}")  # Should output "spam"
```

## Intermediate: Adding Few-Shot Examples

Adding examples helps improve model performance by demonstrating what belongs in each category:

```python
from instructor_classify.schema import ClassificationDefinition, LabelDefinition, Examples
from instructor_classify.classify import Classifier
import instructor
from openai import OpenAI

# Create label definitions with examples
spam_label = LabelDefinition(
    label="spam",
    description="Unsolicited messages trying to sell products or scam recipients",
    examples=Examples(
        examples_positive=[
            "CONGRATULATIONS! You've been selected for our exclusive prize! Click here to claim now!",
            "URGENT: Your account has been compromised. Reply with your password to secure your account.",
            "Make $5000 weekly working from home! Limited time offer, sign up now!"
        ],
        examples_negative=[
            "Could you send me the report by tomorrow?",
            "Your flight to London has been confirmed for May 15th.",
            "Thanks for your inquiry. Our support team will contact you within 24 hours."
        ]
    )
)

not_spam_label = LabelDefinition(
    label="not_spam", 
    description="Legitimate messages with relevant content for the recipient",
    examples=Examples(
        examples_positive=[
            "Your package has been delivered to the front door.",
            "I'm following up on our meeting from yesterday. Are you available tomorrow?",
            "Your monthly statement is now available. Please log in to view."
        ],
        examples_negative=[
            "YOU'VE WON THE LOTTERY! Claim your $1,000,000 prize now!",
            "Hot singles in your area want to meet you tonight!",
            "Your computer has a virus! Call this number immediately!"
        ]
    )
)

# Create classification definition
classification_def = ClassificationDefinition(
    system_message="You are an expert spam detection system. Analyze the given text and determine if it is spam or not.",
    label_definitions=[spam_label, not_spam_label]
)

# Use the classification definition with a classifier
client = instructor.from_openai(OpenAI())
classifier = (
    Classifier(classification_def)
    .with_client(client)
    .with_model("gpt-3.5-turbo")
)

# Make predictions
messages = [
    "URGENT: Your iCloud account has been locked. Click here to verify your identity.",
    "Hi Sarah, here are the documents you requested for the Johnson project.",
    "FREE VACATION ALERT: You've been selected for an all-expenses-paid trip to Hawaii!",
    "Your order #12345 has been shipped and will arrive on Tuesday."
]

for message in messages:
    result = classifier.predict(message)
    print(f"Message: '{message}'")
    print(f"Classification: {result.label}")
    print("---")
```

## Advanced: Building a Complex Schema Programmatically

For more sophisticated classifiers, you might want to generate the schema programmatically:

```python
from instructor_classify.schema import ClassificationDefinition, LabelDefinition, Examples
from instructor_classify.classify import Classifier
import instructor
from openai import OpenAI
import json

# Define spam categories and examples
spam_categories = {
    "phishing": {
        "description": "Messages attempting to trick users into revealing sensitive information",
        "examples_positive": [
            "Dear customer, we've detected suspicious activity on your account. Click here to verify your identity.",
            "Your PayPal account has been limited. Login now to restore full access: [suspicious link]",
            "Attention: Your tax refund is pending. Please confirm your banking details within 24 hours."
        ],
        "examples_negative": [
            "Please reset your password by clicking the official link in this email from our domain.",
            "We've noticed unusual activity and have temporarily frozen your account. Please call our official number."
        ]
    },
    "promotional": {
        "description": "Unsolicited marketing messages promoting products or services",
        "examples_positive": [
            "LIMITED TIME OFFER: 80% OFF ALL PRODUCTS! SHOP NOW!",
            "You've been selected for our EXCLUSIVE VIP discount! Click now before it expires!",
            "Buy one get THREE free! This week only at SuperStore!"
        ],
        "examples_negative": [
            "As a subscriber to our newsletter, here's your monthly discount code.",
            "Thank you for your purchase. Here are other products you might enjoy."
        ]
    },
    "scam": {
        "description": "Fraudulent messages intended to deceive recipients for financial gain",
        "examples_positive": [
            "I am Prince Abioye from Nigeria. I need your help to transfer $5,000,000 USD.",
            "CONGRATULATIONS! You've WON the Microsoft lottery! Send $100 processing fee to claim $1,000,000!",
            "Your computer has been infected with a virus! Call this number immediately to remove it."
        ],
        "examples_negative": [
            "You've won our legitimate contest. No purchase necessary to claim your prize.",
            "We've detected suspicious activity on your account. Please call the number on the back of your card."
        ]
    }
}

# Define legitimate categories
not_spam_categories = {
    "personal": {
        "description": "Genuine personal communications between individuals",
        "examples_positive": [
            "Hey, are we still meeting for lunch tomorrow at 12?",
            "I've attached the photos from our trip. Hope you like them!",
            "Just checking in to see how you're doing. Call me when you get a chance."
        ]
    },
    "business": {
        "description": "Legitimate business communications",
        "examples_positive": [
            "Your invoice #1234 is attached. Payment is due within 30 days.",
            "The meeting has been rescheduled to Tuesday at 2pm in Conference Room A.",
            "Thank you for your application. We'd like to invite you for an interview."
        ]
    },
    "transactional": {
        "description": "Legitimate automated notifications related to user actions or accounts",
        "examples_positive": [
            "Your payment of $50.00 was processed successfully.",
            "Your order has shipped. Tracking number: TRK123456789",
            "Your password was changed at 3:45 PM. If this wasn't you, please contact support."
        ]
    }
}

# Build label definitions programmatically
label_definitions = []

# Add spam categories with subcategories
for category, details in spam_categories.items():
    label = f"spam_{category}"
    examples = Examples(
        examples_positive=details["examples_positive"],
        examples_negative=details.get("examples_negative", [])
    )
    label_def = LabelDefinition(
        label=label,
        description=details["description"],
        examples=examples
    )
    label_definitions.append(label_def)

# Add legitimate categories
for category, details in not_spam_categories.items():
    label = f"legitimate_{category}"
    examples = Examples(
        examples_positive=details["examples_positive"],
        examples_negative=details.get("examples_negative", [])
    )
    label_def = LabelDefinition(
        label=label,
        description=details["description"],
        examples=examples
    )
    label_definitions.append(label_def)

# Create the classification definition
system_message = """
You are an advanced spam detection system with the ability to categorize different types of messages.
Analyze the given text and determine both whether it is spam and what specific type of message it is.
"""

classification_def = ClassificationDefinition(
    system_message=system_message,
    label_definitions=label_definitions
)

# Use the classifier
client = instructor.from_openai(OpenAI())
classifier = (
    Classifier(classification_def)
    .with_client(client)
    .with_model("gpt-4o-mini")  # Using a more capable model for fine-grained classification
)

# Test messages
test_messages = [
    "Dear valued customer, we have detected suspicious activity on your bank account. Please verify your identity by clicking this link: http://suspiciouslink.com",
    "Hey John, just wanted to confirm our meeting tomorrow at 2pm. Let me know if that still works for you. Cheers, Maria",
    "CONGRATULATIONS! You've been selected to receive a FREE iPhone 15! Click here to claim your prize now! Limited time offer!",
    "Your Amazon order #AB12345 has been shipped and will arrive on Thursday, April 20. Track your package here: [legitimate tracking link]",
    "I am a wealthy businessman who needs your help. I can transfer $5,000,000 to your account if you pay a small fee of $1,000 first."
]

for message in test_messages:
    result = classifier.predict(message)
    print(f"Message: '{message[:50]}...'")
    print(f"Classification: {result.label}")
    print("---")

# Optional: Save the definition to YAML for version control
import yaml

def export_to_yaml(classification_def, file_path):
    """Convert a ClassificationDefinition to YAML and save it to a file."""
    # Convert the Pydantic model to a dictionary
    data = classification_def.model_dump()
    
    # Save to YAML
    with open(file_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)
    
    print(f"Saved classification definition to {file_path}")

# Export the programmatically created definition
export_to_yaml(classification_def, "advanced_spam_classifier.yaml")
```

## Hybrid Approach: Loading from Dict or JSON

You can also create a classification definition from a dictionary or JSON:

```python
from instructor_classify.schema import ClassificationDefinition
import json

# Define your schema as a Python dictionary
schema_dict = {
    "system_message": "You are an expert spam detection system.",
    "label_definitions": [
        {
            "label": "spam",
            "description": "Unsolicited messages trying to sell products or scam recipients",
            "examples": {
                "examples_positive": [
                    "URGENT: You have won $1,000,000 in the lottery!",
                    "Click here for a FREE iPhone!"
                ],
                "examples_negative": [
                    "Your meeting is scheduled for tomorrow at 2pm.",
                    "Thanks for your purchase, here's your receipt."
                ]
            }
        },
        {
            "label": "not_spam",
            "description": "Legitimate messages with relevant content for the recipient",
            "examples": {
                "examples_positive": [
                    "Your Amazon order has shipped.",
                    "Meeting notes from yesterday's call."
                ],
                "examples_negative": [
                    "MAKE MONEY FAST! Work from home!",
                    "Your account has been compromised! Click here!"
                ]
            }
        }
    ]
}

# Create from dictionary
classification_def = ClassificationDefinition(**schema_dict)

# Alternative: Load from JSON file
def load_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return ClassificationDefinition(**data)

# If you have a JSON file
# classification_def = load_from_json("spam_classifier.json")

# Use with classifier as before
from instructor_classify.classify import Classifier
import instructor
from openai import OpenAI

client = instructor.from_openai(OpenAI())
classifier = (
    Classifier(classification_def)
    .with_client(client)
    .with_model("gpt-3.5-turbo")
)

# Test
result = classifier.predict("CONGRATULATIONS! You've won a free cruise! Call now to claim!")
print(f"Classification: {result.label}")
```

## Benefits of Programmatic Definition

Programmatic definition offers several advantages:

1. **Dynamic Creation**: Generate schemas based on user input or database content
2. **Validation**: Perform additional validation or transformation before creating the schema
3. **Integration**: Pull examples from existing datasets or databases
4. **Flexibility**: Modify the schema at runtime based on performance or feedback
5. **Code Control**: Keep everything in Python rather than separate YAML files
6. **Testing**: Easier to create test fixtures and variations

However, for production use cases, it's still recommended to export your programmatically created definitions to YAML files for version control and easy sharing with team members.