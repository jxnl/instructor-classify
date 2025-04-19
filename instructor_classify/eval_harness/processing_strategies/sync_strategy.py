"""
Synchronous processing strategy implementation.

This module provides a sequential processing strategy for evaluating examples
without any parallelism.
"""

from typing import Any, Dict, List, TypeVar, Generic
from rich.progress import Progress
from instructor_classify.eval_harness.base import ProcessingStrategy

T = TypeVar('T')  # Classifier type
P = TypeVar('P')  # Input example type
R = TypeVar('R')  # Result type


class SyncProcessingStrategy(ProcessingStrategy[T, P, R]):
    """
    Sequential processing strategy implementation.
    
    Processes examples one by one without any parallelism.
    """
    
    def process_batch(self, classifier: T, examples: List[P], is_multi: bool = False) -> List[R]:
        """
        Process examples sequentially with a progress bar.
        
        Parameters
        ----------
        classifier : T
            The classifier to use
        examples : List[P]
            The examples to process
        is_multi : bool
            Whether this is multi-label classification
            
        Returns
        -------
        List[R]
            The processing results
        """
        results = []
        
        with Progress() as progress:
            task = progress.add_task(
                f"Classifying sequentially", total=len(examples)
            )
            
            for example in examples:
                result = self._process_single_example(classifier, example, is_multi)
                results.append(result)
                progress.update(task, advance=1)
        
        return results
    
    def _process_single_example(self, classifier: Any, example: Any, is_multi: bool = False) -> Dict[str, Any]:
        """
        Process a single example.
        
        Parameters
        ----------
        classifier : Any
            The classifier to use
        example : Any
            The example to process
        is_multi : bool
            Whether this is multi-label classification
            
        Returns
        -------
        Dict[str, Any]
            The processing result
        """
        if is_multi:
            prediction = classifier.predict_multi(example.text)
            is_correct = set(prediction.labels) == set(example.expected_labels)
            expected = example.expected_labels
            predicted = prediction.labels
        else:
            prediction = classifier.predict(example.text)
            is_correct = prediction.label == example.expected_label
            expected = example.expected_label
            predicted = prediction.label
            
        return {
            "prediction": prediction,
            "is_correct": is_correct,
            "text": example.text,
            "expected": expected,
            "predicted": predicted
        }