"""
Parallel processing strategy implementation.

This module provides a thread-based parallel processing strategy for evaluating
examples concurrently.
"""

from typing import Any, Dict, List, TypeVar
from concurrent.futures import ThreadPoolExecutor
import tqdm
from instructor_classify.eval_harness.base import ProcessingStrategy

T = TypeVar('T')  # Classifier type
P = TypeVar('P')  # Input example type
R = TypeVar('R')  # Result type


class ParallelProcessingStrategy(ProcessingStrategy[T, P, R]):
    """
    Thread-based parallel processing strategy implementation.
    
    Processes examples concurrently using a thread pool.
    """
    
    def process_batch(self, classifier: T, examples: List[P], is_multi: bool = False) -> List[R]:
        """
        Process examples in parallel using threads.
        
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
        
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = [
                executor.submit(self._process_single_example, classifier, example, is_multi)
                for example in examples
            ]
            
            for future in tqdm.tqdm(
                futures, total=len(futures), desc=f"Classifying with {self.n_jobs} threads", leave=False
            ):
                results.append(future.result())
        
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