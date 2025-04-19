"""
Asynchronous processing strategy implementation.

This module provides an asyncio-based processing strategy for evaluating
examples concurrently using async/await.
"""

import asyncio
from typing import Any, Dict, List, TypeVar
from tqdm.asyncio import tqdm
from instructor_classify.eval_harness.base import ProcessingStrategy

T = TypeVar('T')  # Classifier type
P = TypeVar('P')  # Input example type
R = TypeVar('R')  # Result type


class AsyncProcessingStrategy(ProcessingStrategy[T, P, R]):
    """
    Asyncio-based processing strategy implementation.
    
    Processes examples concurrently using asyncio.
    """
    
    def process_batch(self, classifier: T, examples: List[P], is_multi: bool = False) -> List[R]:
        """
        Process examples using asyncio with a semaphore for concurrency control.
        
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
        # We need to get an asyncio classifier for this to work
        # The orchestrator should handle this conversion before calling us
        
        # Ensure we have an async capable classifier
        if not hasattr(classifier, 'predict') or not asyncio.iscoroutinefunction(getattr(classifier, 'predict')):
            raise TypeError("Classifier must have async predict methods for AsyncProcessingStrategy")
        
        # Run the async processing
        return asyncio.run(self._process_batch_async(classifier, examples, is_multi))
    
    async def _process_batch_async(self, classifier: Any, examples: List[Any], is_multi: bool = False) -> List[Dict[str, Any]]:
        """
        Process a batch of examples asynchronously.
        
        Parameters
        ----------
        classifier : Any
            The async classifier to use
        examples : List[Any]
            The examples to process
        is_multi : bool
            Whether this is multi-label classification
            
        Returns
        -------
        List[Dict[str, Any]]
            The processing results
        """
        async def _process_async(example):
            # Create async semaphore to limit concurrency
            async with asyncio.Semaphore(self.n_jobs):
                if is_multi:
                    prediction = await classifier.predict_multi(example.text)
                    is_correct = set(prediction.labels) == set(example.expected_labels)
                    expected = example.expected_labels
                    predicted = prediction.labels
                else:
                    prediction = await classifier.predict(example.text)
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
        
        # Create tasks
        tasks = [_process_async(example) for example in examples]
        
        # Process with progress bar
        results = await tqdm.gather(*tasks, desc=f"Classifying with {self.n_jobs} async workers")
        return results