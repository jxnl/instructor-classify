"""
Model initialization pipeline stage.

This module provides the pipeline stage for initializing classifier models.
"""

from typing import Any, Dict, List
import instructor
from openai import OpenAI, AsyncOpenAI

from instructor_classify.eval_harness.base import PipelineStage
from instructor_classify.classify import Classifier, AsyncClassifier
from instructor_classify.eval_harness.caching import DiskCache, CachedClassifier, AsyncCachedClassifier


class InstrumentedClassifier(Classifier):
    """Classifier with instrumentation for cost and latency tracking."""
    
    def __init__(self, definition, analyzer, model_name):
        """
        Initialize the instrumented classifier.
        
        Parameters
        ----------
        definition : ClassificationDefinition
            The classification definition
        analyzer : Any
            Cost analyzer for tracking metrics
        model_name : str
            Name of the model
        """
        super().__init__(definition)
        self.analyzer = analyzer
        self.model_name = model_name
    
    def predict(self, text):
        """
        Predict with instrumentation.
        
        Parameters
        ----------
        text : str
            Text to predict
            
        Returns
        -------
        Any
            Prediction result
        """
        self._validate_client()
        timestamp = self.analyzer.start_timing(self.model_name, text)
        messages, context = self._build_messages(text)
        result = self.client.chat.completions.create_with_completion(
            model=self.model_name,
            response_model=self._classification_model,
            messages=messages,
            context=context,
        )
        input_tokens = result[1].usage.prompt_tokens
        output_tokens = result[1].usage.completion_tokens
        self.analyzer.end_timing(timestamp, input_tokens, output_tokens)
        return result[0]
    
    def predict_multi(self, text):
        """
        Predict multi-label with instrumentation.
        
        Parameters
        ----------
        text : str
            Text to predict
            
        Returns
        -------
        Any
            Prediction result
        """
        self._validate_client()
        timestamp = self.analyzer.start_timing(self.model_name, text)
        messages, context = self._build_messages(text)
        result = self.client.chat.completions.create_with_completion(
            model=self.model_name,
            response_model=self._multi_classification_model,
            messages=messages,
            context=context,
        )
        input_tokens = result[1].usage.prompt_tokens
        output_tokens = result[1].usage.completion_tokens
        self.analyzer.end_timing(timestamp, input_tokens, output_tokens)
        return result[0]


class InstrumentedAsyncClassifier(AsyncClassifier):
    """Async classifier with instrumentation for cost and latency tracking."""
    
    def __init__(self, definition, analyzer, model_name):
        """
        Initialize the instrumented async classifier.
        
        Parameters
        ----------
        definition : ClassificationDefinition
            The classification definition
        analyzer : Any
            Cost analyzer for tracking metrics
        model_name : str
            Name of the model
        """
        super().__init__(definition)
        self.analyzer = analyzer
        self.model_name = model_name
    
    async def predict(self, text):
        """
        Predict with instrumentation.
        
        Parameters
        ----------
        text : str
            Text to predict
            
        Returns
        -------
        Any
            Prediction result
        """
        self._validate_client()
        timestamp = self.analyzer.start_timing(self.model_name, text)
        messages, context = self._build_messages(text)
        result = await self.client.chat.completions.create_with_completion(
            model=self.model_name,
            response_model=self._classification_model,
            messages=messages,
            context=context,
        )
        input_tokens = result[1].usage.prompt_tokens
        output_tokens = result[1].usage.completion_tokens
        self.analyzer.end_timing(timestamp, input_tokens, output_tokens)
        return result[0]
    
    async def predict_multi(self, text):
        """
        Predict multi-label with instrumentation.
        
        Parameters
        ----------
        text : str
            Text to predict
            
        Returns
        -------
        Any
            Prediction result
        """
        self._validate_client()
        timestamp = self.analyzer.start_timing(self.model_name, text)
        messages, context = self._build_messages(text)
        result = await self.client.chat.completions.create_with_completion(
            model=self.model_name,
            response_model=self._multi_classification_model,
            messages=messages,
            context=context,
        )
        input_tokens = result[1].usage.prompt_tokens
        output_tokens = result[1].usage.completion_tokens
        self.analyzer.end_timing(timestamp, input_tokens, output_tokens)
        return result[0]


class ModelStage(PipelineStage):
    """
    Pipeline stage for initializing classifier models.
    """
    
    def __init__(self):
        """Initialize the model stage."""
        super().__init__(name="Model Initialization")
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the model initialization stage.
        
        Parameters
        ----------
        context : Dict[str, Any]
            The pipeline context
            
        Returns
        -------
        Dict[str, Any]
            The updated pipeline context
        """
        config = context["config"]
        definition = context["definition"]
        
        # Get cost analyzer from context (added by analyzer stage)
        cost_analyzer = context.get("cost_analyzer")
        if not cost_analyzer:
            self.console.print("[yellow]Warning: Cost analyzer not found, instrumentation will be limited[/yellow]")
        
        # Initialize cache if enabled
        cache = None
        if config.use_cache:
            cache_dir = config.cache_dir
            self.console.print(f"[green]Initializing disk cache in {cache_dir}[/green]")
            cache = DiskCache(cache_dir)
            context["cache"] = cache
        
        # Initialize clients and classifiers for each model
        classifiers = {}
        async_classifiers = {}
        
        for model_name in config.models:
            self.console.print(f"[green]Initializing classifier for model: {model_name}[/green]")
            
            # Initialize OpenAI client with Instructor
            client = instructor.from_openai(OpenAI())
            
            # Create instrumented classifier
            classifier = InstrumentedClassifier(definition, cost_analyzer, model_name)
            classifier.with_client(client)
            
            # Wrap with cache if enabled
            if cache:
                classifier = CachedClassifier(classifier, cache)
            
            classifiers[model_name] = classifier
            
            # For async mode, also create async classifier
            if config.parallel_mode == "async":
                async_client = instructor.from_openai(AsyncOpenAI())
                
                async_classifier = InstrumentedAsyncClassifier(definition, cost_analyzer, model_name)
                async_classifier.with_client(async_client)
                
                # Wrap with cache if enabled
                if cache:
                    async_classifier = AsyncCachedClassifier(async_classifier, cache)
                
                async_classifiers[model_name] = async_classifier
        
        context["classifiers"] = classifiers
        context["async_classifiers"] = async_classifiers
        
        self.console.print(f"[green]Initialized {len(classifiers)} classifiers[/green]")
        
        return context