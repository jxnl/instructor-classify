# Refactoring Plan for Unified Evaluator

## Current Structure Issues
- The `UnifiedEvaluator` class is monolithic (600+ lines) handling multiple responsibilities
- Processing strategies (sync, parallel, async) are implemented directly in the class
- Analyzers are hard-coded in the constructor 
- Result reporting is mixed with analysis logic
- Error handling and recovery is minimal
- No caching for resilience against failures

## Modular Architecture Plan

### 1. Separation of Concerns
- Split the current monolithic `UnifiedEvaluator` into smaller specialized classes:
  - `EvaluationConfig`: Configuration loading and validation
  - `ModelManager`: Model client handling and instrumentation
  - `EvaluationRunner`: Core evaluation execution
  - `ResultsManager`: Results collection and storage
  - `AnalyticsEngine`: Analysis and visualization
  - `EvaluationOrchestrator`: Coordination of the evaluation process

### 2. Processing Strategy Pattern
- Create a proper strategy pattern for batch processing:
  - `ProcessingStrategy`: Base abstract strategy class
  - `SyncProcessingStrategy`: Sequential processing
  - `ParallelProcessingStrategy`: Thread-based parallel processing
  - `AsyncProcessingStrategy`: Asyncio-based parallel processing
- Enable runtime strategy selection
- Make strategy creation factory-based to simplify instantiation

### 3. Pluggable Analysis
- Create a common analyzer interface:
  - `BaseAnalyzer`: Abstract base class with common methods
  - Concrete implementations inherit from BaseAnalyzer
  - Analyzer registry for dynamic loading
- Allow configurable analyzers in the YAML config
- Support for user-defined custom analyzers

### 5. Reporting System
- Extract reporting logic to dedicated subsystem:
  - `ReportManager`: Coordinates report generation
  - `ConsoleReporter`: Rich console output
  - `FileReporter`: File-based reporting (JSON, CSV, etc.)
  - `VisualizationReporter`: Chart and graph generation
- Define common reporting interfaces

### 8. Execution Pipeline
- Implement configurable execution pipeline:
  - `Pipeline`: Overall pipeline management
  - `PipelineStage`: Base class for pipeline stages
  - Standard stages: ConfigLoad, DataPrep, ModelInit, Execution, Analysis, Reporting
  - Allows for stage customization and insertion of additional stages

### Additional Improvements
- **Disk Caching**: 
  - Add persistent caching for LLM responses
  - Cache intermediate results during pipeline execution
  - Enable resumption after failures
- **Performance Improvements**:
  - Optimize batch sizes for different models
  - Intelligent throttling based on rate limits
  - Progress tracking with ETA
- **Error Handling**:
  - Graceful degradation on failures
  - Automatic retry with exponential backoff
  - Detailed error reporting

## Implementation Phases

1. **Core Architecture**: Create base classes and interfaces
2. **Processing Strategies**: Implement the strategy pattern
3. **Execution Pipeline**: Build configurable pipeline
4. **Caching Layer**: Add resilient caching
5. **Pluggable Analysis**: Extract and modularize analyzers
6. **Reporting System**: Create reporting subsystem
7. **CLI Integration**: Update CLI to leverage new architecture
8. **Testing**: Comprehensive test suite