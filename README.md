# Poetry Generation with Machine Learning

A multi-approach machine learning project for generating poetry using Hidden Markov Models and Recurrent Neural Networks, trained on Shakespeare and Spenser texts.

## Overview

This project explores different machine learning techniques for poetry generation and analysis. Using classical works from Shakespeare and Edmund Spenser, we implement both probabilistic (HMM) and neural network (RNN/LSTM) approaches to understand and generate poetry with proper meter, rhyme, and style.

The project demonstrates the evolution from traditional statistical methods to modern deep learning for natural language generation, with special attention to poetic constraints like syllable counting and rhyme schemes.

## Core Approaches

### 1. Hidden Markov Models (HMM)
**File**: `set6hmm.py`

Full-featured HMM implementation for sequence modeling:

```python
# Train HMM on poetry corpus
hmm = HiddenMarkovModel(A, O)
hmm.supervised_learning(word_sequences, state_sequences)

# Generate new poetry
generated_poem = hmm.generate_emission(poem_length)
```

**Key Features**:
- **Viterbi Algorithm**: Find most likely state sequences
- **Forward-Backward**: Probability computation and training
- **Supervised & Unsupervised Learning**: Multiple training paradigms
- **Poetry Generation**: Sample new text from learned distributions

### 2. Recurrent Neural Networks
**File**: `project3_RNN.ipynb`

LSTM-based character-level text generation:

```python
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
```

**Key Features**:
- **Character-Level Generation**: Learn patterns at character level
- **LSTM Architecture**: Long short-term memory for sequence learning
- **PyTorch Implementation**: Modern deep learning framework
- **Sequence Training**: Next-character prediction with backpropagation

### 3. Poetry Analysis & Utilities
**Files**: `poem_utils.py`, `wordcloud_utils.py`

Specialized tools for poetry processing:

```python
# Parse poetry structure
poems = parse_poems('shakespeare.txt')
words = separate_words(poems)

# Syllable counting for meter
syllable_counts = line_syllable_count(line, syllable_dict)
```

**Features**:
- **Text Parsing**: Extract poems and structure from raw text
- **Syllable Analysis**: Count syllables for metrical analysis
- **Word Mapping**: Convert between words and numerical indices
- **Visualization**: Word clouds showing model learned patterns

## Dataset & Processing

### Text Sources
- **Shakespeare**: 2,615 lines of sonnets and poems
- **Spenser**: 1,512 lines from *The Faerie Queene* and other works
- **Syllable Dictionary**: 3,206 words with syllable counts for meter analysis

### Processing Pipeline
1. **Text Extraction**: Parse poems from raw text files using regex patterns
2. **Tokenization**: Split into words while preserving punctuation
3. **Index Mapping**: Convert words to numerical indices for model training
4. **Syllable Mapping**: Associate words with syllable counts for metrical constraints
5. **Sequence Preparation**: Create training sequences for different model types

## Experimental Notebooks

### Core Implementations
- **`project3_input_poems.ipynb`**: Main HMM experiments (2,357 lines)
- **`project3_RNN.ipynb`**: Neural network approach (864 lines)  
- **`project3_voice_mix.ipynb`**: Advanced voice mixing experiments (245 lines)

### Specialized Experiments
- **`miniproject3_input_as_stanzas.ipynb`**: Stanza-level analysis (1.9MB)
- **`miniproject3_naive_rhyme.ipynb`**: Rhyme pattern detection (389 lines)

### Helper Tools
- **`Emily_project3_helper.py`**: Additional utilities (339 lines)
- **`starter.ipynb`**: Basic setup and data loading

## Key Results & Techniques

### HMM Approach
- **State Interpretation**: Different states capture different poetic styles or themes
- **Word Clouds**: Visualize what each hidden state represents semantically
- **Generation Quality**: Produces grammatically coherent short sequences
- **Rhyme Detection**: Some success in maintaining basic rhyme patterns

### RNN/LSTM Approach  
- **Character-Level**: Learns spelling, punctuation, and basic word structure
- **Sequence Length**: Trains on 40-character sequences with overlapping windows
- **Architecture**: 150 hidden units, cross-entropy loss, SGD optimization
- **Generation**: Can produce Shakespeare-style text with learned patterns

### Comparative Analysis
- **HMMs**: Better for word-level coherence and vocabulary control
- **RNNs**: Superior for character-level patterns and longer dependencies
- **Hybrid Potential**: Combining approaches for different aspects of poetry

## Advanced Features

### Syllable-Aware Generation
```python
# Count syllables for metrical constraints
possible_counts = line_syllable_count(['shall', 'i', 'compare'], syllable_dict)
# Returns possible syllable interpretations for the line
```

### Multi-Modal Analysis
- **Statistical Patterns**: HMM state analysis reveals thematic clustering
- **Neural Representations**: LSTM hidden states capture syntactic patterns
- **Visualization**: Word clouds show semantic groupings by model states

### Rhyme & Meter
- **Syllable Dictionary**: Comprehensive mapping for metrical analysis
- **Rhyme Schemes**: Pattern detection in training data
- **Generation Constraints**: Attempt to maintain poetic structure

## File Structure

```
cs155-project3/
├── README.md                    # This documentation
├── set6hmm.py                  # Core HMM implementation (469 lines)
├── poem_utils.py               # Poetry processing utilities (97 lines)
├── wordcloud_utils.py          # Visualization tools (62 lines)
├── Emily_project3_helper.py    # Additional helper functions (339 lines)
├── project3_RNN.ipynb         # LSTM implementation (864 lines)
├── project3_input_poems.ipynb # Main HMM experiments (2,357 lines)
├── project3_voice_mix.ipynb   # Voice mixing experiments (245 lines)
├── miniproject3_*.ipynb       # Specialized experiments
├── shakespeare.txt            # Shakespeare poetry corpus (98KB)
├── spenser.txt               # Spenser poetry corpus (63KB)
├── syllable_dict.txt         # Syllable count dictionary (33KB)
├── project3.pdf             # Project specification (182KB)
└── __pycache__/             # Python bytecode cache
```

## Technical Requirements

### Dependencies
```bash
pip install torch numpy matplotlib wordcloud pandas requests
```

### Performance Notes
- **HMM Training**: Fast convergence for moderate vocabulary sizes
- **RNN Training**: Requires GPU for reasonable training times
- **Memory Usage**: Character-level models more memory intensive
- **Generation Speed**: HMMs faster for inference, RNNs slower but higher quality

## Usage Examples

### Quick Start with HMM
```python
from poem_utils import parse_poems, separate_words, word_idx_mappings
from set6hmm import supervised_HMM

# Load and process data
poems = parse_poems('shakespeare.txt')  
words = separate_words(poems)
word_to_idx, idx_to_word = word_idx_mappings(words)

# Train model
hmm = supervised_HMM(observation_sequences, state_sequences)

# Generate poetry
generated_text = hmm.generate_emission(100)
```

### RNN Poetry Generation
```python
# Train character-level LSTM
rnn = LSTM(num_chars, hidden_size=150, num_chars)
train_sequences = get_training_seqs(poems, seq_length=40)

# Generate new text
with torch.no_grad():
    generated = generate_text(rnn, seed_text, length=200)
```

## Applications & Extensions

### Educational Use
- **Text Generation**: Understanding sequence modeling fundamentals
- **Comparative ML**: Traditional vs. modern approaches to NLP
- **Poetry Analysis**: Computational literature and digital humanities

### Research Directions
- **Conditional Generation**: Generate in specific styles or meters
- **Hybrid Models**: Combine HMM structure with neural networks
- **Evaluation Metrics**: Develop better measures for poetry quality
- **Interactive Systems**: Real-time poetry composition assistance

## Getting Started

1. **Download Data**: Run data download cells in any notebook
2. **Explore HMMs**: Start with `project3_input_poems.ipynb`
3. **Try RNNs**: Experiment with `project3_RNN.ipynb`
4. **Analyze Results**: Use word cloud utilities to visualize learned patterns
5. **Generate Poetry**: Create new poems using trained models

---

*This project demonstrates the intersection of machine learning and creative writing, showing how computational methods can both analyze and generate poetic text while respecting traditional literary constraints.*
