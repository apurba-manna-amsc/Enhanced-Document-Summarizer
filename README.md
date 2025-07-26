# 📄 Enhanced Document Summarizer

*A sophisticated multi-input document summarization system leveraging advanced clustering optimization and adaptive content selection*

**🚀 Live Demo**: [https://enhanced-document-summarizer-mxgqg6hrrdnvedqj8vww8g.streamlit.app/](https://enhanced-document-summarizer-mxgqg6hrrdnvedqj8vww8g.streamlit.app/)

## 👨‍💻 Author
**Apurba Manna**  
📧 Email: [98apurbamanna@gmail.com](mailto:98apurbamanna@gmail.com)  
💼 LinkedIn: [apurba-manna](https://linkedin.com/in/apurba-manna)  
🐙 GitHub: [apurba-manna-amsc](https://github.com/apurba-manna-amsc)

## 🎯 Problem Statement

Traditional document summarization approaches face several critical challenges:

1. **Information Redundancy**: Simple extractive methods often select similar or overlapping content
2. **Semantic Representation**: Difficulty in capturing diverse semantic themes across large documents
3. **Optimal Granularity**: Determining the right level of content compression without losing key information
4. **Multi-Source Integration**: Inability to effectively combine and summarize content from different input types (PDFs, text, videos)
5. **Representative Selection**: Lack of intelligent chunk selection that considers both content diversity and semantic coherence

### The Core Challenge
Given a document with `n` text chunks, how do we intelligently select `k` representative chunks (where `k << n`) that:
- Maximize semantic diversity
- Minimize information redundancy  
- Preserve the most important content
- Maintain logical coherence

---

## 🧠 My Solution Approach

I developed a **multi-stage pipeline** that combines advanced clustering optimization with adaptive content selection, leveraging both unsupervised learning and geometric optimization principles.

### Solution Architecture

```
Input Sources → Text Processing → Embedding Generation → Clustering Optimization → Adaptive Selection → LLM Summarization
     ↓               ↓                    ↓                      ↓                     ↓                ↓
Multiple Types    Chunking &         Sentence           Multi-Metric K         Weighted Cluster    Context-Aware
(PDF, Text,      Normalization      Transformers        Selection with         Representative      Summary with
YouTube)                                                Elbow Detection         Selection           Groq LLaMA
```

---

## 📊 Stage 1: Multi-Input Processing & Embedding Generation

### Text Chunking Strategy
I use **Recursive Character Text Splitter** with optimized parameters:
- **Chunk Size**: 2000 characters (optimal for context preservation)
- **Overlap**: 200 characters (maintains semantic continuity)
- **Separation Hierarchy**: Paragraphs → Sentences → Words → Characters

### Embedding Generation
Utilizing **Sentence-BERT (all-MiniLM-L6-v2)** for semantic embeddings:

```python
embeddings = SentenceTransformer.encode(text_chunks)
# Output: embeddings ∈ ℝ^(n×384) where n = number of chunks
```

**Why Sentence-BERT?**
- Superior semantic understanding compared to traditional TF-IDF
- Captures contextual relationships between chunks
- Efficient computational complexity: O(n·d) where d=384

---

## 🧮 Stage 2: Optimal Clustering with Multi-Metric Optimization

This is the **core innovation** of my approach - a sophisticated method to determine the optimal number of clusters using multiple validation metrics.

### Problem Formulation
Given embeddings `X ∈ ℝ^(n×d)`, find optimal `k*` such that:

```
k* = argmax{k∈[k_min, k_max]} f(k)
```

Where `f(k)` is a composite function combining multiple clustering metrics.

### Multi-Metric Evaluation Framework

#### 1. **Silhouette Score (Cohesion & Separation)**
Measures how well-separated clusters are:

```
S(k) = (1/n) Σᵢ (bᵢ - aᵢ) / max(aᵢ, bᵢ)
```

Where:
- `aᵢ`: Mean intra-cluster distance for point i
- `bᵢ`: Mean distance to nearest cluster for point i
- Range: [-1, 1], **higher is better**

#### 2. **Calinski-Harabasz Index (Variance Ratio)**
Evaluates cluster density and separation:

```
CH(k) = [Tr(Bₖ)/(k-1)] / [Tr(Wₖ)/(n-k)]
```

Where:
- `Bₖ`: Between-cluster scatter matrix
- `Wₖ`: Within-cluster scatter matrix
- `Tr(·)`: Matrix trace
- **Higher values indicate better clustering**

**Mathematical Detail:**
```
Bₖ = Σⱼ nⱼ(μⱼ - μ)(μⱼ - μ)ᵀ
Wₖ = Σⱼ Σᵢ∈Cⱼ (xᵢ - μⱼ)(xᵢ - μⱼ)ᵀ
```

#### 3. **Davies-Bouldin Index (Cluster Similarity)**
Measures average similarity between most similar clusters:

```
DB(k) = (1/k) Σᵢ max_j≠i [(σᵢ + σⱼ) / d(cᵢ, cⱼ)]
```

Where:
- `σᵢ`: Average distance from cluster i centroid to points in cluster i
- `d(cᵢ, cⱼ)`: Distance between centroids i and j
- **Lower values indicate better clustering**

### Composite Score Calculation

#### Step 1: Normalization
Apply min-max normalization to make metrics comparable:

```
Ŝ(k) = [S(k) - min(S)] / [max(S) - min(S)]
ĈH(k) = [CH(k) - min(CH)] / [max(CH) - min(CH)]
D̂B(k) = [max(DB) - DB(k)] / [max(DB) - min(DB)]  # Inverted since lower is better
```

#### Step 2: Weighted Composite Score
```
Composite(k) = w₁·Ŝ(k) + w₂·ĈH(k) + w₃·D̂B(k)
```

Default weights: `w₁ = w₂ = w₃ = 1/3` (equal importance)

#### Step 3: Over-clustering Penalty
To prevent excessive fragmentation:

```
Penalty(k) = λ · (k - k_min) / (k_max - k_min)
PenalizedScore(k) = Composite(k) - Penalty(k)
```

Where `λ = 0.03` (empirically optimized)

### Geometric Elbow Detection

The **elbow method** identifies the optimal k by finding the point of maximum curvature in the score curve.

#### Mathematical Implementation:
1. **Line Definition**: Draw line from first point `(k_min, score_min)` to last point `(k_max, score_max)`
2. **Distance Calculation**: For each point `(k, score(k))`, compute perpendicular distance to the line:

```
distance(k) = |ax₀ + by₀ + c| / √(a² + b²)
```

Where the line equation is `ax + by + c = 0`

3. **Optimal Selection**: 
```
k* = argmax{k} distance(k)
```

#### Geometric Intuition:
- The elbow represents the point where additional clusters provide diminishing returns
- Maximum distance from the line indicates the steepest change in clustering quality
- This approach is robust against noise and outliers in the metric values

### Example Optimization Result:

| k | Silhouette | Calinski-H | Davies-B | Composite | Penalty | Final Score | Distance | Selected |
|---|------------|------------|----------|-----------|---------|-------------|----------|----------|
| 2 | 0.342      | 1247.3     | 1.423    | 0.234     | 0.000   | 0.234       | 0.123    |          |
| 3 | 0.456      | 1834.7     | 1.187    | 0.687     | 0.015   | 0.672       | **0.341**| ✅       |
| 4 | 0.441      | 1923.1     | 1.203    | 0.712     | 0.030   | 0.682       | 0.298    |          |
| 5 | 0.438      | 1876.4     | 1.298    | 0.689     | 0.045   | 0.644       | 0.187    |          |

**Result**: k* = 3 selected due to maximum elbow distance (0.341)

---

## 🎨 Stage 3: Adaptive Representative Chunk Selection

After determining optimal clusters, I implement a **sophisticated selection algorithm** that considers both cluster characteristics and global document structure.

### Cluster Feature Analysis

For each cluster `j`, compute:

#### 1. **Cluster Size Weight**
```
w_size(j) = |Cⱼ| / n
```
Larger clusters get higher weight (more information content)

#### 2. **Cluster Spread Weight**  
```
w_spread(j) = (1/|Cⱼ|) Σᵢ∈Cⱼ ||xᵢ - μⱼ||₂
```
Higher spread indicates diverse content within cluster

#### 3. **Combined Cluster Weight**
```
w_total(j) = α · w_size(j) + β · w_spread(j)
```
Default: `α = 0.7, β = 0.3` (size-dominant strategy)

### Adaptive Allocation Strategy

#### Step 1: Weight-Based Allocation
Distribute total representatives across clusters proportionally:

```
reps(j) = round[(w_total(j) / Σₖ w_total(k)) · total_reps]
```

#### Step 2: Constraint Satisfaction
Ensure: `1 ≤ reps(j) ≤ |Cⱼ|` for all clusters

#### Step 3: Deficit/Surplus Adjustment
```python
while allocated_total ≠ target_total:
    if deficit:
        # Allocate to highest-weight cluster with capacity
        best_cluster = argmax{j: reps(j) < |Cⱼ|} w_total(j)
        reps(best_cluster) += 1
    else:
        # Remove from lowest-weight cluster with reps > 1  
        worst_cluster = argmin{j: reps(j) > 1} w_total(j)
        reps(worst_cluster) -= 1
```

### Representative Selection within Clusters

For each cluster, select chunks closest to centroid:

```
selected(j) = argmin{reps(j)} {||xᵢ - μⱼ||₂ : i ∈ Cⱼ}
```

### Hybrid Global Enhancement

**Innovation**: Add globally representative chunks beyond cluster-based selection:

```python
global_distances = [min_k ||xᵢ - μₖ||₂ for i in range(n)]
global_selection = argsort(global_distances)[:global_top_k]
```

This ensures we capture universally important content that might be underrepresented in cluster-based selection.

---

## 🔧 Stage 4: Context-Aware Summarization

### LLM Integration with Groq
Using **LLaMA-3-8B** via Groq API for final summarization:

```python
system_prompt = f"""
Generate a {summary_type} summary in ~{max_words} words.
Focus on: {get_focus_areas(summary_type)}
Maintain logical flow and highlight key insights.
"""

summary = groq_client.chat.completions.create(
    model="llama3-8b-8192",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": combined_representative_chunks}
    ],
    temperature=0.3,  # Low temperature for factual consistency
    max_tokens=max_words * 2
)
```

### Summary Type Specialization

Different prompting strategies for different use cases:
- **General**: Comprehensive overview with balanced coverage
- **Key Findings**: Focus on conclusions, results, and insights  
- **Risks**: Highlight potential issues, challenges, and mitigation strategies
- **Legal Clauses**: Extract terms, conditions, and legal implications

---

## 📈 Performance Analysis & Metrics

### Compression Efficiency
```
Compression Ratio = selected_chunks / total_chunks
Information Retention = semantic_similarity(original, summary)
Diversity Score = average_pairwise_distance(selected_embeddings)
```

### Quality Indicators

1. **Clustering Quality**: 
   - Optimal silhouette score achieved
   - Balanced cluster sizes
   - Clear semantic separation

2. **Selection Quality**:
   - Representative diversity
   - Content coverage across document sections
   - Minimal semantic redundancy

3. **Summary Quality**:
   - Coherence and readability
   - Key information preservation
   - Appropriate length and focus

### Example Performance Metrics:
```
Total Input Chunks: 47
Selected Representatives: 8 (17% compression)
Optimal Clusters: 3
Silhouette Score: 0.456
Calinski-Harabasz: 1834.7
Davies-Bouldin: 1.187
Final Summary: 487 words
Processing Time: 12.3 seconds
```

---

## 🏗️ Technical Implementation Details

### Key Dependencies & Architecture

```python
# Core ML/NLP Stack
sentence-transformers==2.2.2  # Semantic embeddings
scikit-learn==1.3.0          # Clustering algorithms
numpy==1.24.3                # Numerical computations
scipy==1.11.1               # Statistical functions

# LLM Integration
groq==0.4.1                 # LLaMA access
langchain==0.1.0            # Document processing

# Web Interface
streamlit==1.28.1           # Interactive UI
plotly==5.17.0             # Analytics visualization

# Document Processing
PyPDF2==3.0.1              # PDF extraction
python-docx==0.8.11        # DOCX processing
youtube-transcript-api==0.6.1  # Video transcripts
```

### Scalability Considerations

1. **Memory Optimization**: 
   - Embedding caching to avoid recomputation
   - Batch processing for large documents
   - Lazy loading of models

2. **Computational Efficiency**:
   - Vectorized operations using NumPy
   - Parallel clustering evaluation
   - Early stopping for poor k values

3. **Storage Management**:
   - Session state for web interface
   - Temporary file cleanup
   - Result caching between runs

---

## 🎯 Innovation Highlights

### 1. **Multi-Metric Clustering Optimization**
Unlike traditional single-metric approaches, I combine three complementary metrics with geometric elbow detection for robust k-selection.

### 2. **Adaptive Weighted Selection**
The selection algorithm dynamically balances cluster size and content diversity, ensuring optimal representation across semantic themes.

### 3. **Hybrid Global Enhancement**
Addition of globally representative chunks beyond cluster-based selection captures universally important content.

### 4. **Multi-Source Integration**
Seamless processing of PDFs, plain text, and YouTube transcripts in a unified pipeline.

### 5. **Comprehensive Analytics**
Detailed visualization and metrics for every stage of the process, enabling deep insight into the summarization quality.

---

## 🔬 Mathematical Foundations Summary

The system leverages several key mathematical concepts:

1. **Unsupervised Learning**: K-means clustering with multiple validation metrics
2. **Optimization Theory**: Multi-objective optimization with penalty functions
3. **Geometric Analysis**: Elbow detection using perpendicular distance calculations
4. **Linear Algebra**: High-dimensional embedding space operations
5. **Statistical Analysis**: Normalization and scoring functions
6. **Graph Theory**: Semantic similarity networks for content selection

This approach demonstrates deep understanding of:
- **Machine Learning**: Clustering algorithms and validation
- **Optimization**: Multi-criteria decision making
- **Geometry**: Distance calculations and curve analysis  
- **Statistics**: Normalization and scoring methodologies
- **NLP**: Semantic embeddings and text processing

---

## 🚀 Usage Example

```python
from summarizer import EnhancedDocumentSummarizer

# Initialize
summarizer = EnhancedDocumentSummarizer(groq_api_key="your_key")

# Summarize with multiple inputs
results = summarizer.summarize_document(
    document_file=uploaded_pdf,
    plain_text="Additional context...",
    youtube_url="https://youtube.com/watch?v=...",
    total_representative_chunks=8,
    summary_type="key_findings",
    max_words=500,
    k_range=(2, 12)
)

# Access results
print(f"Summary: {results['summary']}")
print(f"Compression: {results['metadata']['compression_ratio']}")
print(f"Optimal k: {results['metadata']['optimal_k']}")
```

---

## 📚 Future Enhancements

1. **Dynamic Embedding Models**: Adaptation to domain-specific embeddings
2. **Multi-Language Support**: Extension to non-English documents  
3. **Hierarchical Clustering**: Implementation of HDBSCAN for density-based clustering
4. **Active Learning**: User feedback integration for iterative improvement
5. **Distributed Processing**: Scaling to enterprise-level document volumes

---

## 🎓 Learning Showcase

This project demonstrates proficiency in:

- **Advanced Machine Learning**: Multi-metric clustering optimization
- **Mathematical Modeling**: Complex scoring functions and geometric analysis
- **Software Architecture**: Scalable, modular pipeline design
- **Data Science**: Comprehensive analytics and visualization
- **NLP Engineering**: Multi-modal text processing and semantic analysis
- **System Integration**: LLM integration with traditional ML approaches

The implementation showcases not just coding ability, but deep understanding of the underlying mathematical principles and their practical application to solve complex information processing challenges.

---

*Built with passion for intelligent document processing and advanced NLP techniques. This project represents a comprehensive exploration of clustering optimization, adaptive content selection, and multi-source document summarization.*
