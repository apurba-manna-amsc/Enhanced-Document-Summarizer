from typing import Tuple, Dict, List, Any, Optional
import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import rankdata
import warnings
import os
import requests
import pickle
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
import time
from sentence_transformers import SentenceTransformer
from groq import Groq
import json
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
import re
import tempfile
import streamlit as st

warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

class EnhancedDocumentSummarizer:
    """
    Enhanced clustering-based document summarization system with multi-metric clustering
    optimization and adaptive representative chunk selection.
    """
    
    def __init__(self, 
                 groq_api_key: str = None, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 llm_model: str = "llama3-8b-8192"):
        """
        Initialize the enhanced document summarizer.
        
        Args:
            groq_api_key: Groq API key for LLM calls
            embedding_model: Sentence Transformer model name
            llm_model: Groq model to use for summarization
        """
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        self.llm_model = llm_model
        
        # Initialize Sentence Transformer for embeddings
        print(f"🔄 Loading embedding model: {embedding_model}")
        self.embeddings_model = SentenceTransformer(embedding_model)
        
        # Initialize Groq client
        self.groq_client = Groq(api_key=self.groq_api_key)
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            length_function=len
        )
        
        # Cache for embeddings and clustering results
        self.cache = {
            'embeddings': None,
            'text_chunks': None,
            'clustering_results': {}
        }

    def get_video_id(self, url: str) -> str:
        """Extract video ID from YouTube URL."""
        match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
        if match:
            return match.group(1)
        else:
            raise ValueError("Invalid YouTube URL")

    def get_youtube_transcript(self, video_url: str) -> str:
        """Get transcript from YouTube video."""
        try:
            video_id = self.get_video_id(video_url)
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            full_text = " ".join([entry['text'] for entry in transcript])
            return full_text
        except (TranscriptsDisabled, NoTranscriptFound):
            raise Exception("No transcript available for this video.")
        except VideoUnavailable:
            raise Exception("Video is unavailable.")
        except Exception as e:
            raise Exception(f"Error getting transcript: {str(e)}")
        
    def load_document(self, file_path: str) -> List[str]:
        """
        Load and extract text from document (PDF or DOCX).
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of text chunks
        """
        try:
            if file_path.lower().endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            elif file_path.lower().endswith('.docx'):
                loader = Docx2txtLoader(file_path)
            else:
                raise ValueError("Unsupported file format. Use PDF or DOCX files.")
                
            documents = loader.load()
            
            # Split documents into chunks
            chunks = self.text_splitter.split_documents(documents)
            
            # Extract text content from chunks
            text_chunks = [chunk.page_content for chunk in chunks]
            
            return text_chunks
            
        except Exception as e:
            raise Exception(f"Error loading document: {str(e)}")

    def process_plain_text(self, text: str) -> List[str]:
        """
        Process plain text input into chunks.
        
        Args:
            text: Plain text input
            
        Returns:
            List of text chunks
        """
        try:
            # Create a temporary document-like object for the text splitter
            chunks = self.text_splitter.split_text(text)
            return chunks
        except Exception as e:
            raise Exception(f"Error processing plain text: {str(e)}")

    def process_multiple_inputs(self, 
                              document_file=None, 
                              plain_text: str = "", 
                              youtube_url: str = "",
                              progress_callback=None) -> List[str]:
        """
        Process multiple input types and combine them into text chunks.
        
        Args:
            document_file: Uploaded document file
            plain_text: Plain text input
            youtube_url: YouTube video URL
            progress_callback: Function to call for progress updates
            
        Returns:
            Combined list of text chunks
        """
        all_chunks = []
        sources = []
        
        try:
            # Process document file
            if document_file is not None:
                if progress_callback:
                    progress_callback("Loading document...")
                
                # Save uploaded file to temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{document_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(document_file.read())
                    tmp_file_path = tmp_file.name
                
                try:
                    doc_chunks = self.load_document(tmp_file_path)
                    all_chunks.extend(doc_chunks)
                    sources.append(f"Document ({len(doc_chunks)} chunks)")
                finally:
                    # Clean up temporary file
                    os.unlink(tmp_file_path)
            
            # Process plain text
            if plain_text.strip():
                if progress_callback:
                    progress_callback("Processing plain text...")
                    
                text_chunks = self.process_plain_text(plain_text)
                all_chunks.extend(text_chunks)
                sources.append(f"Plain text ({len(text_chunks)} chunks)")
            
            # Process YouTube transcript
            if youtube_url.strip():
                if progress_callback:
                    progress_callback("Getting YouTube transcript...")
                    
                transcript = self.get_youtube_transcript(youtube_url)
                yt_chunks = self.process_plain_text(transcript)
                all_chunks.extend(yt_chunks)
                sources.append(f"YouTube ({len(yt_chunks)} chunks)")
            
            if not all_chunks:
                raise ValueError("No valid input provided")
            
            # Cache the combined chunks
            self.cache['text_chunks'] = all_chunks
            
            print(f"✅ Combined processing complete. Sources: {', '.join(sources)}")
            print(f"✅ Total chunks: {len(all_chunks)}")
            
            return all_chunks
            
        except Exception as e:
            raise Exception(f"Error processing inputs: {str(e)}")
    
    def generate_embeddings(self, texts: List[str], use_cache: bool = True, progress_callback=None) -> np.ndarray:
        """
        Generate embeddings for text chunks using Sentence Transformers.
        
        Args:
            texts: List of text chunks
            use_cache: Whether to use cached embeddings
            progress_callback: Function to call for progress updates
            
        Returns:
            Numpy array of embeddings
        """
        # Check cache first
        if use_cache and self.cache['embeddings'] is not None:
            if len(self.cache['embeddings']) == len(texts):
                print("✅ Using cached embeddings")
                return self.cache['embeddings']
        
        try:
            if progress_callback:
                progress_callback("Generating embeddings...")
            
            print("🔄 Generating embeddings with Sentence Transformers...")
            
            # Generate embeddings using Sentence Transformers
            embeddings = self.embeddings_model.encode(
                texts, 
                convert_to_numpy=True,
                show_progress_bar=False,  # Disable internal progress bar for Streamlit
                batch_size=32
            )
            
            # Cache the embeddings
            if use_cache:
                self.cache['embeddings'] = embeddings
            
            print(f"✅ Generated embeddings: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            raise Exception(f"Error generating embeddings: {str(e)}")
    
    def calculate_clustering_metrics(self, embeddings: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        Calculate all clustering validation metrics.
        
        Args:
            embeddings: Array of embeddings
            labels: Cluster labels
            
        Returns:
            Dictionary with metric scores
        """
        try:
            silhouette = silhouette_score(embeddings, labels)
            calinski = calinski_harabasz_score(embeddings, labels)
            davies_bouldin = davies_bouldin_score(embeddings, labels)
            
            return {
                'silhouette': silhouette,
                'calinski_harabasz': calinski,
                'davies_bouldin': davies_bouldin
            }
        except Exception as e:
            print(f"Warning: Error calculating metrics: {str(e)}")
            return {'silhouette': 0, 'calinski_harabasz': 0, 'davies_bouldin': float('inf')}
    
    def find_optimal_clusters_multi_metric(self, 
                                        embeddings: np.ndarray, 
                                        k_range: Tuple[int, int] = (2, 15),
                                        metric_weights: Dict[str, float] = None,
                                        min_silhouette: float = 0.0,
                                        penalty_alpha: float = 0.03,
                                        progress_callback=None) -> Tuple[int, Dict]:
        """
        Find optimal number of clusters using multi-metric + elbow detection + penalty strategy.
        """

        if metric_weights is None:
            metric_weights = {
                'silhouette': 1/3,
                'calinski_harabasz': 1/3,
                'davies_bouldin': 1/3
            }

        if progress_callback:
            progress_callback("Finding optimal clusters...")

        print(f"🔍 Running optimal cluster search from k={k_range[0]} to {k_range[1]}")

        min_k, max_k = k_range
        max_k = min(max_k, len(embeddings) - 1)
        k_values = list(range(min_k, max_k + 1))
        results = []

        for k in k_values:
            print(f" → Testing k={k}")
            cache_key = f"k_{k}"

            if cache_key in self.cache['clustering_results']:
                metrics = self.cache['clustering_results'][cache_key]['metrics']
                print(f"   ✅ Using cached results")
            else:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(embeddings)
                metrics = self.calculate_clustering_metrics(embeddings, labels)

                self.cache['clustering_results'][cache_key] = {
                    'kmeans': kmeans,
                    'labels': labels,
                    'metrics': metrics
                }

            results.append({
                'k': k,
                **metrics
            })

        # Convert to arrays
        metrics_data = {
            'k': np.array([r['k'] for r in results]),
            'silhouette': np.array([r['silhouette'] for r in results]),
            'calinski_harabasz': np.array([r['calinski_harabasz'] for r in results]),
            'davies_bouldin': np.array([r['davies_bouldin'] for r in results])
        }

        def min_max_normalize(arr):
            return np.ones_like(arr) if np.max(arr) == np.min(arr) else (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

        norm_metrics = {
            'silhouette': min_max_normalize(metrics_data['silhouette']),
            'calinski_harabasz': min_max_normalize(metrics_data['calinski_harabasz']),
            'davies_bouldin': 1 - min_max_normalize(metrics_data['davies_bouldin'])
        }

        # Composite score
        composite_scores = np.zeros(len(k_values))
        for metric, weight in metric_weights.items():
            composite_scores += norm_metrics[metric] * weight

        # Penalty for over-clustering
        penalty = penalty_alpha * (metrics_data['k'] - min_k)
        penalized_scores = composite_scores - penalty

        # Silhouette threshold filtering
        valid = metrics_data['silhouette'] >= min_silhouette
        if not np.any(valid):
            warnings.warn(f"No k meets silhouette threshold {min_silhouette}, relaxing condition.")
            valid = np.ones_like(metrics_data['silhouette'], dtype=bool)

        # Elbow detection on penalized scores
        def detect_elbow(scores: np.ndarray) -> Tuple[int, np.ndarray]:
            if len(scores) < 3:
                return 0, np.zeros_like(scores)
            
            x = np.arange(len(scores))
            y = scores
            
            p1 = np.array([x[0], y[0]])
            p2 = np.array([x[-1], y[-1]])
            
            line_vec = p2 - p1
            line_len = np.linalg.norm(line_vec)
            
            if line_len == 0:
                return 0, np.zeros_like(scores)
            
            line_unit = line_vec / line_len
            
            distances = []
            for i in range(len(scores)):
                point = np.array([x[i], y[i]])
                point_vec = point - p1
                projection = np.dot(point_vec, line_unit) * line_unit
                perp_distance = np.linalg.norm(point_vec - projection)
                distances.append(perp_distance)
            
            distances = np.array(distances)
            return int(np.argmax(distances)), distances

        elbow_idx, elbow_distances = detect_elbow(penalized_scores[valid])
        
        # Create full distances array for all k values
        full_distances = np.zeros(len(k_values))
        if np.any(valid):
            x = np.arange(len(penalized_scores))
            y = penalized_scores
            p1 = np.array([x[0], y[0]])
            p2 = np.array([x[-1], y[-1]])
            line_vec = p2 - p1
            line_len = np.linalg.norm(line_vec)
            
            if line_len > 0:
                line_unit = line_vec / line_len
                for i in range(len(penalized_scores)):
                    point = np.array([x[i], y[i]])
                    point_vec = point - p1
                    projection = np.dot(point_vec, line_unit) * line_unit
                    perp_distance = np.linalg.norm(point_vec - projection)
                    full_distances[i] = perp_distance
        
        optimal_k = metrics_data['k'][valid][elbow_idx]

        print(f"✅ Selected k = {optimal_k} using elbow+penalty method")

        # Rank report
        ranks = {metric: rankdata(-norm_metrics[metric], method='min') for metric in norm_metrics}
        composite_ranks = rankdata(-composite_scores, method='min')

        detailed_results = []
        for i, k in enumerate(k_values):
            detailed_results.append({
                'k': k,
                'silhouette': metrics_data['silhouette'][i],
                'calinski_harabasz': metrics_data['calinski_harabasz'][i],
                'davies_bouldin': metrics_data['davies_bouldin'][i],
                'silhouette_norm': norm_metrics['silhouette'][i],
                'calinski_norm': norm_metrics['calinski_harabasz'][i],
                'davies_bouldin_norm': norm_metrics['davies_bouldin'][i],
                'composite_score': composite_scores[i],
                'penalty': penalty[i],
                'penalized_score': penalized_scores[i],
                'distance': full_distances[i],
                'silhouette_rank': ranks['silhouette'][i],
                'calinski_rank': ranks['calinski_harabasz'][i],
                'davies_bouldin_rank': ranks['davies_bouldin'][i],
                'composite_rank': composite_ranks[i],
                'meets_threshold': valid[i],
                'is_optimal': (k == optimal_k)
            })

        return optimal_k, {
            'detailed_results': detailed_results,
            'metrics_data': metrics_data,
            'normalized_metrics': norm_metrics,
            'composite_scores': composite_scores,
            'penalized_scores': penalized_scores,
            'composite_ranks': composite_ranks,
            'penalty': penalty,
            'elbow_index': elbow_idx,
            'elbow_distances': full_distances
        }

    def calculate_cluster_features(self, embeddings: np.ndarray, 
                                 kmeans: KMeans, 
                                 cluster_labels: np.ndarray) -> Dict[int, Dict]:
        """Calculate features for each cluster (size and spread)."""
        cluster_features = {}
        
        for cluster_id in range(kmeans.n_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            
            if len(cluster_indices) == 0:
                continue
                
            cluster_embeddings = embeddings[cluster_indices]
            centroid = kmeans.cluster_centers_[cluster_id]
            
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            spread = np.mean(distances)
            
            cluster_features[cluster_id] = {
                'size': len(cluster_indices),
                'spread': spread,
                'indices': cluster_indices,
                'distances_to_centroid': distances
            }
        
        return cluster_features
    
    def adaptive_chunk_selection(self, 
                               embeddings: np.ndarray,
                               text_chunks: List[str],
                               kmeans: KMeans,
                               cluster_labels: np.ndarray,
                               total_chunks: int,
                               size_weight: float = 0.7,
                               spread_weight: float = 0.3,
                               global_top_k: int = None,
                               hybrid: bool = True) -> Tuple[List[str], Dict]:
        """Select representative chunks using adaptive sampling with optional hybrid approach."""
        print(f"🔄 Adaptive chunk selection (total: {total_chunks})")
        
        # Calculate cluster features
        cluster_features = self.calculate_cluster_features(embeddings, kmeans, cluster_labels)
        
        # Calculate weights for each cluster
        total_size = sum(features['size'] for features in cluster_features.values())
        max_spread = max(features['spread'] for features in cluster_features.values()) if cluster_features else 1
        
        cluster_weights = {}
        for cluster_id, features in cluster_features.items():
            size_ratio = features['size'] / total_size
            spread_ratio = features['spread'] / max_spread
            weight = size_weight * size_ratio + spread_weight * spread_ratio
            cluster_weights[cluster_id] = weight
        
        # Convert weights to number of representatives per cluster
        total_weight = sum(cluster_weights.values())
        cluster_reps = {}
        allocated_chunks = 0
        
        for cluster_id, weight in cluster_weights.items():
            reps = round((weight / total_weight) * total_chunks)
            reps = max(1, min(reps, cluster_features[cluster_id]['size']))
            cluster_reps[cluster_id] = reps
            allocated_chunks += reps
        
        # Adjust if we over/under allocated
        while allocated_chunks != total_chunks:
            if allocated_chunks < total_chunks:
                best_cluster = max(cluster_weights.keys(), 
                                 key=lambda x: cluster_weights[x] if cluster_reps[x] < cluster_features[x]['size'] else -1)
                if cluster_reps[best_cluster] < cluster_features[best_cluster]['size']:
                    cluster_reps[best_cluster] += 1
                    allocated_chunks += 1
                else:
                    break
            else:
                worst_cluster = min(cluster_weights.keys(), 
                                  key=lambda x: cluster_weights[x] if cluster_reps[x] > 1 else float('inf'))
                if cluster_reps[worst_cluster] > 1:
                    cluster_reps[worst_cluster] -= 1
                    allocated_chunks -= 1
                else:
                    break
        
        # Select representative chunks from each cluster
        selected_chunks = []
        cluster_selections = {}
        
        for cluster_id, num_reps in cluster_reps.items():
            features = cluster_features[cluster_id]
            closest_indices = np.argsort(features['distances_to_centroid'])[:num_reps]
            
            cluster_chunks = []
            for idx in closest_indices:
                original_idx = features['indices'][idx]
                chunk = text_chunks[original_idx]
                selected_chunks.append({
                    'text': chunk,
                    'original_idx': original_idx,
                    'cluster_id': cluster_id,
                    'distance_to_centroid': features['distances_to_centroid'][idx],
                    'selection_method': 'adaptive'
                })
                cluster_chunks.append(chunk)
            
            cluster_selections[cluster_id] = {
                'chunks': cluster_chunks,
                'num_selected': num_reps,
                'cluster_size': features['size'],
                'cluster_weight': cluster_weights[cluster_id]
            }
        
        # Hybrid approach: add globally closest chunks
        if hybrid and global_top_k and global_top_k > 0:
            print(f"   Adding {global_top_k} globally closest chunks...")
            
            global_distances = []
            for i, embedding in enumerate(embeddings):
                min_distance = float('inf')
                closest_centroid = -1
                
                for cluster_id in range(kmeans.n_clusters):
                    distance = np.linalg.norm(embedding - kmeans.cluster_centers_[cluster_id])
                    if distance < min_distance:
                        min_distance = distance
                        closest_centroid = cluster_id
                
                global_distances.append({
                    'idx': i,
                    'distance': min_distance,
                    'closest_cluster': closest_centroid
                })
            
            global_distances.sort(key=lambda x: x['distance'])
            
            selected_indices = {chunk['original_idx'] for chunk in selected_chunks}
            added_global = 0
            
            for item in global_distances:
                if added_global >= global_top_k:
                    break
                    
                if item['idx'] not in selected_indices:
                    chunk = text_chunks[item['idx']]
                    selected_chunks.append({
                        'text': chunk,
                        'original_idx': item['idx'],
                        'cluster_id': item['closest_cluster'],
                        'distance_to_centroid': item['distance'],
                        'selection_method': 'global_top_k'
                    })
                    selected_indices.add(item['idx'])
                    added_global += 1
        
        representative_texts = [chunk['text'] for chunk in selected_chunks]
        
        selection_info = {
            'cluster_features': cluster_features,
            'cluster_weights': cluster_weights,
            'cluster_reps': cluster_reps,
            'cluster_selections': cluster_selections,
            'total_selected': len(selected_chunks),
            'adaptive_selected': sum(cluster_reps.values()),
            'global_selected': len(selected_chunks) - sum(cluster_reps.values()) if hybrid else 0,
            'detailed_selections': selected_chunks
        }
        
        return representative_texts, selection_info
    
    def generate_summary(self, representative_chunks: List[str], 
                        summary_type: str = "general",
                        max_words: int = 500,
                        progress_callback=None) -> str:
        """Generate summary using selected representative chunks with Groq Llama3."""
        if progress_callback:
            progress_callback("Generating summary...")
        
        print(f"🔄 Generating {summary_type} summary with Groq Llama3...")
        
        combined_text = "\n\n".join(representative_chunks)
        
        prompts = {
            "general": f"Please provide a comprehensive summary of the following document content in approximately {max_words} words. Focus on the main themes, key points, and overall structure:",
            "key_findings": f"Extract and summarize the key findings, conclusions, and important insights from the following document content in approximately {max_words} words:",
            "risks": f"Identify and summarize the main risks, challenges, and potential issues mentioned in the following document content in approximately {max_words} words:",
            "legal_clauses": f"Extract and summarize the important legal clauses, terms, and conditions from the following document content in approximately {max_words} words:"
        }
        
        system_message = f"""You are an expert document summarization assistant. {prompts.get(summary_type, prompts['general'])}

Guidelines:
- Be concise but comprehensive
- Maintain the logical flow of information
- Highlight the most important points
- Use clear, professional language
- Structure the summary with clear paragraphs
- Stay within the word limit of approximately {max_words} words"""

        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": combined_text}
                ],
                model=self.llm_model,
                temperature=0.3,
                max_tokens=max_words * 2,
                top_p=0.9,
                stream=False
            )
            
            summary = chat_completion.choices[0].message.content
            
            print(f"✅ Summary generated successfully with Groq Llama3")
            return summary
            
        except Exception as e:
            raise Exception(f"Error generating summary with Groq: {str(e)}")
    
    def visualize_clustering_metrics(self, optimization_results: Dict) -> plt.Figure:
        """Create clustering metrics visualization and return the figure."""
        detailed_results = optimization_results['detailed_results']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Multi-Metric Clustering Optimization', fontsize=16, fontweight='bold')
        
        k_values = [r['k'] for r in detailed_results]
        optimal_k = next(r['k'] for r in detailed_results if r['is_optimal'])
        
        # Raw metrics
        ax1 = axes[0, 0]
        ax1.plot(k_values, [r['silhouette'] for r in detailed_results], 'o-', label='Silhouette Score', linewidth=2)
        ax1.axvline(optimal_k, color='red', linestyle='--', alpha=0.7, label=f'Optimal k={optimal_k}')
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Silhouette Score')
        ax1.set_title('Silhouette Score vs k')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[0, 1]
        ax2.plot(k_values, [r['calinski_harabasz'] for r in detailed_results], 'o-', color='orange', label='Calinski-Harabasz Index', linewidth=2)
        ax2.axvline(optimal_k, color='red', linestyle='--', alpha=0.7, label=f'Optimal k={optimal_k}')
        ax2.set_xlabel('Number of Clusters (k)')
        ax2.set_ylabel('Calinski-Harabasz Index')
        ax2.set_title('Calinski-Harabasz Index vs k')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3 = axes[1, 0]
        ax3.plot(k_values, [r['davies_bouldin'] for r in detailed_results], 'o-', color='green', label='Davies-Bouldin Index', linewidth=2)
        ax3.axvline(optimal_k, color='red', linestyle='--', alpha=0.7, label=f'Optimal k={optimal_k}')
        ax3.set_xlabel('Number of Clusters (k)')
        ax3.set_ylabel('Davies-Bouldin Index')
        ax3.set_title('Davies-Bouldin Index vs k (Lower is Better)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Distance plot for elbow detection
        ax4 = axes[1, 1]
        distances = [r['distance'] for r in detailed_results]
        colors = ['red' if r['is_optimal'] else 'blue' for r in detailed_results]
        bars = ax4.bar(k_values, distances, color=colors, alpha=0.7)
        ax4.set_xlabel('Number of Clusters (k)')
        ax4.set_ylabel('Distance from Line (Elbow Detection)')
        ax4.set_title('Elbow Detection Analysis')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, distance in zip(bars, distances):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + max(distances)*0.01,
                    f'{distance:.2f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        return fig
    
    def visualize_chunk_selection(self, selection_info: Dict) -> plt.Figure:
        """Create chunk selection visualization and return the figure."""
        cluster_selections = selection_info['cluster_selections']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Adaptive Chunk Selection Analysis', fontsize=16, fontweight='bold')
        
        # Cluster sizes vs selected chunks
        cluster_ids = list(cluster_selections.keys())
        cluster_sizes = [info['cluster_size'] for info in cluster_selections.values()]
        selected_counts = [info['num_selected'] for info in cluster_selections.values()]
        weights = [info['cluster_weight'] for info in cluster_selections.values()]
        
        x = np.arange(len(cluster_ids))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, cluster_sizes, width, label='Total Chunks', alpha=0.7)
        bars2 = ax1.bar(x + width/2, selected_counts, width, label='Selected Chunks', alpha=0.7)
        
        ax1.set_xlabel('Cluster ID')
        ax1.set_ylabel('Number of Chunks')
        ax1.set_title('Cluster Sizes vs Selected Chunks')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'C{i}' for i in cluster_ids])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(height)}', ha='center', va='bottom')
        
        # Cluster weights
        bars3 = ax2.bar(cluster_ids, weights, alpha=0.7, color='green')
        ax2.set_xlabel('Cluster ID')
        ax2.set_ylabel('Cluster Weight')
        ax2.set_title('Cluster Importance Weights')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, weight in zip(bars3, weights):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(weights)*0.01,
                    f'{weight:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    def summarize_document(self, 
                          document_file=None,
                          plain_text: str = "",
                          youtube_url: str = "",
                          total_representative_chunks: int = 8,
                          summary_type: str = "general",
                          max_words: int = 500,
                          k_range: Tuple[int, int] = (2, 10),
                          metric_weights: Dict[str, float] = None,
                          min_silhouette: float = 0.0,
                          size_weight: float = 0.7,
                          spread_weight: float = 0.3,
                          global_top_k: int = 2,
                          hybrid: bool = True,
                          use_cache: bool = True,
                          progress_callback=None) -> Dict[str, Any]:
        """
        Complete enhanced document summarization pipeline for multiple input types.
        
        Args:
            document_file: Uploaded document file
            plain_text: Plain text input
            youtube_url: YouTube video URL
            total_representative_chunks: Total number of representative chunks to select
            summary_type: Type of summary to generate
            max_words: Maximum words in summary
            k_range: Range of k values to test for clustering
            metric_weights: Weights for clustering metrics
            min_silhouette: Minimum silhouette score threshold
            size_weight: Weight for cluster size importance (α)
            spread_weight: Weight for cluster spread importance (β)
            global_top_k: Number of globally closest chunks to include in hybrid mode
            hybrid: Whether to use hybrid chunk selection
            use_cache: Whether to use cached results
            progress_callback: Function to call for progress updates
            
        Returns:
            Dictionary with summary and detailed metadata
        """
        try:
            print("🚀 Starting enhanced document summarization pipeline...")
            
            # Step 1: Process multiple inputs
            text_chunks = self.process_multiple_inputs(
                document_file, plain_text, youtube_url, progress_callback
            )
            
            # Step 2: Generate embeddings
            embeddings = self.generate_embeddings(text_chunks, use_cache, progress_callback)
            
            # Step 3: Find optimal clusters using multi-metric approach
            optimal_k, optimization_results = self.find_optimal_clusters_multi_metric(
                embeddings, k_range, metric_weights, min_silhouette, progress_callback=progress_callback
            )
            
            # Step 4: Get the optimal clustering results
            cache_key = f"k_{optimal_k}"
            if cache_key in self.cache['clustering_results']:
                kmeans = self.cache['clustering_results'][cache_key]['kmeans']
                cluster_labels = self.cache['clustering_results'][cache_key]['labels']
            else:
                # This shouldn't happen, but just in case
                kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(embeddings)
            
            # Step 5: Adaptive representative chunk selection
            representative_chunks, selection_info = self.adaptive_chunk_selection(
                embeddings, text_chunks, kmeans, cluster_labels,
                total_representative_chunks, size_weight, spread_weight,
                global_top_k, hybrid
            )
            
            # Step 6: Generate summary
            summary = self.generate_summary(representative_chunks, summary_type, max_words, progress_callback)
            
            # Step 7: Prepare comprehensive results
            results = {
                "summary": summary,
                "metadata": {
                    "total_chunks": len(text_chunks),
                    "optimal_k": optimal_k,
                    "total_representative_chunks": len(representative_chunks),
                    "adaptive_chunks": selection_info['adaptive_selected'],
                    "global_chunks": selection_info['global_selected'],
                    "summary_type": summary_type,
                    "hybrid_mode": hybrid,
                    "compression_ratio": f"{len(representative_chunks)}/{len(text_chunks)} ({len(representative_chunks)/len(text_chunks)*100:.1f}%)",
                    "clustering_method": "multi_metric_ranking",
                    "selection_method": "adaptive_hybrid" if hybrid else "adaptive"
                },
                "clustering_optimization": {
                    "tested_k_range": k_range,
                    "optimal_k": optimal_k,
                    "metric_weights_used": metric_weights or {"silhouette": 1/3, "calinski_harabasz": 1/3, "davies_bouldin": 1/3},
                    "min_silhouette_threshold": min_silhouette,
                    "detailed_results": optimization_results['detailed_results']
                },
                "chunk_selection": {
                    "total_requested": total_representative_chunks,
                    "total_selected": selection_info['total_selected'],
                    "size_weight": size_weight,
                    "spread_weight": spread_weight,
                    "global_top_k": global_top_k,
                    "hybrid_enabled": hybrid,
                    "cluster_breakdown": selection_info['cluster_selections'],
                    "detailed_selections": selection_info['detailed_selections']
                },
                "optimization_results": optimization_results,
                "selection_info": selection_info
            }
            
            print("✅ Enhanced document summarization completed successfully!")
            return results
            
        except Exception as e:
            print(f"❌ Error in summarization pipeline: {str(e)}")
            return {"error": str(e)}