import streamlit as st
import os
from dotenv import load_dotenv
import time
import pandas as pd
import matplotlib.pyplot as plt
from summarizer import EnhancedDocumentSummarizer

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Enhanced Document Summarizer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'summarization_results' not in st.session_state:
        st.session_state.summarization_results = None
    if 'summarizer' not in st.session_state:
        st.session_state.summarizer = None

def create_progress_callback():
    """Create a progress callback function for the summarizer."""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def update_progress(message):
        steps = {
            "Loading document...": 10,
            "Processing plain text...": 10,
            "Getting YouTube transcript...": 10,
            "Generating embeddings...": 30,
            "Finding optimal clusters...": 60,
            "Generating summary...": 90
        }
        
        progress = steps.get(message, 0)
        progress_bar.progress(progress)
        status_text.text(f"🔄 {message}")
        
        if "summary" in message.lower():
            time.sleep(1)  # Give a moment to see the final step
            progress_bar.progress(100)
            status_text.text("✅ Completed!")
    
    return update_progress

def display_summary_results(results):
    """Display the summarization results."""
    if "error" in results:
        st.error(f"❌ Error: {results['error']}")
        return
    
    # Main summary
    st.markdown("## 📝 Summary")
    st.markdown("---")
    
    # Summary type badge
    summary_type = results['metadata']['summary_type'].title()
    st.markdown(f"**Summary Type:** `{summary_type}`")
    st.markdown(f"**Word Count:** ~{results['metadata'].get('word_count', 'N/A')} words")
    
    # Display the summary in a nice box
    st.markdown(
        f"""
        <div style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #28a745;">
        {results['summary']}
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Key metrics
    st.markdown("## 📊 Processing Metrics")
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Chunks", results['metadata']['total_chunks'])
    
    with col2:
        st.metric("Optimal Clusters", results['metadata']['optimal_k'])
    
    with col3:
        st.metric("Representative Chunks", results['metadata']['total_representative_chunks'])
    
    with col4:
        compression_ratio = results['metadata']['compression_ratio']
        st.metric("Compression Ratio", compression_ratio)

def display_analytics(results):
    """Display comprehensive analytics showing the complete problem-solving process."""
    if "error" in results:
        st.error(f"❌ No analytics available due to error: {results['error']}")
        return
    
    st.markdown("# 📊 Processing Analytics")
    st.markdown("**Complete walkthrough of how we solved the document summarization problem**")
    st.markdown("---")
    
    # STEP 1: PROBLEM & INPUT ANALYSIS
    st.markdown("## 🎯 STEP 1: Problem & Input Analysis")
    st.markdown("**What we received and how we processed it**")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### Input Processing")
        metadata = results['metadata']
        st.write(f"📄 **Total document chunks created:** {metadata['total_chunks']}")
        st.write(f"🎯 **Target summary type:** {metadata['summary_type'].title()}")
        st.write(f"📝 **Requested summary length:** ~500 words")
        st.write(f"🔧 **Processing method:** Multi-source combination")
        
        st.markdown("### The Challenge")
        st.info(f"""
        **Problem:** From {metadata['total_chunks']} text chunks, select the most representative ones for summarization.
        
        **Why this matters:** Simply taking the first few chunks or random sampling might miss important information scattered throughout the document.
        """)
    
    with col2:
        st.markdown("### Quick Stats")
        st.metric("Input Chunks", metadata['total_chunks'])
        st.metric("Target Representatives", metadata['total_representative_chunks'])
        compression_pct = (metadata['total_representative_chunks'] / metadata['total_chunks']) * 100
        st.metric("Compression", f"{compression_pct:.1f}%")
    
    st.markdown("---")
    
    # STEP 2: CLUSTERING OPTIMIZATION
    st.markdown("## 🧮 STEP 2: Clustering Optimization")
    st.markdown("**Finding the optimal number of clusters to group similar content**")
    
    st.markdown("### Why Clustering?")
    st.info("""
    **Strategy:** Group similar chunks together, then pick representatives from each group.
    This ensures we capture diverse information rather than redundant content.
    """)
    
    # Clustering details
    clustering = results['clustering_optimization']
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Optimization Process")
        st.write(f"🔍 **Tested k range:** {clustering['tested_k_range'][0]} to {clustering['tested_k_range'][1]} clusters")
        st.write(f"🎯 **Chosen k:** {clustering['optimal_k']} clusters")
        st.write(f"⚖️ **Selection method:** Multi-metric scoring with elbow detection")
    
    with col2:
        st.markdown("### Scoring Metrics")
        st.write("🔹 **Silhouette Score:** How well-separated clusters are (higher = better)")
        st.write("🔹 **Calinski-Harabasz:** Cluster density ratio (higher = better)")
        st.write("🔹 **Davies-Bouldin:** Within-cluster scatter (lower = better)")
        st.write("🔹 **Penalty:** Prevents over-clustering")
    
    # Detailed k comparison table
    st.markdown("### Detailed K-Value Analysis")
    st.markdown("*How we evaluated each possible number of clusters:*")
    
    # Create the detailed table
    detailed_results = clustering['detailed_results']
    table_data = []
    
    for result in detailed_results:
        row = {
            'k': result['k'],
            'Silhouette': f"{result['silhouette']:.3f}",
            'Calinski-H': f"{result['calinski_harabasz']:.0f}",
            'Davies-B': f"{result['davies_bouldin']:.3f}",
            'Composite': f"{result['composite_score']:.3f}",
            'Penalty': f"{result['penalty']:.3f}",
            'Final Score': f"{result['penalized_score']:.3f}",
            'Elbow Distance': f"{result['distance']:.3f}",
            'Selected': "✅ OPTIMAL" if result['is_optimal'] else ""
        }
        table_data.append(row)
    
    df_clustering = pd.DataFrame(table_data)
    st.dataframe(df_clustering, use_container_width=True)
    
    # Explanation of choice
    optimal_result = next(r for r in detailed_results if r['is_optimal'])
    st.success(f"""
    **Why k={optimal_result['k']} was chosen:**
    - Highest elbow distance ({optimal_result['distance']:.3f}) indicating best balance
    - Good silhouette score ({optimal_result['silhouette']:.3f}) showing well-separated clusters  
    - Reasonable penalty ({optimal_result['penalty']:.3f}) avoiding over-clustering
    """)
    
    st.markdown("---")
    
    # STEP 3: ADAPTIVE CHUNK SELECTION
    st.markdown("## 🎨 STEP 3: Adaptive Chunk Selection")
    st.markdown("**Smart selection of representative chunks from each cluster**")
    
    st.markdown("### Selection Strategy")
    chunk_selection = results['chunk_selection']
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### Algorithm Parameters")
        st.write(f"⚖️ **Size Weight (α):** {chunk_selection['size_weight']} - *Favor larger clusters*")
        st.write(f"📊 **Spread Weight (β):** {chunk_selection['spread_weight']} - *Consider cluster diversity*")
        st.write(f"🌐 **Global Top-K:** {chunk_selection['global_top_k']} - *Add globally best chunks*")
        st.write(f"🔄 **Hybrid Mode:** {chunk_selection['hybrid_enabled']} - *Combine strategies*")
    
    with col2:
        st.markdown("### Selection Results")
        st.write(f"🎯 **Requested:** {chunk_selection['total_requested']} chunks")
        st.write(f"✅ **Selected:** {chunk_selection['total_selected']} chunks")
        st.write(f"🎨 **Adaptive:** {results['metadata']['adaptive_chunks']} chunks")
        st.write(f"🌐 **Global:** {results['metadata']['global_chunks']} chunks")
    
    st.markdown("### How We Distributed Selections Across Clusters")
    st.info("""
    **Smart Distribution:** Each cluster gets representatives based on:
    - **Size:** Larger clusters get more representatives (they contain more information)
    - **Spread:** Clusters with diverse content get more representatives  
    - **Global Quality:** We also add the globally best chunks regardless of cluster
    """)
    
    # Cluster breakdown table
    cluster_breakdown = chunk_selection['cluster_breakdown']
    cluster_data = []
    
    for cluster_id, info in cluster_breakdown.items():
        cluster_data.append({
            'Cluster': f"C{cluster_id}",
            'Total Chunks': info['cluster_size'],
            'Selected': info['num_selected'],
            'Selection %': f"{(info['num_selected']/info['cluster_size'])*100:.1f}%",
            'Weight Score': f"{info['cluster_weight']:.3f}",
            'Why Selected': f"Size: {info['cluster_size']}, Weight: {info['cluster_weight']:.2f}"
        })
    
    df_clusters = pd.DataFrame(cluster_data)
    st.dataframe(df_clusters, use_container_width=True)
    
    # Global selections if any
    global_selections = [s for s in chunk_selection['detailed_selections'] if s['selection_method'] == 'global_top_k']
    if global_selections:
        st.markdown("### Additional Global Selections")
        st.write("*These chunks were added because they were globally closest to cluster centers:*")
        
        global_data = []
        for i, sel in enumerate(global_selections, 1):
            global_data.append({
                'Selection': f"#{i}",
                'From Cluster': f"C{sel['cluster_id']}",
                'Distance Score': f"{sel['distance_to_centroid']:.3f}",
                'Reason': "Globally representative content"
            })
        
        df_global = pd.DataFrame(global_data)
        st.dataframe(df_global, use_container_width=True)
    
    st.markdown("---")
    
    # STEP 4: FINAL RESULTS
    st.markdown("## 📈 STEP 4: Final Results & Performance")
    st.markdown("**What we achieved and how efficient our solution was**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Compression Achieved")
        original_chunks = metadata['total_chunks']
        selected_chunks = metadata['total_representative_chunks']
        compression_ratio = selected_chunks / original_chunks
        st.metric("Compression Ratio", f"{compression_ratio:.1%}")
        st.write(f"📉 Reduced from {original_chunks} to {selected_chunks} chunks")
        st.write(f"💾 **Information Efficiency:** Captured diverse content with {(1-compression_ratio)*100:.1f}% reduction")
    
    with col2:
        st.markdown("### Quality Indicators")
        st.metric("Optimal Clusters", metadata['optimal_k'])
        st.metric("Selection Method", metadata['selection_method'].replace('_', ' ').title())
        st.write(f"🎯 **Strategy:** {metadata['clustering_method'].replace('_', ' ').title()}")
        st.write(f"🔧 **Hybrid Mode:** {'Enabled' if results['metadata'].get('hybrid_mode', False) else 'Disabled'}")
    
    with col3:
        st.markdown("### Summary Output")
        word_count = results['metadata'].get('word_count', 'Estimated 400-600')
        st.metric("Summary Type", metadata['summary_type'].title())
        st.write(f"📝 **Word Count:** ~{word_count} words")
        st.write(f"🎨 **Content Coverage:** Multi-cluster representation")
        st.write(f"✅ **Quality:** Diverse, non-redundant summary")
    
    # Final summary
    st.markdown("### 🎯 Solution Summary")
    st.success(f"""
    **Our Approach Successfully:**
    1. 📊 **Analyzed** {metadata['total_chunks']} input chunks from multiple sources
    2. 🧮 **Optimized** clustering using multi-metric scoring (found k={metadata['optimal_k']})
    3. 🎨 **Selected** {metadata['total_representative_chunks']} representative chunks using adaptive weighting
    4. 📝 **Generated** a {metadata['summary_type']} summary with {compression_ratio:.1%} compression ratio
    
    **Result:** High-quality summary that captures diverse information while avoiding redundancy.
    """)

def main():
    """Main Streamlit application."""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">📄 Enhanced Document Summarizer</h1>', unsafe_allow_html=True)
    st.markdown(
        "**Multi-input document summarization with advanced clustering and analytics**\n\n"
        "Upload documents, paste text, or provide YouTube links for intelligent summarization."
    )
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # API Key configuration
        st.subheader("API Settings")
        groq_api_key = st.text_input(
            "Groq API Key (Optional)",
            type="password",
            help="If not provided, will use the key from .env file"
        )
        
        # Summary configuration
        st.subheader("Summary Settings")
        summary_type = st.selectbox(
            "Summary Type",
            ["general", "key_findings", "risks", "legal_clauses"],
            help="Type of summary to generate"
        )
        
        max_words = st.slider(
            "Maximum Words",
            min_value=100,
            max_value=1000,
            value=500,
            step=50,
            help="Maximum number of words in the summary"
        )
        
        total_chunks = st.slider(
            "Representative Chunks",
            min_value=3,
            max_value=20,
            value=8,
            help="Number of representative chunks to select for summarization"
        )
        
        # Advanced settings (collapsed)
        with st.expander("Advanced Settings"):
            k_min = st.number_input("Min Clusters (K)", min_value=2, max_value=10, value=2)
            k_max = st.number_input("Max Clusters (K)", min_value=5, max_value=50, value=15)
    
    # Main content area with tabs
    main_tabs = st.tabs(["📝 Summarization", "📊 Analytics"])
    
    # Tab 1: Summarization
    with main_tabs[0]:
        st.markdown("## Input Sources")
        st.markdown("Provide one or more input sources below:")
        
        # Input sections
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📁 Document Upload")
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=['pdf', 'docx'],
                help="Upload a PDF or DOCX document"
            )
            
            st.markdown("### 📺 YouTube Video")
            youtube_url = st.text_input(
                "YouTube URL",
                placeholder="https://www.youtube.com/watch?v=...",
                help="Provide a YouTube video URL to extract and summarize the transcript"
            )
        
        with col2:
            st.markdown("### 📝 Plain Text")
            plain_text = st.text_area(
                "Enter text directly",
                height=200,
                placeholder="Paste your text content here...",
                help="Enter any text content you want to summarize"
            )
        
        # Summarization button
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.button("🚀 Generate Summary", type="primary", use_container_width=True):
                # Validate inputs
                if not uploaded_file and not plain_text.strip() and not youtube_url.strip():
                    st.error("⚠️ Please provide at least one input source!")
                    return
                
                # Initialize summarizer
                try:
                    api_key = groq_api_key if groq_api_key else os.getenv("GROQ_API_KEY")
                    if not api_key:
                        st.error("❌ No Groq API key provided! Please set it in .env file or enter it in the sidebar.")
                        return
                    
                    # Create progress callback
                    progress_callback = create_progress_callback()
                    
                    # Initialize summarizer
                    st.session_state.summarizer = EnhancedDocumentSummarizer(groq_api_key=api_key)
                    
                    # Run summarization
                    with st.spinner("Processing your inputs..."):
                        results = st.session_state.summarizer.summarize_document(
                            document_file=uploaded_file,
                            plain_text=plain_text,
                            youtube_url=youtube_url,
                            total_representative_chunks=total_chunks,
                            summary_type=summary_type,
                            max_words=max_words,
                            k_range=(k_min, k_max),
                            progress_callback=progress_callback
                        )
                    
                    # Store results in session state
                    st.session_state.summarization_results = results
                    
                    # Clear progress indicators
                    st.empty()
                    
                except Exception as e:
                    st.error(f"❌ Error initializing summarizer: {str(e)}")
                    return
        
        # Display results if available
        if st.session_state.summarization_results:
            st.markdown("---")
            display_summary_results(st.session_state.summarization_results)
    
    # Tab 2: Analytics
    with main_tabs[1]:
        if st.session_state.summarization_results:
            display_analytics(st.session_state.summarization_results)
        else:
            st.info("📊 Analytics will appear here after you generate a summary.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; padding: 2rem;'>"
        "Enhanced Document Summarizer with Multi-Metric Clustering<br>"
        "Built with Streamlit, Groq, and Advanced NLP Techniques"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()