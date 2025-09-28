# Multilingual-RAG-Chatbot-using-ChromaDB
Production-ready multilingual RAG system with Bengali-English translation, ChromaDB vector storage, and Alibaba embeddings for intelligent product search and query processing.
# Multilingual RAG with ChromaDB and Alibaba Embeddings

A production-ready Retrieval-Augmented Generation (RAG) system designed for multilingual product queries, featuring Bengali-English translation, intent extraction, and semantic search capabilities.

## 🌟 Features

- **Multilingual Support**: Native Bengali and English query processing with automatic translation
- **Advanced Intent Extraction**: Identifies user intent, product categories, and key terms
- **Custom Embedding Integration**: Uses Alibaba's `gte-multilingual-base` model for superior multilingual embeddings
- **Multiple Search Strategies**: Combines semantic search, keyword extraction, and intent-based filtering
- **Product Catalog Management**: Structured parsing of product data with category organization
- **ChromaDB Vector Storage**: Persistent vector database with custom embedding functions

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Query    │───▶│   Translation    │───▶│ Intent Analysis │
│  (Bengali/EN)   │    │   & Processing   │    │  & Extraction   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Response      │◄───│  LLM Generation  │◄───│ Multi-Strategy  │
│   Generation    │    │   (Groq/Llama)   │    │    Search       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
                                                ┌─────────────────┐
                                                │   ChromaDB      │
                                                │   + Alibaba     │
                                                │   Embeddings    │
                                                └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install python-dotenv yake langchain-groq chromadb langdetect nltk sentence-transformers torch
```

### Environment Setup

Create a `.env` file:
```env
GROQ_API_KEY=your_groq_api_key_here
```

### Data Format

Your product data should be in this format:
```
# Electronics & Technology
[Smartphone Pro Max](https://example.com/phone) Product Description: Latest flagship smartphone with advanced features) price:45000

# Beauty & Personal Care  
[Premium Face Cream](https://example.com/cream) Product Description: Anti-aging cream with natural ingredients) price:2500
```

### Running the System

```python
from multilingual_rag import *

# Initialize the system
docs, content = load_data_file()  # Load your products.txt
collection = build_chroma_db(docs)  # Build vector database

# Query the system
answer = answer_bengali_custom("ভালো সাবান কি আছে?")
print(answer)
```

## 🔧 Configuration

### Translation Toggle
```python
USE_NLP_TRANSLATION = True  # Enable/disable LLM-based translation
```

### Custom Embedding Model
The system uses `Alibaba-NLP/gte-multilingual-base` by default. To change:
```python
embedding_function = AlibabaEmbeddingFunction("your-preferred-model")
```

### Search Parameters
```python
# Adjust search behavior
TOP_K = 6  # Number of results to retrieve
BATCH_SIZE = 32  # Processing batch size
```

## 📊 System Components

### 1. **AlibabaEmbeddingFunction**
Custom embedding wrapper for ChromaDB integration with multilingual support.

### 2. **Multi-Strategy Search**
- **Semantic Search**: Vector similarity using embeddings
- **Intent-Based Search**: Category filtering with extracted intent
- **Keyword Search**: YAKE-based keyword extraction and matching
- **Cross-lingual Search**: Searches in both source and translated languages

### 3. **Intent Analysis**
Extracts structured information from queries:
- Main intent classification
- Product category mapping
- Action type identification
- Target user profiling
- Key term extraction

### 4. **Product Categories**
Supports 13+ predefined categories:
- Electronics & Technology
- Beauty & Personal Care
- Fashion & Clothing
- Home & Kitchen Appliances
- And more...

## 📈 Performance Features

- **Batch Processing**: Efficient document processing in configurable batches
- **Persistent Storage**: ChromaDB maintains embeddings between sessions
- **Memory Management**: Optimized for large product catalogs
- **Error Handling**: Comprehensive error recovery and logging

## 🛠️ Advanced Usage

### Custom Search Implementation
```python
def custom_search_strategy(query, filters=None):
    results = collection.query(
        query_texts=[query],
        n_results=10,
        where=filters
    )
    return process_results(results)
```

### Adding New Product Categories
```python
VALID_CATEGORIES = [
    "Electronics & Technology",
    "Your Custom Category",
    # Add more categories
]
```

### Integration with Different LLMs
```python
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="your-preferred-model",
    temperature=0.0
)
```

## 📋 API Reference

### Core Functions

- `load_data_file()`: Load and parse product data
- `build_chroma_db(docs)`: Initialize vector database
- `answer_bengali_custom(query)`: Main query processing function
- `translate_to_en(text)`: Bengali to English translation
- `extract_intent(text_bn, text_en)`: Intent and entity extraction
- `extract_keywords(text)`: Keyword extraction using YAKE

### Utility Functions

- `parse_product_line(line)`: Parse individual product entries
- `search_chromadb_custom(query, method, top_k)`: Vector search interface

## 🔍 Example Queries

```python
# Bengali queries
answer_bengali_custom("ভালো ফোন কি আছে?")
answer_bengali_custom("বই কি আছে পড়ার?")

# English queries work too
answer_bengali_custom("What smartphones are available?")
```

## 🏷️ Product Data Schema

```python
{
    "id": "unique_uuid",
    "name": "Product Name",
    "url": "product_url",
    "description": "Product description",
    "price": 12345,
    "category": "Product Category",
    "text": "Searchable text representation"
}
```

## 🐛 Troubleshooting

### Common Issues

1. **ChromaDB Error**: Ensure proper embedding function initialization
2. **Translation Failures**: Check Groq API key and model availability
3. **Memory Issues**: Reduce batch size for large datasets
4. **Search Quality**: Adjust embedding model or search parameters

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.INFO)
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Alibaba-NLP](https://huggingface.co/Alibaba-NLP) for the multilingual embedding model
- [ChromaDB](https://www.trychroma.com/) for vector database capabilities
- [Groq](https://groq.com/) for fast LLM inference
- [YAKE](https://github.com/LIAAD/yake) for keyword extraction

## 📞 Support

For support, please open an issue on GitHub or contact the maintainers.

---

**Built with ❤️ for multilingual AI applications**
