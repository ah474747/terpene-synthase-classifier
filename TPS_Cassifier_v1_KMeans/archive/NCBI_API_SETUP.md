# NCBI API Setup for Large-Scale Data Collection

## Overview

To efficiently download ~60,000 terpene synthase sequences from NCBI, we need to comply with their API guidelines and use proper rate limiting.

## API Key Setup (Recommended)

### 1. Get an NCBI API Key

1. **Create NCBI Account**: Go to https://www.ncbi.nlm.nih.gov/account/
2. **Sign in** to your account
3. **Navigate to Account Settings**: https://www.ncbi.nlm.nih.gov/account/settings/
4. **Find "API Key Management"** section
5. **Click "Create API Key"**
6. **Copy the generated key** and keep it secure

### 2. Benefits of API Key

- **Rate Limit**: Increases from 3 requests/second to 10 requests/second
- **Priority**: Higher priority during peak usage
- **Monitoring**: Better tracking of your usage

## Rate Limiting Guidelines

### Without API Key
- **Limit**: 3 requests per second
- **Delay**: 0.33 seconds between requests

### With API Key
- **Limit**: 10 requests per second  
- **Delay**: 0.1 seconds between requests

## Required Parameters

All NCBI API calls must include:
- `email`: Your email address (required)
- `tool`: Tool name for identification (required)
- `api_key`: Your API key (optional but recommended)

## Query Optimization

### Proper Search Syntax
```
"terpene synthase"[Title] OR "terpene synthase"[Abstract]
```

### Batch Processing
- Use `retmax=10000` for large result sets
- Process in batches of 100-500 sequences
- Implement pagination with `retstart` parameter

## Updated Script Usage

### 1. Edit the Script
```python
collector = TerpeneSynthaseDataCollector(
    email="your_email@example.com",  # Your actual email
    api_key="your_api_key_here",     # Your NCBI API key
    tool_name="terpene_classifier_phase2"
)
```

### 2. Run with Higher Limits
```python
# For 60,000 sequences, use higher limits
raw_sequences = collector.collect_all_sequences(max_results_per_term=10000)
```

## Time Estimates

### For ~60,000 sequences:

**Without API Key (3 req/sec):**
- Search requests: ~7 requests = 2.3 seconds
- Fetch requests: ~600 requests = 200 seconds
- **Total time**: ~3.5 minutes

**With API Key (10 req/sec):**
- Search requests: ~7 requests = 0.7 seconds  
- Fetch requests: ~600 requests = 60 seconds
- **Total time**: ~1 minute

## Best Practices

1. **Run during off-peak hours**: 9 PM - 5 AM Eastern Time
2. **Monitor your usage**: Check NCBI's usage statistics
3. **Handle errors gracefully**: Implement retry logic
4. **Respect rate limits**: Don't exceed the limits
5. **Use proper query syntax**: Be specific in your searches

## Troubleshooting

### Common Issues

1. **"Too many requests"**: Reduce rate or get API key
2. **"Invalid query"**: Check search term syntax
3. **"No results"**: Try broader search terms
4. **"Connection timeout"**: Implement retry logic

### Error Handling

The updated script includes:
- Proper rate limiting
- Error handling for failed requests
- Progress tracking
- Graceful degradation

## Example Usage

```bash
# Run the updated collector
python3 phase2_data_collector.py

# The script will:
# 1. Check for API key
# 2. Apply proper rate limiting
# 3. Use optimized queries
# 4. Process in efficient batches
# 5. Save results incrementally
```

## Next Steps

1. **Get your API key** from NCBI
2. **Update the script** with your email and API key
3. **Run the collection** during off-peak hours
4. **Monitor progress** and adjust as needed
5. **Process results** with the processing script
