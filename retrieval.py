import requests
from typing import List, Dict
import torch
def get_similar_documents(query: str, model, supabase_url: str, supabase_key: str, top_k: int = 1) -> List[Dict]:
    """Retrieve similar documents from Supabase using vector search"""
    try:
        # Generate embedding and normalize like in new.py
        embedding = model.encode(query)
        embedding_tensor = torch.tensor(embedding).to(torch.float32)
        normalized_embedding = embedding_tensor / torch.norm(embedding_tensor, p=2)
        
        headers = {
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "query_embedding": normalized_embedding.tolist(),
            "match_count": top_k,
        }

        # Changed endpoint to match_db1 from new.py
        response = requests.post(
            f"{supabase_url}/rest/v1/rpc/match_db1",
            headers=headers,
            json=payload,
            timeout=600,
        )

        response.raise_for_status()

        results = response.json()
        return [
            {
                "id": doc.get("id"),
                "similarity": doc.get("similarity"),  
                "content": doc.get("content", "")
            } 
            for doc in results
        ]
    
    except requests.exceptions.HTTPError as e:
        raise Exception(f"Supabase API Error: {e.response.status_code} - {e.response.text}")
    except Exception as e:
        raise Exception(f"Retrieval failed: {str(e)}")